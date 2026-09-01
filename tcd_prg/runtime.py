"""Construction helpers shared by training, evaluation, and cache tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcd_prg.config import TCDPRGConfig
from tcd_prg.datasets import TaskOrientedClutterAdapter, collate_unified
from tcd_prg.datasets.collate import collate_global_grasp
from tcd_prg.execution import ExternalFR5AG16095Certifier
from tcd_prg.observation import (
    CachedObservationProvider,
    ExternalPyBulletObservationProvider,
    SavedObservationProvider,
)
from tcd_prg.paths import project_path


def create_observation_provider(config: TCDPRGConfig, allow_render: bool = False):
    scene_root = Path(config.dataset.root) / config.dataset.scene_subdir
    if config.observation.provider == "saved":
        return SavedObservationProvider(
            scene_root,
            scene_root / "metadata.json",
            config.dataset.scene_points,
            width=config.observation.render_width,
            height=config.observation.render_height,
        )

    def external_provider() -> ExternalPyBulletObservationProvider:
        return ExternalPyBulletObservationProvider(
            config.observation.pybullet_python,
            project_path(config.observation.worker_script),
            scene_root,
            config.observation.runtime_mesh_root,
            config.observation.render_width,
            config.observation.render_height,
            config.observation.render_temporary_root,
        )

    if config.observation.provider == "rendered":
        return external_provider()
    if config.observation.provider != "cached":
        raise ValueError(f"Unknown observation provider {config.observation.provider}")
    # Rendering is an explicit caller capability. Formal training and bounded
    # prefetch use a renderer as cache fallback; evaluation remains cache-only.
    fallback = external_provider() if allow_render else None
    return CachedObservationProvider(
        config.cache.directory,
        fallback,
        max_bytes=int(config.cache.max_gb * (1 << 30)),
        min_free_bytes=int(config.cache.min_free_gb * (1 << 30)),
    )


def create_adapter(config: TCDPRGConfig, allow_render: bool = False):
    if config.dataset.adapter != "task_oriented_clutter":
        raise ValueError(f"No registered adapter named {config.dataset.adapter}")
    # Legacy cached observations were keyed with point_count=0 (the complete
    # rendered cloud).  Keep that request contract unchanged and apply the
    # configured model point budget in the collator after the cache read.
    # Otherwise changing dataset.scene_points would turn every strict-cache
    # lookup into a miss and incorrectly require regenerating the cache.
    cache_point_count = (
        0 if config.observation.provider == "cached" else config.dataset.scene_points
    )
    return TaskOrientedClutterAdapter(
        config.dataset.root,
        observation_provider=create_observation_provider(config, allow_render),
        point_count=cache_point_count,
        renderer_version=config.observation.renderer_version,
        camera_profile=config.observation.camera_profile,
        functional_region_root=config.dataset.functional_region_root,
        verifier_wrong_region_negatives=config.sampling.wrong_region_grasps,
        verifier_collision_negatives=config.sampling.collision_or_approach_negative_grasps,
        verifier_approach_negatives=config.sampling.collision_or_approach_negative_grasps,
        sampling_seed=config.training.seed,
        scene_subdir=config.dataset.scene_subdir,
        step_labels_subdir=config.dataset.step_labels_subdir,
        action_labels_subdir=config.dataset.action_labels_subdir,
        global_positive_grasps_per_object=config.sampling.global_positive_grasps_per_object,
        global_negative_grasps_per_object=config.sampling.global_negative_grasps_per_object,
        grasp_width_bounds=(config.model.min_grasp_width_m, config.model.max_grasp_width_m),
        index_cache_dir=config.cache.index_directory,
        data_fraction=config.training.data_fraction,
        split_ratios=config.training.split_ratios,
        split_seed=config.training.seed,
        scene_start=config.training.scene_start,
        scene_count=config.training.scene_count,
    )


def create_action_certifier(config: TCDPRGConfig) -> ExternalFR5AG16095Certifier:
    urdf = Path(config.dataset.fr5_ag_urdf)
    return ExternalFR5AG16095Certifier(
        config.observation.pybullet_python,
        project_path(config.observation.certification_worker_script),
        urdf.parent.parent,
        config.observation.runtime_mesh_root,
        Path(config.dataset.root) / config.dataset.scene_subdir,
        config.observation.certification_temporary_root,
    )


def _apply_training_augmentation(
    config: TCDPRGConfig, batch: dict[str, Any]
) -> dict[str, Any]:
    from tcd_prg.datasets.augmentation_debug import claim_debug_batch, save_debug_batch
    from tcd_prg.datasets.pointcloud_augmentation import PointCloudAugmentation

    debug_directory = claim_debug_batch(
        config.output_dir,
        config.augmentation.debug.save_first_batches,
    )
    rgb_before = batch["rgb"].clone() if debug_directory is not None else None
    xyz_before = batch["xyz"].clone() if debug_directory is not None else None
    point_mask_before = (
        batch["point_mask"].clone() if debug_directory is not None else None
    )
    PointCloudAugmentation(config.augmentation)(batch)
    if debug_directory is not None and rgb_before is not None:
        save_debug_batch(
            debug_directory, batch, rgb_before, xyz_before, point_mask_before
        )
    return batch


@dataclass(slots=True)
class UnifiedBatchCollator:
    """Pickle-safe Windows DataLoader collator with provider-backed exact geometry."""

    config: TCDPRGConfig
    training: bool = False
    include_graspnet: bool | None = None

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        grid_size = (
            self.config.backbone.grid_size_m
            if self.config.backbone.backend == "point_transformer_v3"
            else None
        )
        batch = collate_unified(
            samples,
            grid_size_m=grid_size,
            training=self.training,
            point_count=self.config.dataset.scene_points,
            graspnet_point_count=max(
                self.config.graspnet.scene_input_points,
                self.config.graspnet.target_input_points,
            ),
            graspnet_view_index=self.config.graspnet.camera_view_index,
            include_graspnet=(
                self.config.losses.task_grasp > 0
                if self.include_graspnet is None
                else self.include_graspnet
            ),
        )
        return _apply_training_augmentation(self.config, batch) if self.training else batch


@dataclass(slots=True)
class PushValueBatchCollator:
    """Attach immutable offline Q/safety labels to the ordinary Stage-C batch."""

    config: TCDPRGConfig
    training: bool = False

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        import numpy as np
        import torch
        from tcd_prg.datasets.push_value import PushActionValueStore

        batch = UnifiedBatchCollator(
            self.config, training=self.training, include_graspnet=False
        )(samples)
        root = self.config.training.push_value_root
        if not root:
            raise RuntimeError("Stage-C Q training requires training.push_value_root")
        horizons = self.config.training.push_value_horizons
        store = PushActionValueStore(root, horizons)
        shape = (*batch["candidate_action_id"].shape, horizons)
        q = torch.full(shape, float("nan"), dtype=torch.float32)
        q_valid = torch.zeros(shape, dtype=torch.bool)
        safe = torch.zeros(shape[:2], dtype=torch.bool)
        safety_valid = torch.zeros(shape[:2], dtype=torch.bool)
        sequence = torch.zeros(shape[:2], dtype=torch.bool)
        scene_payloads: dict[int, dict[str, Any]] = {}
        for row, sample in enumerate(samples):
            scene_id = int(sample.observation.scene_id)
            if scene_id not in scene_payloads:
                scene_payloads[scene_id] = store.load_scene(scene_id)
            payload = scene_payloads[scene_id]
            action_ids = np.asarray(payload["action_id"], np.int64)
            if len(action_ids) and np.any(action_ids[1:] <= action_ids[:-1]):
                raise RuntimeError("Action-value IDs must be strictly increasing")
            candidates = np.asarray(sample.candidates.candidate_action_ids, np.int64)
            if len(action_ids):
                locations = np.searchsorted(action_ids, candidates)
                matched = locations < len(action_ids)
                matched &= action_ids[np.minimum(locations, len(action_ids) - 1)] == candidates
            else:
                locations = np.zeros_like(candidates)
                matched = np.zeros_like(candidates, dtype=bool)
            for local in np.flatnonzero(matched):
                source = int(locations[local])
                q[row, local] = torch.from_numpy(payload["q_value"][source].astype(np.float32))
                q_valid[row, local] = torch.from_numpy(payload["q_valid"][source].astype(bool))
                safe[row, local] = bool(payload["safe"][source])
                safety_valid[row, local] = bool(payload["safety_valid"][source])
                sequence[row, local] = bool(payload["part_of_success_sequence"][source])
        batch.update(
            push_q_target=q,
            push_q_valid=q_valid,
            push_safe_target=safe,
            push_safety_valid=safety_valid,
            push_part_of_success_sequence=sequence,
        )
        return batch


@dataclass(slots=True)
class StageBBinaryBatchCollator:
    config: TCDPRGConfig
    training: bool = True

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        from tcd_prg.datasets.collate import collate_stageb_binary

        grid_size = (
            self.config.backbone.grid_size_m
            if self.config.backbone.backend == "point_transformer_v3" else None
        )
        batch = collate_stageb_binary(
            samples, grid_size_m=grid_size, training=self.training,
            point_count=self.config.dataset.scene_points,
        )
        return _apply_training_augmentation(self.config, batch) if self.training else batch


@dataclass(slots=True)
class GlobalGraspBatchCollator:
    """Pickle-safe minimal collator for direct Global Grasp supervision."""

    config: TCDPRGConfig
    training: bool = True

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        grid_size = (
            self.config.backbone.grid_size_m
            if self.config.backbone.backend == "point_transformer_v3"
            else None
        )
        batch = collate_global_grasp(
            samples,
            grid_size_m=grid_size,
            training=self.training,
            point_count=self.config.dataset.scene_points,
            graspnet_point_count=max(
                self.config.graspnet.scene_input_points,
                self.config.graspnet.target_input_points,
            ),
            graspnet_view_index=self.config.graspnet.camera_view_index,
        )
        return _apply_training_augmentation(self.config, batch) if self.training else batch
