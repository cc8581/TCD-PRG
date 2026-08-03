"""Construction helpers shared by training, evaluation, and cache tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tcd_prg.config import TCDPRGConfig
from tcd_prg.datasets import TaskOrientedClutterAdapter, collate_unified, load_candidate_batch
from tcd_prg.execution import ExternalFR5AG16095Certifier
from tcd_prg.geometry.gripper_provider import ExactAG16095GeometryProvider
from tcd_prg.models.grasp_verifier import build_verifier_inputs
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
            scene_root, scene_root / "metadata.json", config.dataset.scene_points
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
    # Rendering is an explicit caller capability. Formal training passes
    # allow_render=False, while the offline prefetch command passes True.
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
    return TaskOrientedClutterAdapter(
        config.dataset.root,
        observation_provider=create_observation_provider(config, allow_render),
        point_count=config.dataset.scene_points,
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
        global_grasp_library_subdir=config.dataset.global_grasp_library_subdir,
        global_grasp_certification_subdir=config.dataset.global_grasp_certification_subdir,
        pick_remove_global_association_subdir=config.dataset.pick_remove_global_association_subdir,
        global_positive_grasps_per_object=config.sampling.global_positive_grasps_per_object,
        global_intrinsic_negative_grasps_per_object=config.sampling.global_intrinsic_negative_grasps_per_object,
        global_scene_negative_grasps_per_object=config.sampling.global_scene_negative_grasps_per_object,
    )


def create_gripper_provider(
    config: TCDPRGConfig, allow_generate: bool = False
) -> ExactAG16095GeometryProvider:
    return ExactAG16095GeometryProvider(
        config.observation.pybullet_python,
        project_path(config.observation.gripper_worker_script),
        config.dataset.fr5_ag_urdf,
        config.observation.gripper_cache_dir,
        point_count=config.grasp_verifier.gripper_points,
        seed=config.training.seed,
        allow_generate=allow_generate,
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


@dataclass(slots=True)
class UnifiedBatchCollator:
    """Pickle-safe Windows DataLoader collator with cache-only geometry reads."""

    config: TCDPRGConfig
    gripper_provider: ExactAG16095GeometryProvider | None = None
    _candidate_manifest_validated: bool = field(default=False, init=False)

    def __call__(self, samples: list[Any]) -> dict[str, Any]:
        batch = collate_unified(samples)
        # generated candidate manifest 每个 worker 只验证一次，单条 entry 仍逐样本校验来源签名。
        if self.config.training.generated_policy_candidate_cache:
            batch["generated_policy_candidates"] = load_candidate_batch(
                samples,
                self.config.training.generated_policy_candidate_cache,
                self.config,
                self.config.training.generated_policy_checkpoint_sha256,
                validate_manifest=not self._candidate_manifest_validated,
            )
            self._candidate_manifest_validated = True
        # 纯 generated-policy 已缓存几何/Verifier 证据，不再构造昂贵的局部夹爪输入。
        pure_generated_policy = (
            self.config.training.generated_policy_candidate_ratio == 1.0
            and self.config.losses.policy_candidate > 0
            and self.config.losses.verify_overall == 0
        )
        if self.config.ablation.use_gripper_scene_verifier and not pure_generated_policy:
            if self.gripper_provider is None:
                raise RuntimeError("The verifier is enabled but no gripper provider was configured")
            batch["verifier_inputs"] = build_verifier_inputs(
                batch,
                self.gripper_provider,
                local_scene_points=self.config.grasp_verifier.local_scene_points,
                local_radius_m=self.config.model.verifier_local_radius_m,
            )
        return batch
