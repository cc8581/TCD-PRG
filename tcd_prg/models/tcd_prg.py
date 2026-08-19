"""Complete object-centric TCD-PRG model with sensor-only perception input."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from tcd_prg.config import (
    AblationConfig, BackboneConfig, GraspNetConfig, ModelConfig
)
from tcd_prg.geometry.camera import (
    camera_to_world_points,
    camera_to_world_rotations,
    graspnet_to_tcd_rotation,
    look_at_rotation_world_camera,
    world_to_camera_points,
)

from .backbones import (
    PointTransformerV3SceneGeometryBackbone,
    TaskConditionedPointTransformer,
)
from .graspnet import FrozenGraspNetProposalGenerator
from .push import PushHead
from .region import TaskRegionHead
from .task_grasp import AGWidthAdapter, TaskGraspScorer


SENSOR_KEYS = frozenset({
    "xyz", "rgb", "point_mask", "grid_coord", "source_view",
    "graspnet_xyz_world", "graspnet_point_mask",
    "camera2_eye_world", "camera2_target_world", "camera2_up_world",
    "camera2_valid",
    "teacher_target_crop_mask", "teacher_target_identity_valid",
})
TASK_KEYS = frozenset({
    "task_category_id", "task_region_id",
    "target_prompt_xyz", "target_prompt_label", "target_prompt_valid",
    "target_reid_token", "target_reid_center", "target_reid_valid",
})
FORBIDDEN_PERCEPTION_KEYS = frozenset({
    "instance_id", "instance_id_gt", "target_mask", "target_mask_gt",
    "target_object", "object_pose", "object_present", "object_active",
    "object_category_id", "object_category_id_gt", "relation_graph",
    "task_block_graph",
})


class TCDPRGModel(nn.Module):
    """One PTv3 pass, predicted instances, then all manipulation heads.

    `batch` may still contain GT fields for loss construction, but this class
    whitelists perception/task fields and never reads GT instance/target/graph
    tensors during its scene forward.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        ablation: AblationConfig | None = None,
        backbone_config: BackboneConfig | None = None,
        graspnet_config: GraspNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.ablation = ablation or AblationConfig()
        backbone_config = backbone_config or BackboneConfig(backend="legacy")
        graspnet_config = graspnet_config or GraspNetConfig()
        c = self.config

        scene_backbone = None
        if backbone_config.backend == "point_transformer_v3":
            scene_backbone = PointTransformerV3SceneGeometryBackbone(
                dim=c.feature_dim,
                source_root=backbone_config.source_root,
                grid_size_m=backbone_config.grid_size_m,
                enable_flash_attention=backbone_config.enable_flash_attention,
                patch_size=backbone_config.patch_size,
                activation_checkpointing=c.activation_checkpointing,
            )

        self.encoder = TaskConditionedPointTransformer(
            dim=c.feature_dim,
            task_dim=c.task_dim,
            num_categories=c.num_categories,
            num_regions=c.num_task_regions,
            attention_points=backbone_config.attention_points,
            activation_checkpointing=c.activation_checkpointing,
            scene_backbone=scene_backbone,
            instance_queries=c.instance_queries,
            instance_decoder_layers=c.instance_decoder_layers,
            instance_decoder_heads=c.instance_decoder_heads,
            instance_objectness_threshold=c.instance_objectness_threshold,
            target_temperature=c.target_query_temperature,
            target_prompt_radius_m=c.target_prompt_radius_m,
            target_prompt_sigma_m=c.target_prompt_sigma_m,
            target_prompt_weight=c.target_prompt_weight,
            target_category_weight=c.target_category_weight,
            target_objectness_weight=c.target_objectness_weight,
            target_center_weight=c.target_center_weight,
            target_learned_weight=c.target_learned_weight,
            target_reid_weight=c.target_reid_weight,
            target_reid_center_weight=c.target_reid_center_weight,
            target_reid_max_center_distance_m=c.target_reid_max_center_distance_m,
        )
        self.region_head = TaskRegionHead(c.feature_dim)
        self.graspnet = FrozenGraspNetProposalGenerator(
            source_root=graspnet_config.source_root,
            checkpoint=graspnet_config.checkpoint,
            proposal_count=max(
                graspnet_config.global_proposals, graspnet_config.target_proposals
            ),
            input_points=max(
                graspnet_config.scene_input_points, graspnet_config.target_input_points
            ),
            freeze=graspnet_config.freeze,
            num_view=graspnet_config.num_view,
            num_angle=graspnet_config.num_angle,
            num_depth=graspnet_config.num_depth,
            cylinder_radius=graspnet_config.cylinder_radius,
            hmin=graspnet_config.hmin,
            hmax_list=graspnet_config.hmax_list,
        )
        self.graspnet_config = graspnet_config
        self.target_prompt_min_support = float(c.target_prompt_min_support)
        self.target_prompt_min_margin = float(c.target_prompt_min_margin)
        self.task_grasp = TaskGraspScorer(
            c.feature_dim,
            layers=c.task_grasp_scorer_layers,
            heads=c.task_grasp_scorer_heads,
            local_radius_m=c.task_grasp_local_radius_m,
            residual_scale=c.task_grasp_residual_scale,
        )
        self.ag_width = AGWidthAdapter(
            c.feature_dim,
            max_width_m=c.max_grasp_width_m,
            local_radius_m=c.task_grasp_local_radius_m,
        )
        self.push = PushHead(
            c.feature_dim,
            c.num_direction_bins,
            c.push_direction_feature_dim,
            c.push_direction_transformer_layers,
            c.push_direction_transformer_heads,
            c.push_direction_contact_topk,
            c.push_object_topk,
        )
    @staticmethod
    def _sensor(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        source = batch.get("model_inputs", batch)
        sensor = {key: source[key] for key in SENSOR_KEYS if key in source}
        missing = {"xyz", "rgb", "point_mask"} - sensor.keys()
        if missing:
            raise KeyError(f"Missing sensor model inputs: {sorted(missing)}")
        return sensor

    @staticmethod
    def _task(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        source = batch.get("task_inputs", batch)
        required = {"task_category_id", "task_region_id"}
        missing = required - source.keys()
        if missing:
            raise KeyError(f"Missing task inputs: {sorted(missing)}")
        return {key: source[key] for key in TASK_KEYS if key in source}

    @staticmethod
    def assert_no_gt_in_model_inputs(batch: Mapping[str, Any]) -> None:
        """Hard guard used by tests/real inference for nested formal inputs."""
        if "model_inputs" not in batch:
            return
        leaked = FORBIDDEN_PERCEPTION_KEYS & set(batch["model_inputs"])
        if leaked:
            raise RuntimeError(
                "Ground-truth fields leaked into model_inputs: "
                + ", ".join(sorted(leaked))
            )

    def _encode_scene(self, batch: Mapping[str, Any]) -> tuple[Any, dict[str, Tensor], dict[str, Tensor]]:
        self.assert_no_gt_in_model_inputs(batch)
        sensor = self._sensor(batch)
        task = self._task(batch)
        encoded = self.encoder(
            sensor["xyz"],
            sensor["rgb"],
            sensor["point_mask"],
            task["task_category_id"],
            task["task_region_id"],
            use_task_region_condition=self.ablation.use_task_region_condition,
            grid_coord=sensor.get("grid_coord"),
            target_prompt_xyz=task.get("target_prompt_xyz"),
            target_prompt_label=task.get("target_prompt_label"),
            target_prompt_valid=task.get("target_prompt_valid"),
            target_reid_token=task.get("target_reid_token"),
            target_reid_center=task.get("target_reid_center"),
            target_reid_valid=task.get("target_reid_valid"),
        )
        return encoded, sensor, task

    def _encode_scene_instances(
        self, batch: Mapping[str, Any]
    ) -> tuple[Any, Any, dict[str, Tensor]]:
        self.assert_no_gt_in_model_inputs(batch)
        sensor = self._sensor(batch)
        scene, instance = self.encoder.forward_scene_instances(
            sensor["xyz"], sensor["rgb"], sensor["point_mask"],
            grid_coord=sensor.get("grid_coord"),
        )
        return scene, instance, sensor

    def _forward_global_grasp_neutral(
        self, scene: Any, instance: Any, sensor: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        proposals = self._forward_camera_graspnet(
            sensor,
            target_probability=None,
            instance_probability=instance.mask_probability,
            strict_target_crop=False,
            proposal_count=self.graspnet_config.global_proposals,
            input_points=self.graspnet_config.scene_input_points,
        )
        width = self.ag_width(
            proposals,
            scene.point_features,
            sensor["xyz"],
            sensor["point_mask"],
            sensor["point_mask"].to(sensor["xyz"].dtype),
        )
        return {**proposals, **width}

    def _forward_camera_graspnet(
        self,
        sensor: dict[str, Tensor],
        *,
        target_probability: Tensor | None,
        instance_probability: Tensor,
        strict_target_crop: bool,
        proposal_count: int,
        input_points: int,
        target_identity_valid: Tensor | None = None,
        target_crop_mask_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run official GraspNet on the independent Camera2 cloud."""

        required = {
            "graspnet_xyz_world", "graspnet_point_mask", "camera2_eye_world",
            "camera2_target_world", "camera2_up_world", "camera2_valid",
        }
        missing = required - sensor.keys()
        if missing:
            raise KeyError(f"Missing Camera2 GraspNet sensor inputs: {sorted(missing)}")
        camera_xyz_world = sensor["graspnet_xyz_world"]
        camera_point_mask = sensor["graspnet_point_mask"].bool()
        rotation_world_camera = look_at_rotation_world_camera(
            sensor["camera2_eye_world"],
            sensor["camera2_target_world"],
            sensor["camera2_up_world"],
        )
        camera_xyz = world_to_camera_points(
            camera_xyz_world, rotation_world_camera, sensor["camera2_eye_world"]
        )

        reference_mask = sensor["point_mask"].bool()
        if "source_view" in sensor:
            reference_mask = reference_mask & (
                sensor["source_view"].long()
                == int(self.graspnet_config.camera_view_index)
            )
        nearest_scene, transfer_distance, transfer_valid = self._nearest_reference_index(
            camera_xyz_world,
            camera_point_mask,
            sensor["xyz"],
            reference_mask,
            max_distance_m=self.graspnet_config.camera_transfer_max_distance_m,
        )
        camera_instance_probability = torch.gather(
            instance_probability,
            2,
            nearest_scene[:, None].expand(-1, instance_probability.shape[1], -1),
        ) * transfer_valid[:, None].to(instance_probability.dtype)
        identity_valid = (
            torch.ones_like(sensor["camera2_valid"], dtype=torch.bool)
            if target_identity_valid is None
            else target_identity_valid.bool()
        )
        if target_crop_mask_override is not None:
            crop_mask = (
                camera_point_mask
                & target_crop_mask_override.bool()
                & sensor["camera2_valid"][:, None].bool()
            )
            crop_count = crop_mask.sum(-1)
            target_grasp_valid = (
                sensor["camera2_valid"].bool()
                & identity_valid
                & (crop_count >= self.graspnet_config.target_min_crop_points)
            )
            crop_mask = crop_mask & target_grasp_valid[:, None]
            camera_target_probability = target_crop_mask_override.to(camera_xyz.dtype)
        elif strict_target_crop:
            if target_probability is None:
                raise ValueError("strict target crop requires predicted target_probability")
            camera_target_probability = target_probability.gather(
                1, nearest_scene
            ) * transfer_valid.to(target_probability.dtype)
            crop_mask = camera_point_mask & transfer_valid & (
                camera_target_probability >= self.graspnet_config.target_crop_probability
            )
            crop_count = crop_mask.sum(-1)
            target_grasp_valid = (
                sensor["camera2_valid"].bool()
                & identity_valid
                & (crop_count >= self.graspnet_config.target_min_crop_points)
            )
            crop_mask = crop_mask & target_grasp_valid[:, None]
        else:
            camera_target_probability = camera_point_mask.to(camera_xyz.dtype)
            crop_mask = camera_point_mask & sensor["camera2_valid"][:, None].bool()
            crop_count = crop_mask.sum(-1)
            target_grasp_valid = sensor["camera2_valid"].bool() & crop_mask.any(-1)

        proposal = self.graspnet(
            camera_xyz,
            crop_mask,
            instance_probability=camera_instance_probability,
            proposal_count=proposal_count,
            input_points=input_points,
        )
        translation_camera = proposal["translation_world"]
        rotation_camera_graspnet = proposal["rotation_matrix"]
        rotation_camera_tcd = graspnet_to_tcd_rotation(rotation_camera_graspnet)
        translation_world = camera_to_world_points(
            translation_camera, rotation_world_camera, sensor["camera2_eye_world"]
        )
        rotation_world_tcd = camera_to_world_rotations(
            rotation_camera_tcd, rotation_world_camera
        )
        valid = proposal["valid"].bool() & target_grasp_valid[:, None]
        scene_attention, scene_found = self._nearest_scene_point(
            sensor["xyz"], sensor["point_mask"], translation_world, valid
        )
        scene_membership = torch.gather(
            instance_probability,
            2,
            scene_attention[:, None].expand(-1, instance_probability.shape[1], -1),
        ).transpose(1, 2)
        graspnet_width = proposal["width_m"]
        return {
            **proposal,
            "translation_camera": translation_camera,
            "rotation_matrix_camera_graspnet": rotation_camera_graspnet,
            "rotation_matrix_camera_tcd": rotation_camera_tcd,
            "translation_world": translation_world,
            "rotation_matrix": rotation_world_tcd,
            "graspnet_width_m": graspnet_width,
            "width_m": graspnet_width,
            "graspnet_attention_point_index": proposal["attention_point_index"],
            "attention_point_index": scene_attention,
            "object_logits": scene_membership.clamp_min(1e-6).log(),
            "valid": valid & scene_found,
            "target_grasp_valid": target_grasp_valid,
            "target_crop_mask": crop_mask,
            "target_crop_probability": camera_target_probability,
            "target_crop_points": crop_count,
            "camera_transfer_reference_index": nearest_scene,
            "camera_transfer_distance_m": transfer_distance,
            "camera_transfer_valid": transfer_valid,
            "camera_transfer_coverage": (
                transfer_valid.float().sum(-1)
                / camera_point_mask.float().sum(-1).clamp_min(1.0)
            ),
            "target_identity_valid": identity_valid,
            "graspnet_valid_proposals_per_row": valid.sum(-1),
        }

    def _forward_global_grasp(
        self, encoded: Any, sensor: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        # Reuse the exact neutral features/instances produced before task FiLM.
        class _Scene:
            pass
        scene = _Scene()
        scene.point_features = encoded.scene_point_features
        scene.global_scene_token = encoded.scene_global_token
        return self._forward_global_grasp_neutral(
            scene, encoded.instance, sensor
        )

    def _target_identity_gate(self, encoded: Any) -> Tensor:
        """Fail closed unless a prompt or a valid ReID track identifies the target."""

        prompt_support = encoded.target_prompt_support
        if prompt_support.ndim == 2:
            rows = torch.arange(
                prompt_support.shape[0], device=prompt_support.device
            )
            prompt_support = prompt_support[
                rows, encoded.target_query_index
            ]
        prompt_valid = (
            encoded.target_prompt_used.bool()
            & (prompt_support >= self.target_prompt_min_support)
        )
        reid_valid = (
            ~encoded.target_prompt_used.bool()
            & encoded.target_reid_used.bool()
        )
        return (
            (prompt_valid | reid_valid)
            & (encoded.target_selection_margin >= self.target_prompt_min_margin)
        )

    def forward_global_grasp(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        scene, instance, sensor = self._encode_scene_instances(batch)
        return {
            "scene": scene,
            "instance": instance,
            "global_grasp": self._forward_global_grasp_neutral(
                scene, instance, sensor
            ),
        }

    @staticmethod
    @torch.no_grad()
    def _nearest_reference_index(
        query: Tensor,
        query_mask: Tensor,
        reference: Tensor,
        reference_mask: Tensor,
        chunk_size: int = 1024,
        max_distance_m: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Nearest reference point without materializing an N-by-M matrix."""

        result = torch.zeros(
            query.shape[:2], dtype=torch.long, device=query.device
        )
        nearest_distance = torch.full(
            query.shape[:2], float("inf"), dtype=torch.float32, device=query.device
        )
        for row in range(query.shape[0]):
            queries = torch.nonzero(query_mask[row], as_tuple=False).flatten()
            references = torch.nonzero(
                reference_mask[row], as_tuple=False
            ).flatten()
            if not len(queries) or not len(references):
                continue
            for start in range(0, len(queries), chunk_size):
                local = queries[start : start + chunk_size]
                distance = torch.cdist(
                    query[row, local].float(), reference[row, references].float()
                )
                local_distance, local_index = distance.min(-1)
                result[row, local] = references[local_index]
                nearest_distance[row, local] = local_distance
        valid = query_mask.bool() & torch.isfinite(nearest_distance)
        if max_distance_m is not None:
            valid = valid & (nearest_distance <= float(max_distance_m))
        return result, nearest_distance, valid

    @staticmethod
    @torch.no_grad()
    def _nearest_scene_point(
        xyz: Tensor, point_mask: Tensor, query: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Nearest visible sensor point for candidate reference positions."""
        safe = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0).float()
        distance = torch.cdist(safe, xyz.float())
        domain = point_mask[:, None] & valid[:, :, None]
        distance = distance.masked_fill(~domain, float("inf"))
        found = domain.any(-1)
        return distance.argmin(-1), found

    def _forward_region(
        self, encoded: Any, sensor: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        return self.region_head(
            encoded.point_features,
            encoded.target_token,
            encoded.task_token,
            encoded.target_probability,
            sensor["point_mask"],
        )

    def _forward_task_grasps(
        self,
        encoded: Any,
        sensor: dict[str, Tensor],
        region: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        target_identity_valid = self._target_identity_gate(encoded)
        teacher_crop = sensor.get("teacher_target_crop_mask") if self.training else None
        if teacher_crop is not None:
            target_identity_valid = sensor.get(
                "teacher_target_identity_valid", teacher_crop.bool().any(-1)
            ).bool()
        target_proposals = self._forward_camera_graspnet(
            sensor,
            target_probability=encoded.target_instance_probability,
            instance_probability=encoded.instance.mask_probability,
            strict_target_crop=True,
            proposal_count=self.graspnet_config.target_proposals,
            input_points=self.graspnet_config.target_input_points,
            target_identity_valid=target_identity_valid,
            target_crop_mask_override=teacher_crop,
        )
        target_proposals["target_hard_mask_fused"] = (
            encoded.target_instance_probability
            >= self.graspnet_config.target_crop_probability
        ) & sensor["point_mask"].bool()
        target_proposals["target_identity_teacher_forced"] = torch.full_like(
            target_identity_valid, teacher_crop is not None, dtype=torch.bool
        )
        fp32_proposals = {
            key: value.float() if value.is_floating_point() else value
            for key, value in target_proposals.items()
        }
        with torch.autocast(device_type=sensor["xyz"].device.type, enabled=False):
            target_width = self.ag_width(
                fp32_proposals,
                encoded.point_features.float(),
                sensor["xyz"].float(),
                sensor["point_mask"],
                encoded.target_probability.float(),
            )
            fp32_proposals = {**fp32_proposals, **target_width}
            task_grasp = self.task_grasp(
                fp32_proposals,
                encoded.point_features.float(),
                sensor["xyz"].float(),
                sensor["point_mask"],
                region["region_probability"].float(),
                encoded.target_probability.float(),
                encoded.task_token.float(),
                encoded.target_token.float(),
            )
        return task_grasp

    def _forward_push(
        self,
        encoded: Any,
        sensor: dict[str, Tensor],
        region: dict[str, Tensor],
        training_hints: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        hints = training_hints or {}
        return self.push(
            encoded.point_features,
            sensor["xyz"],
            encoded.instance.mask_probability,
            sensor["point_mask"],
            encoded.object_tokens,
            encoded.object_mask,
            encoded.task_token,
            encoded.target_token,
            encoded.target_instance_probability,
            region["region_probability"],
            forced_direction_point_mask=hints.get("push_direction_point_mask"),
        )

    def forward_instances(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Task-free sensor-only instance inference for real-scene acquisition."""
        scene, instance, sensor = self._encode_scene_instances(batch)
        return {"scene": scene, "instance": instance, "sensor": sensor}

    def forward_perception(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Stage A: instance/target/region only; no GraspNet or Push execution."""
        encoded, sensor, task = self._encode_scene(batch)
        region = self._forward_region(encoded, sensor)
        return {
            "encoded": encoded,
            "sensor": sensor,
            "task": task,
            "instance": encoded.instance,
            "region": region,
            "task_grasp": None,
            "global_grasp": None,
            "push": None,
        }

    def forward_grasp(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Stage B: perception + Grasp only; Push is not executed."""
        encoded, sensor, task = self._encode_scene(batch)
        region = self._forward_region(encoded, sensor)
        task_grasp = self._forward_task_grasps(encoded, sensor, region)
        return {
            "encoded": encoded,
            "sensor": sensor,
            "task": task,
            "instance": encoded.instance,
            "region": region,
            "task_grasp": task_grasp,
            "global_grasp": None,
            "push": None,
        }

    def forward_push(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Stage C: perception + Push only; GraspNet/TaskGrasp are not executed."""
        encoded, sensor, task = self._encode_scene(batch)
        region = self._forward_region(encoded, sensor)
        push = self._forward_push(
            encoded, sensor, region, batch.get("training_hints")
        )
        return {
            "encoded": encoded,
            "sensor": sensor,
            "task": task,
            "instance": encoded.instance,
            "region": region,
            "task_grasp": None,
            "global_grasp": None,
            "push": push,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        forward_mode: str = "full",
    ) -> dict[str, Any]:
        if forward_mode == "instances":
            return self.forward_instances(batch)
        if forward_mode == "perception":
            return self.forward_perception(batch)
        if forward_mode == "grasp":
            return self.forward_grasp(batch)
        if forward_mode == "push":
            return self.forward_push(batch)
        if forward_mode == "global_grasp":
            return self.forward_global_grasp(batch)
        if forward_mode != "full":
            raise ValueError(f"Unsupported forward_mode={forward_mode}")

        encoded, sensor, task = self._encode_scene(batch)
        region = self._forward_region(encoded, sensor)
        task_grasp = self._forward_task_grasps(encoded, sensor, region)
        global_grasp = self._forward_global_grasp(encoded, sensor)
        push = self._forward_push(
            encoded, sensor, region, batch.get("training_hints")
        )
        return {
            "encoded": encoded,
            "sensor": sensor,
            "task": task,
            "instance": encoded.instance,
            "region": region,
            "task_grasp": task_grasp,
            "global_grasp": global_grasp,
            "push": push,
        }
