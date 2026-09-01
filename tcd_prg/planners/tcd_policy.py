"""Closed-loop TCD-PRG policy using sensor-only perception inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tcd_prg.baselines.base import GlobalGraspPrediction, ManipulationPolicy
from tcd_prg.config import TCDPRGConfig
from tcd_prg.constants import ActionType
from tcd_prg.datasets.collate import grid_sample_indices
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.models import TCDPRGModel

from .candidate_generator import DenseCandidateGenerator
from .target_tracker import TargetIdentityTracker


@dataclass(slots=True)
class EncodedPolicyState:
    observation: SceneObservation | None
    cpu_batch: dict[str, Any]
    device_batch: dict[str, Any]
    output: dict[str, Any]


class TCDPRGPolicy(ManipulationPolicy):
    def __init__(
        self,
        model: TCDPRGModel,
        config: TCDPRGConfig,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.device = next(model.parameters()).device
        self.generator = DenseCandidateGenerator(
            config.model,
            use_push_potential=config.ablation.use_push_potential,
        )
        self.preparation_actions = 0
        self.target_tracker = TargetIdentityTracker()

    def _sensor_task_batch(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        task_category_id: int,
        task_region_id: int,
        point_valid: np.ndarray | None = None,
        source_view: np.ndarray | None = None,
        camera_parameters: tuple[Any, ...] | None = None,
        target_prompt_xyz: np.ndarray | None = None,
        target_prompt_label: np.ndarray | None = None,
        continue_target: bool = False,
    ) -> dict[str, Any]:
        if point_valid is None:
            point_valid = np.ones(len(xyz), dtype=bool)
        else:
            point_valid = np.asarray(point_valid, dtype=bool)
            if point_valid.shape != (len(xyz),):
                raise ValueError("point_valid must be [N]")
        if source_view is not None:
            source_view = np.asarray(source_view)
            if source_view.shape != (len(xyz),):
                raise ValueError("source_view must be [N]")
        valid_index = np.flatnonzero(point_valid)
        if not len(valid_index):
            raise ValueError("Sensor observation contains no valid points")
        selected = valid_index[
            grid_sample_indices(
                xyz[valid_index],
                self.config.backbone.grid_size_m,
                training=False,
            )
        ]
        model_inputs = {
            "xyz": torch.from_numpy(xyz[selected])[None].float(),
            "rgb": torch.from_numpy(rgb[selected])[None].float(),
            "point_mask": torch.ones(1, len(selected), dtype=torch.bool),
        }
        camera_index = self.config.graspnet.camera_view_index
        camera_usable = (
            source_view is not None
            and camera_parameters is not None
            and camera_index < len(camera_parameters)
        )
        camera_indices = (
            np.flatnonzero(point_valid & (source_view == camera_index))
            if camera_usable
            else np.empty((0,), np.int64)
        )
        if len(camera_indices):
            local = grid_sample_indices(
                xyz[camera_indices], self.config.backbone.grid_size_m, training=False
            )
            camera_indices = camera_indices[local]
            limit = max(
                self.config.graspnet.scene_input_points,
                self.config.graspnet.target_input_points,
            )
            if len(camera_indices) > limit:
                camera_indices = camera_indices[
                    np.linspace(0, len(camera_indices) - 1, limit, dtype=np.int64)
                ]
        camera_xyz = xyz[camera_indices] if len(camera_indices) else np.zeros((1, 3), np.float32)
        model_inputs["graspnet_xyz_world"] = torch.from_numpy(camera_xyz)[None].float()
        model_inputs["graspnet_point_mask"] = torch.tensor(
            [[True] * len(camera_indices)] if len(camera_indices) else [[False]],
            dtype=torch.bool,
        )
        model_inputs["source_view"] = torch.from_numpy(
            source_view[selected] if source_view is not None else np.full(len(selected), -1)
        )[None].long()
        if camera_usable:
            camera = camera_parameters[camera_index]
            model_inputs["camera2_eye_world"] = torch.from_numpy(
                np.asarray(camera.eye_world, np.float32)
            )[None]
            model_inputs["camera2_target_world"] = torch.from_numpy(
                np.asarray(camera.target_world, np.float32)
            )[None]
            model_inputs["camera2_up_world"] = torch.from_numpy(
                np.asarray(camera.up_world, np.float32)
            )[None]
        else:
            model_inputs["camera2_eye_world"] = torch.zeros(1, 3)
            model_inputs["camera2_target_world"] = torch.tensor([[0.0, 0.0, 1.0]])
            model_inputs["camera2_up_world"] = torch.tensor([[0.0, -1.0, 0.0]])
        model_inputs["camera2_valid"] = torch.tensor(
            [bool(camera_usable and len(camera_indices))], dtype=torch.bool
        )
        task_inputs = {
            "task_category_id": torch.tensor([int(task_category_id)], dtype=torch.long),
            "task_region_id": torch.tensor([int(task_region_id)], dtype=torch.long),
        }
        if target_prompt_xyz is None:
            task_inputs.update(
                {
                    "target_prompt_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
                    "target_prompt_label": torch.ones((1, 1), dtype=torch.long),
                    "target_prompt_valid": torch.zeros((1, 1), dtype=torch.bool),
                }
            )
        else:
            prompt = np.asarray(target_prompt_xyz, np.float32).reshape(-1, 3)
            labels = (
                np.ones(len(prompt), np.int64)
                if target_prompt_label is None
                else np.asarray(target_prompt_label, np.int64).reshape(-1)
            )
            if len(labels) != len(prompt):
                raise ValueError("target_prompt_label length must match target_prompt_xyz")
            task_inputs.update(
                {
                    "target_prompt_xyz": torch.from_numpy(prompt)[None].float(),
                    "target_prompt_label": torch.from_numpy(labels)[None].long(),
                    "target_prompt_valid": torch.ones((1, len(prompt)), dtype=torch.bool),
                }
            )
        if continue_target:
            state = self.target_tracker.state
            if state is None:
                raise RuntimeError("continue_target=True but no tracked target identity exists")
            if int(task_category_id) != int(state.category_id):
                raise RuntimeError(
                    f"tracked target category={state.category_id} but task requests {task_category_id}"
                )
            task_inputs.update(self.target_tracker.task_inputs())
        return {
            "model_inputs": model_inputs,
            "task_inputs": task_inputs,
        }

    @staticmethod
    def target_prompt_from_instance(
        xyz: np.ndarray, instance_id: np.ndarray, target_query: int
    ) -> np.ndarray:
        """Return an observed point near the selected predicted instance centroid."""
        xyz = np.asarray(xyz, np.float32)
        instance_id = np.asarray(instance_id, np.int64)
        points = np.flatnonzero(instance_id == int(target_query))
        if not len(points):
            raise RuntimeError(f"predicted target query {target_query} has no visible fused points")
        cloud = xyz[points]
        centroid = cloud.mean(0)
        index = points[np.linalg.norm(cloud - centroid, axis=-1).argmin()]
        return xyz[index].copy()

    @staticmethod
    def target_prompt_from_mask(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Dataset/evaluation adapter: synthesize one observable target click."""
        xyz = np.asarray(xyz, np.float32)
        points = np.flatnonzero(np.asarray(mask, bool))
        if not len(points):
            raise RuntimeError("target instance has no visible prompt point")
        cloud = xyz[points]
        centroid = cloud.mean(0)
        return xyz[points[np.linalg.norm(cloud - centroid, axis=-1).argmin()]].copy()

    def _batch(self, observation: SceneObservation) -> dict[str, Any]:
        """Dataset/evaluation adapter: GT fields are deliberately ignored."""
        observation.validate()
        # Category/region are the closed-vocabulary task specification.  The
        # GT target mask/instance id/object state are not copied into model input.
        category = int(observation.object_category_id[observation.target_object])
        point_valid = (
            observation.point_valid
            if observation.point_valid is not None
            else np.ones(len(observation.xyz), dtype=bool)
        )
        visible_target = observation.target_mask & point_valid
        prompt = (
            self.target_prompt_from_mask(observation.xyz, visible_target)
            if bool(np.any(visible_target))
            else None
        )
        return self._sensor_task_batch(
            observation.xyz,
            observation.rgb,
            category,
            int(observation.task_region_id),
            point_valid=observation.point_valid,
            source_view=observation.source_view,
            camera_parameters=observation.camera_parameters,
            target_prompt_xyz=prompt,
        )

    def _encode_batch(
        self,
        batch: dict[str, Any],
        observation: SceneObservation | None,
    ) -> EncodedPolicyState:
        device = {
            key: (
                {subkey: value.to(self.device) for subkey, value in nested.items()}
                if isinstance(nested, dict)
                else nested
            )
            for key, nested in batch.items()
        }
        with torch.no_grad():
            output = self.model(device)
        state = EncodedPolicyState(observation, batch, device, output)
        task_inputs = batch.get("task_inputs", {})
        if "encoded" in output and "task_category_id" in task_inputs:
            prompt_valid = task_inputs.get("target_prompt_valid")
            reid_valid = task_inputs.get("target_reid_valid")
            establishes_identity = bool(
                (prompt_valid is not None and bool(prompt_valid.any()))
                or (reid_valid is not None and bool(reid_valid.any()))
            )
            if establishes_identity:
                self.target_tracker.update(
                    output["encoded"], int(task_inputs["task_category_id"][0])
                )
        return state

    def encode_observation(self, observation: SceneObservation) -> EncodedPolicyState:
        return self._encode_batch(self._batch(observation), observation)

    def segment_fused_scene(
        self,
        xyz_m: np.ndarray,
        rgb: np.ndarray,
        *,
        mask_threshold: float = 0.5,
        minimum_points: int = 16,
    ) -> dict[str, Any]:
        """Predict instance ids/categories directly from fused XYZRGB."""
        xyz_m = np.asarray(xyz_m, np.float32)
        rgb = np.asarray(rgb, np.float32)
        if len(xyz_m) != len(rgb) or not len(xyz_m):
            raise ValueError("xyz/rgb must contain the same non-zero point count")
        valid = np.ones(len(xyz_m), dtype=bool)
        selected = np.flatnonzero(valid)
        selected = selected[
            grid_sample_indices(
                xyz_m[selected],
                self.config.backbone.grid_size_m,
                training=False,
            )
        ]
        model_inputs = {
            "xyz": torch.from_numpy(xyz_m[selected])[None].float().to(self.device),
            "rgb": torch.from_numpy(rgb[selected])[None].float().to(self.device),
            "point_mask": torch.ones(1, len(selected), dtype=torch.bool, device=self.device),
        }
        with torch.no_grad():
            output = self.model({"model_inputs": model_inputs}, forward_mode="instances")
        instance = output["instance"]
        object_valid = instance.object_mask[0]
        probability = instance.mask_probability[0].clone()
        probability = probability * object_valid[:, None].to(probability.dtype)
        best_probability, best_query = probability.max(0)
        assigned_query = torch.full_like(best_query, -1)
        categories: dict[int, int] = {}
        objectness_by_instance: dict[int, float] = {}
        center_by_instance: dict[int, np.ndarray] = {}
        for query in torch.nonzero(object_valid, as_tuple=False).flatten().tolist():
            points = (best_query == query) & (best_probability >= float(mask_threshold))
            if int(points.sum()) < int(minimum_points):
                continue
            # Preserve the learned query id so GUI instance ids and action
            # acted_object ids use the same runtime identity space.
            assigned_query[points] = query
            categories[query] = int(instance.category_logits[0, query].argmax().item())
            objectness_by_instance[query] = float(
                torch.sigmoid(instance.objectness_logits[0, query]).item()
            )
            center_by_instance[query] = (
                instance.centers_world[0, query].detach().cpu().numpy().astype(np.float32)
            )
        dense_instance = np.full(len(xyz_m), -1, np.int64)
        dense_instance[selected] = assigned_query.detach().cpu().numpy().astype(np.int64)
        return {
            "instance_id": dense_instance,
            "category_by_instance": categories,
            "objectness_by_instance": objectness_by_instance,
            "center_by_instance": center_by_instance,
        }

    def encode_fused_scene(
        self,
        xyz_m: np.ndarray,
        rgb: np.ndarray,
        task_category_id: int,
        task_region_id: int,
        *,
        point_valid: np.ndarray | None = None,
        source_view: np.ndarray | None = None,
        camera_parameters: tuple[Any, ...] | None = None,
        target_prompt_xyz: np.ndarray | None = None,
        target_prompt_label: np.ndarray | None = None,
        continue_target: bool = False,
        enforce_target_confidence: bool = False,
    ) -> EncodedPolicyState:
        """Real entry: fused XYZRGB + task semantics + observable target identity."""
        batch = self._sensor_task_batch(
            np.asarray(xyz_m, np.float32),
            np.asarray(rgb, np.float32),
            int(task_category_id),
            int(task_region_id),
            point_valid=point_valid,
            source_view=source_view,
            camera_parameters=camera_parameters,
            target_prompt_xyz=target_prompt_xyz,
            target_prompt_label=target_prompt_label,
            continue_target=continue_target,
        )
        state = self._encode_batch(batch, None)
        if enforce_target_confidence:
            self._validate_target_selection(state)
        return state

    def _validate_target_selection(self, state: EncodedPolicyState) -> None:
        encoded = state.output["encoded"]
        query = int(encoded.target_query_index[0].detach().cpu())
        margin = float(encoded.target_selection_margin[0].detach().cpu())
        support = float(encoded.target_prompt_support[0, query].detach().cpu())
        prompt_used = bool(encoded.target_prompt_used[0].detach().cpu())
        if prompt_used and support < float(self.config.model.target_prompt_min_support):
            raise RuntimeError(
                f"target prompt support too low ({support:.3f} < "
                f"{self.config.model.target_prompt_min_support:.3f}); reacquire/reselect target"
            )
        if margin < float(self.config.model.target_prompt_min_margin):
            raise RuntimeError(
                f"target query ambiguous (top1-top2 margin={margin:.3f} < "
                f"{self.config.model.target_prompt_min_margin:.3f}); request another prompt"
            )

    def select_clear_target(self, segmentation: dict[str, Any]) -> int:
        """Rule-based clear-scene helper: choose the most confident predicted object."""
        scores = segmentation.get("objectness_by_instance", {})
        if not scores:
            raise RuntimeError("no valid predicted instance is available for clearing")
        return int(max(scores, key=scores.get))

    @staticmethod
    def _action(candidates: dict[str, Tensor], index: int) -> dict[str, Any]:
        def array(name: str):
            value = candidates[name][0, index].detach().cpu().numpy()
            return value.item() if value.ndim == 0 else value

        kind = int(array("type"))
        action = {
            "candidate_index": index,
            # This is a predicted query id, not a simulator object index.
            "acted_object": int(array("object")),
            "action_type": kind,
            "proposal_score": float(array("proposal_score")),
        }
        if kind == int(ActionType.PUSH):
            action.update(
                push_contact_world=array("contact_world"),
                push_direction_world=array("direction_world"),
                push_distance_m=float(array("push_distance_m")),
                effective_probability=float(array("effective_probability")),
                q_value=array("push_q_value"),
                safety_probability=float(array("push_safety_probability")),
            )
        else:
            action.update(
                grasp_pose_world=array("pose_world"),
                grasp_width_m=float(array("width_m")),
                removal_destination_world=array("destination_world"),
            )
        return action

    def generate_candidates(self, encoded: EncodedPolicyState) -> dict[str, Any]:
        with torch.no_grad():
            candidates = self.generator.generate(self.model, encoded.device_batch, encoded.output)
            task_mask = candidates["type"] == int(ActionType.TASK_GRASP)
            candidates["task_grasp_query_count"] = torch.full(
                (candidates["type"].shape[0],),
                encoded.output["task_grasp"]["task_valid_logit"].shape[1],
                dtype=torch.long,
                device=self.device,
            )
            candidates["task_grasp_after_nms_count"] = (candidates["valid"] & task_mask).sum(-1)
            # DenseCandidateGenerator has already applied task-grasp NMS and
            # the calibrated probability threshold. One executable grasp is enough
            # for execution; dataset grasp-count labels are not runtime hard gates.
            candidates["unique_task_grasp_count"] = (candidates["valid"] & task_mask).sum(-1)
            # PUSH rows already come from the shared decoder's post-NMS set.
            candidates["push_after_nms_count"] = (
                candidates["valid"] & (candidates["type"] == int(ActionType.PUSH))
            ).sum(-1)
        return {
            "encoded": encoded,
            "candidates": candidates,
            "certification_reasons": [],
        }

    def select_action(self, candidates: dict[str, Any]) -> dict[str, Any] | None:
        tensors = candidates["candidates"]
        valid = tensors["valid"][0]
        for action_type in (
            ActionType.TASK_GRASP,
            ActionType.PICK_REMOVE,
            ActionType.PUSH,
        ):
            indices = torch.nonzero(
                valid & (tensors["type"][0] == int(action_type)),
                as_tuple=False,
            ).flatten()
            if not len(indices):
                continue
            ranking_score = (
                tensors["effective_probability"]
                if action_type == ActionType.PUSH
                else tensors["proposal_score"]
            )
            if action_type == ActionType.PUSH and not bool(
                torch.isfinite(ranking_score[0, indices]).all()
            ):
                raise RuntimeError("PUSH selection requires a loaded PushEffectivenessEvaluator")
            order = indices[ranking_score[0, indices].argsort(descending=True, stable=True)]
            for index_tensor in order:
                index = int(index_tensor)
                action = self._action(tensors, index)
                action["selection_score"] = float(ranking_score[0, index])
                point_index = int(tensors["point_index"][0, index])
                if point_index >= 0:
                    action["association_point_world"] = (
                        candidates["encoded"]
                        .cpu_batch["model_inputs"]["xyz"][0, point_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                return action
        return None

    def predict_grasps(self, encoded: EncodedPolicyState) -> list[dict[str, Any]]:
        candidates = self.generate_candidates(encoded)["candidates"]
        indices = torch.nonzero(
            candidates["valid"][0] & (candidates["type"][0] == int(ActionType.TASK_GRASP)),
            as_tuple=False,
        ).flatten()
        return [self._action(candidates, int(index)) for index in indices]

    def predict_task_grasps(self, encoded: EncodedPolicyState) -> list[dict[str, Any]]:
        return self.predict_grasps(encoded)

    def predict_global_grasps(self, encoded: EncodedPolicyState) -> list[GlobalGraspPrediction]:
        decoded = self.generator.global_predictions(
            encoded.device_batch,
            encoded.output,
            self.config.model.candidate_topk,
        )[0]
        predictions: list[GlobalGraspPrediction] = []
        for index in range(len(decoded["scene_score"])):
            predictions.append(
                GlobalGraspPrediction(
                    object_index=int(decoded["object"][index]),
                    contact_point_world=decoded["contact_world"][index].detach().cpu().numpy(),
                    grasp_pose_world=decoded["pose_world"][index].detach().cpu().numpy(),
                    width_m=float(decoded["width_m"][index]),
                    raw_score=float(decoded["raw_score"][index]),
                    scene_score=float(decoded["scene_score"][index]),
                    intrinsic_score=None,
                    certified=False,
                    source="tcd_prg_global",
                )
            )
        return predictions

    def reset(self) -> None:
        self.preparation_actions = 0
        self.target_tracker.reset()

    def update_after_action(self, action: Any, observation: SceneObservation | None) -> None:
        if isinstance(action, dict) and "action_type" in action:
            action_type = int(action["action_type"])
            if action_type != int(ActionType.TASK_GRASP):
                self.preparation_actions += 1
