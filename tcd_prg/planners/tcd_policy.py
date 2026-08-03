"""Closed-loop TCD-PRG policy using one shared scene encoding per observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from tcd_prg.baselines.base import GlobalGraspPrediction, ManipulationPolicy
from tcd_prg.config import TCDPRGConfig
from tcd_prg.constants import ActionType, MAX_PREPARATION_ACTIONS
from tcd_prg.datasets.collate import grid_sample_indices
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.geometry.grasp_nms import task_grasp_nms
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.grasp_verifier import build_verifier_inputs
from tcd_prg.models.policy.router import MaskedHierarchicalCandidateRouter

from .candidate_generator import DenseCandidateGenerator


def apply_verified_candidate_count_gate(
    candidate_type: Tensor,
    candidate_object: Tensor,
    candidate_pose_world: Tensor,
    candidate_width_m: Tensor,
    candidate_score: Tensor,
    candidate_valid: Tensor,
    required_count: Tensor,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    width_threshold_m: float,
    approach_threshold_deg: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """NMS verified grasps, then apply the adaptive unique-grasp count gate."""

    task = candidate_type == int(ActionType.TASK_GRASP)
    unique_task = task_grasp_nms(
        candidate_pose_world,
        candidate_width_m,
        candidate_score,
        candidate_object,
        candidate_valid & task,
        translation_threshold_m=translation_threshold_m,
        rotation_threshold_deg=rotation_threshold_deg,
        width_threshold_m=width_threshold_m,
        approach_threshold_deg=approach_threshold_deg,
    )
    unique_count = unique_task.sum(-1)
    count_gate = unique_count >= required_count
    deduplicated = candidate_valid & (~task | unique_task)
    return deduplicated & (~task | count_gate[:, None]), unique_count, unique_task


class CandidateCertifier(Protocol):
    def certify(self, action: dict[str, Any]) -> tuple[bool, str]: ...


@dataclass(slots=True)
class EncodedPolicyState:
    observation: SceneObservation
    cpu_batch: dict[str, Tensor]
    device_batch: dict[str, Tensor]
    output: dict[str, Any]


class TCDPRGPolicy(ManipulationPolicy):
    """Formal policy: predicted geometry/graph plus deterministic valid masks."""

    def __init__(
        self,
        model: TCDPRGModel,
        config: TCDPRGConfig,
        gripper_provider: Any | None = None,
        certifier: CandidateCertifier | None = None,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.device = next(model.parameters()).device
        self.generator = DenseCandidateGenerator(config.model)
        self.gripper_provider = gripper_provider
        self.certifier = certifier
        self.preparation_actions = 0

    def _batch(self, observation: SceneObservation) -> dict[str, Tensor]:
        observation.validate()
        point_valid = (
            observation.point_valid
            if observation.point_valid is not None
            else np.ones(len(observation.xyz), dtype=bool)
        )
        valid_index = np.flatnonzero(point_valid)
        selected = valid_index[grid_sample_indices(
            observation.xyz[valid_index], self.config.backbone.grid_size_m, training=False
        )]
        required_grasp_count = int(observation.metadata.get(
            "required_grasp_count", self.config.model.default_required_grasp_count
        ))
        if required_grasp_count > self.config.model.max_required_grasp_count:
            raise ValueError(
                f"required_grasp_count={required_grasp_count} exceeds declared "
                f"max_required_grasp_count={self.config.model.max_required_grasp_count}"
            )
        if required_grasp_count > self.config.model.task_grasp_candidates:
            raise ValueError(
                f"required_grasp_count={required_grasp_count} exceeds "
                f"task_grasp_candidates={self.config.model.task_grasp_candidates}"
            )
        cpu = {
            "xyz": torch.from_numpy(observation.xyz[selected])[None].float(),
            "rgb": torch.from_numpy(observation.rgb[selected])[None].float(),
            "instance_id": torch.from_numpy(observation.instance_id[selected])[None].long(),
            "point_mask": torch.ones(1, len(selected), dtype=torch.bool),
            "target_mask": torch.from_numpy(observation.target_mask[selected])[None].bool(),
            "target_object": torch.tensor([observation.target_object], dtype=torch.long),
            "object_mask": torch.from_numpy(observation.physical_active)[None].bool(),
            "object_present": torch.from_numpy(observation.object_present)[None].bool(),
            "object_active": torch.from_numpy(observation.object_active)[None].bool(),
            "task_category_id": torch.tensor(
                [observation.object_category_id[observation.target_object]], dtype=torch.long
            ),
            "task_region_id": torch.tensor([observation.task_region_id], dtype=torch.long),
            "remaining_steps": torch.tensor(
                [max(0, MAX_PREPARATION_ACTIONS - self.preparation_actions)], dtype=torch.long
            ),
            "required_grasp_count": torch.tensor([required_grasp_count], dtype=torch.long),
        }
        return cpu

    def encode_observation(self, observation: SceneObservation) -> EncodedPolicyState:
        # 观测只编码一次；候选生成、Verifier 和 Router 均复用 EncodedPolicyState。
        cpu = self._batch(observation)
        device = {key: value.to(self.device) for key, value in cpu.items()}
        with torch.no_grad():
            output = self.model(device)
        return EncodedPolicyState(observation, cpu, device, output)

    @staticmethod
    def _action(candidates: dict[str, Tensor], index: int) -> dict[str, Any]:
        def array(name: str):
            value = candidates[name][0, index].detach().cpu().numpy()
            return value.item() if value.ndim == 0 else value

        kind = int(array("type"))
        action = {
            "candidate_index": index,
            "action_type": kind,
            "acted_object": int(array("object")),
            "proposal_score": float(array("proposal_score")),
        }
        if kind == int(ActionType.PUSH):
            action.update(
                push_contact_world=array("contact_world"),
                push_direction_world=array("direction_world"),
                push_distance_m=float(array("push_distance_m")),
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
            candidates = self.generator.generate(
                self.model, encoded.device_batch, encoded.output
            )
            task_query_count = encoded.output["task_grasp"]["quality_logit"].shape[1]
            task_mask = candidates["type"] == int(ActionType.TASK_GRASP)
            candidates["task_grasp_query_count"] = torch.full(
                (candidates["type"].shape[0],), task_query_count,
                dtype=torch.long, device=self.device,
            )
            candidates["task_grasp_after_nms_count"] = (
                candidates["valid"] & task_mask
            ).sum(-1)
            if self.config.ablation.use_gripper_scene_verifier:
                if self.gripper_provider is None:
                    raise RuntimeError("Verifier-enabled inference requires AG geometry provider")
                verifier_batch = self.generator.verifier_batch(encoded.cpu_batch, candidates)
                verifier_inputs_cpu = build_verifier_inputs(
                    verifier_batch,
                    self.gripper_provider,
                    self.config.grasp_verifier.local_scene_points,
                    self.config.model.verifier_local_radius_m,
                )
                verifier_inputs = {
                    key: value.to(self.device) for key, value in verifier_inputs_cpu.items()
                }
                encoded.output["verifier"] = self.model.verify_cached(
                    encoded.device_batch, encoded.output, verifier_inputs
                )
                grasp = candidates["type"] != int(ActionType.PUSH)
                learned_valid = torch.sigmoid(
                    encoded.output["verifier"]["overall_logit"]
                ) >= self.config.model.verifier_validity_threshold
                if self.config.model.verifier_hard_gate:
                    candidates["valid"] &= ~grasp | learned_valid
            candidates["task_grasp_after_verifier_count"] = (
                candidates["valid"] & task_mask
            ).sum(-1)
            candidates["task_grasp_after_certifier_count"] = (
                candidates["valid"] & task_mask
            ).sum(-1)
            # Task Grasp 必须在 SE(3) NMS 后仍达到 required_grasp_count，才能进入 Router。
            (
                candidates["valid"], verified_count, unique_task_mask
            ) = apply_verified_candidate_count_gate(
                candidates["type"], candidates["object"], candidates["pose_world"],
                candidates["width_m"], candidates["proposal_score"], candidates["valid"],
                encoded.device_batch["required_grasp_count"],
                translation_threshold_m=self.config.model.grasp_nms_translation_m,
                rotation_threshold_deg=self.config.model.grasp_nms_rotation_deg,
                width_threshold_m=self.config.model.grasp_nms_width_m,
                approach_threshold_deg=self.config.model.grasp_nms_approach_deg,
            )
            candidates["verified_unique_grasp_count"] = verified_count
            # 兼容字段也采用“去重后的抓取数”语义，不能用原始 query 数代替。
            candidates["verified_candidate_count"] = verified_count
            candidates["task_grasp_nms_keep"] = unique_task_mask
            router = self.model.route_cached(
                encoded.device_batch, encoded.output, candidates
            )
            # PUSH NMS 以 Router 综合证据排序，去除同物体上接触点和方向近似的重复动作。
            nms_valid = self.generator.apply_push_nms(candidates, router.candidate_logits)
            candidates["push_after_nms_count"] = (
                nms_valid & (candidates["type"] == int(ActionType.PUSH))
            ).sum(-1)
            if not torch.equal(nms_valid, candidates["valid"]):
                candidates["valid"] = nms_valid
                router = self.model.route_cached(
                    encoded.device_batch, encoded.output, candidates
                )
        return {
            "encoded": encoded,
            "candidates": candidates,
            "router": router,
            "certification_reasons": [],
        }

    def select_action(self, candidates: dict[str, Any]) -> dict[str, Any] | None:
        tensors = candidates["candidates"]
        selected = MaskedHierarchicalCandidateRouter.select(
            candidates["router"], tensors["type"], tensors["object"]
        )
        if int(selected[0]) < 0:
            return None
        action = self._action(tensors, int(selected[0]))
        action["router_score"] = float(
            candidates["router"].candidate_logits[0, int(selected[0])]
        )
        return action

    def predict_grasps(self, encoded: EncodedPolicyState) -> list[dict[str, Any]]:
        candidates = self.generate_candidates(encoded)["candidates"]
        indices = torch.nonzero(
            candidates["valid"][0]
            & (candidates["type"][0] == int(ActionType.TASK_GRASP)),
            as_tuple=False,
        ).flatten()
        return [self._action(candidates, int(index)) for index in indices]

    def predict_task_grasps(self, encoded: EncodedPolicyState) -> list[dict[str, Any]]:
        return self.predict_grasps(encoded)

    def predict_global_grasps(
        self, encoded: EncodedPolicyState
    ) -> list[GlobalGraspPrediction]:
        decoded = self.generator.global_predictions(
            encoded.device_batch, encoded.output, self.config.model.candidate_topk
        )[0]
        predictions: list[GlobalGraspPrediction] = []
        for index in range(len(decoded["scene_score"])):
            predictions.append(GlobalGraspPrediction(
                object_index=int(decoded["object"][index]),
                contact_point_world=decoded["contact_world"][index].detach().cpu().numpy(),
                grasp_pose_world=decoded["pose_world"][index].detach().cpu().numpy(),
                width_m=float(decoded["width_m"][index]),
                raw_score=float(decoded["raw_score"][index]),
                scene_score=float(decoded["scene_score"][index]),
                intrinsic_score=None,
                certified=False,
                source="tcd_prg_global",
            ))
        return predictions

    def reset(self) -> None:
        self.preparation_actions = 0

    def update_after_action(self, action: Any, observation: SceneObservation) -> None:
        if isinstance(action, dict) and "action_type" in action:
            action_type = int(action["action_type"])
            if action_type != int(ActionType.TASK_GRASP):
                self.preparation_actions += 1
