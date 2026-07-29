"""Closed-loop TCD-PRG policy using one shared scene encoding per observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from tcd_prg.baselines.base import ManipulationPolicy
from tcd_prg.config import TCDPRGConfig
from tcd_prg.constants import ActionType, MAX_PREPARATION_ACTIONS
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.grasp_verifier import build_verifier_inputs
from tcd_prg.models.policy.router import MaskedHierarchicalCandidateRouter

from .candidate_generator import DenseCandidateGenerator


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
        self.previous_action: int | None = None

    def _batch(self, observation: SceneObservation) -> dict[str, Tensor]:
        observation.validate()
        point_valid = (
            observation.point_valid
            if observation.point_valid is not None
            else np.ones(len(observation.xyz), dtype=bool)
        )
        cpu = {
            "xyz": torch.from_numpy(observation.xyz)[None].float(),
            "rgb": torch.from_numpy(observation.rgb)[None].float(),
            "instance_id": torch.from_numpy(observation.instance_id)[None].long(),
            "point_mask": torch.from_numpy(point_valid)[None].bool(),
            "target_mask": torch.from_numpy(observation.target_mask)[None].bool(),
            "target_object": torch.tensor([observation.target_object], dtype=torch.long),
            "object_pose": torch.from_numpy(observation.object_pose)[None].float(),
            "object_mask": torch.from_numpy(observation.object_present)[None].bool(),
            "object_active": torch.from_numpy(observation.object_active)[None].bool(),
            "task_category_id": torch.tensor(
                [observation.object_category_id[observation.target_object]], dtype=torch.long
            ),
            "task_region_id": torch.tensor([observation.task_region_id], dtype=torch.long),
            "remaining_steps": torch.tensor(
                [max(0, MAX_PREPARATION_ACTIONS - self.preparation_actions)], dtype=torch.long
            ),
        }
        return cpu

    def encode_observation(self, observation: SceneObservation) -> EncodedPolicyState:
        if self.certifier is not None and hasattr(self.certifier, "set_observation"):
            self.certifier.set_observation(observation)  # type: ignore[attr-defined]
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
                push_approach_mode=int(array("push_approach_mode")),
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
            if self.previous_action is not None:
                candidates["previous_action"] = torch.tensor(
                    [self.previous_action], device=self.device
                )
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
                task = candidates["type"] == int(ActionType.TASK_GRASP)
                task_valid = torch.sigmoid(
                    encoded.output["verifier"]["task_compatibility_logit"]
                ) >= self.config.model.verifier_validity_threshold
                candidates["valid"] &= ~grasp | (learned_valid & (~task | task_valid))
            certification_reasons: list[str] = []
            if self.certifier is not None:
                valid_indices = [index for index in range(candidates["type"].shape[1])
                                 if bool(candidates["valid"][0, index])]
                actions = [self._action(candidates, index) for index in valid_indices]
                if actions:
                    decisions = (
                        self.certifier.certify_many(actions)  # type: ignore[attr-defined]
                        if hasattr(self.certifier, "certify_many")
                        else [self.certifier.certify(action) for action in actions]
                    )
                else:
                    decisions = []
                decision_by_index = dict(zip(valid_indices, decisions, strict=True))
                for index in range(candidates["type"].shape[1]):
                    if index not in decision_by_index:
                        certification_reasons.append("masked_by_model")
                        continue
                    ok, reason = decision_by_index[index]
                    candidates["valid"][0, index] = ok
                    certification_reasons.append(reason)
            router = self.model.route_cached(
                encoded.device_batch, encoded.output, candidates
            )
        return {
            "encoded": encoded,
            "candidates": candidates,
            "router": router,
            "certification_reasons": certification_reasons,
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

    def reset(self) -> None:
        self.preparation_actions = 0
        self.previous_action = None

    def update_after_action(self, action: Any, observation: SceneObservation) -> None:
        if isinstance(action, dict) and "action_type" in action:
            self.previous_action = int(action["action_type"])
            if self.previous_action != int(ActionType.TASK_GRASP):
                self.preparation_actions += 1
