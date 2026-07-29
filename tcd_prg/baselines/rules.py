"""Rule routers over a shared learned candidate generator and safety mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tcd_prg.constants import ActionType
from tcd_prg.datasets.types import ActionCandidateGroup, SceneObservation

from .base import ManipulationPolicy


@dataclass(slots=True)
class RulePolicy(ManipulationPolicy):
    """Select by fixed action priority without bypassing candidate validity."""

    allowed: tuple[ActionType, ...]
    candidate_policy: ManipulationPolicy | None = None

    def encode_observation(self, observation: SceneObservation) -> Any:
        if self.candidate_policy is None:
            return observation
        return self.candidate_policy.encode_observation(observation)

    def generate_candidates(self, encoded: Any) -> Any:
        if self.candidate_policy is None:
            return encoded
        return self.candidate_policy.generate_candidates(encoded)

    def select_action(self, candidates: Any) -> Any:
        # Offline label-group compatibility.
        if isinstance(candidates, ActionCandidateGroup):
            for kind in self.allowed:
                index = np.flatnonzero(
                    candidates.valid_mask & (candidates.action_type == int(kind))
                )
                if len(index):
                    return int(candidates.candidate_action_ids[index[0]])
            return None
        if not isinstance(candidates, dict) or "candidates" not in candidates:
            raise TypeError("RulePolicy requires a generated candidate dictionary")
        tensors = candidates["candidates"]
        scores = candidates["router"].candidate_logits
        for kind in self.allowed:
            eligible = tensors["valid"][0] & (tensors["type"][0] == int(kind))
            index = torch.nonzero(eligible, as_tuple=False).flatten()
            if len(index):
                selected = int(index[scores[0, index].argmax()])
                decoder = getattr(self.candidate_policy, "_action", None)
                if decoder is None:
                    raise TypeError("Candidate policy cannot decode its candidate tensors")
                action = decoder(tensors, selected)
                action["router_score"] = float(scores[0, selected])
                return action
        return None

    def predict_grasps(self, encoded: Any) -> Any:
        return self.candidate_policy.predict_grasps(encoded) if self.candidate_policy else None

    def reset(self) -> None:
        if self.candidate_policy is not None:
            self.candidate_policy.reset()

    def update_after_action(self, action: Any, observation: SceneObservation) -> None:
        if self.candidate_policy is not None:
            self.candidate_policy.update_after_action(action, observation)


class DirectGraspOnlyPolicy(RulePolicy):
    def __init__(self, candidate_policy: ManipulationPolicy | None = None) -> None:
        super().__init__((ActionType.TASK_GRASP,), candidate_policy)


class PushOnlyPolicy(RulePolicy):
    def __init__(self, candidate_policy: ManipulationPolicy | None = None) -> None:
        super().__init__((ActionType.TASK_GRASP, ActionType.PUSH), candidate_policy)


class PickRemoveOnlyPolicy(RulePolicy):
    def __init__(self, candidate_policy: ManipulationPolicy | None = None) -> None:
        super().__init__((ActionType.TASK_GRASP, ActionType.PICK_REMOVE), candidate_policy)


class FixedPriorityPolicy(RulePolicy):
    def __init__(self, candidate_policy: ManipulationPolicy | None = None) -> None:
        super().__init__(
            (ActionType.TASK_GRASP, ActionType.PICK_REMOVE, ActionType.PUSH),
            candidate_policy,
        )
