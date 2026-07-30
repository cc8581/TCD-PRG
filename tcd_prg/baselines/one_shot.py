"""One-shot open-loop sequence baseline for controlled comparison."""

from __future__ import annotations

from typing import Any

import torch

from tcd_prg.constants import ActionType, MAX_PREPARATION_ACTIONS
from tcd_prg.datasets.types import SceneObservation

from .base import ManipulationPolicy


class OneShotSequencePolicy(ManipulationPolicy):
    """Freeze the first observation and execute its top-ranked action sequence.

    This intentionally does not update candidates after actions.  It is the
    explicit open-loop comparison against TCD-PRG's re-observe/re-plan policy.
    """

    def __init__(self, policy: ManipulationPolicy,
                 horizon: int = MAX_PREPARATION_ACTIONS) -> None:
        self.policy = policy
        self.horizon = horizon
        self._sequence: list[dict[str, Any]] = []
        self._planned = False

    def encode_observation(self, observation: SceneObservation) -> Any:
        if not self._planned:
            return self.policy.encode_observation(observation)
        return None

    def generate_candidates(self, encoded: Any) -> Any:
        if self._planned:
            return self._sequence
        generated = self.policy.generate_candidates(encoded)
        tensors = generated.get("candidates") if isinstance(generated, dict) else None
        router = generated.get("router") if isinstance(generated, dict) else None
        if tensors is None or router is None:
            raise TypeError("OneShotSequencePolicy requires scored candidate tensors")
        valid = torch.nonzero(tensors["valid"][0], as_tuple=False).flatten()
        ranked = valid[router.candidate_logits[0, valid].argsort(descending=True)]
        selected_objects: set[int] = set()
        preparations = []
        task_grasps = []
        action_decoder = getattr(self.policy, "_action", None)
        if action_decoder is None:
            raise TypeError("Wrapped policy does not expose candidate action decoding")
        for index in ranked.tolist():
            action = action_decoder(tensors, index)
            if action["action_type"] == int(ActionType.TASK_GRASP):
                task_grasps.append(action)
            elif action["acted_object"] not in selected_objects and len(preparations) < self.horizon:
                selected_objects.add(action["acted_object"])
                preparations.append(action)
        self._sequence = preparations + task_grasps[:1]
        self._planned = True
        return self._sequence

    def select_action(self, candidates: Any) -> dict[str, Any] | None:
        return self._sequence.pop(0) if self._sequence else None

    def predict_grasps(self, encoded: Any) -> Any:
        return self.policy.predict_grasps(encoded)

    def predict_task_grasps(self, encoded: Any) -> Any:
        return self.policy.predict_task_grasps(encoded)

    def predict_global_grasps(self, encoded: Any) -> Any:
        return self.policy.predict_global_grasps(encoded)

    def reset(self) -> None:
        self._sequence.clear()
        self._planned = False
        self.policy.reset()

    def update_after_action(self, action: Any, observation: SceneObservation) -> None:
        # Deliberately do not re-encode the new observation.
        pass
