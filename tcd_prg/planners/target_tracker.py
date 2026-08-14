"""Lightweight cross-frame target identity continuity for closed-loop execution."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class TargetIdentityState:
    token: Tensor       # CPU [D]
    center_world: Tensor  # CPU [3]
    category_id: int
    query_id: int
    selection_margin: float
    prompt_support: float


class TargetIdentityTracker:
    """Stores the selected target descriptor between re-observations.

    The tracker never assumes query ids are stable across frames.  On the next
    frame the model receives the previous token/center as a continuity prior and
    reselects among the newly predicted queries.
    """

    def __init__(self) -> None:
        self.state: TargetIdentityState | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, encoded, category_id: int) -> TargetIdentityState:
        query = int(encoded.target_query_index[0].detach().cpu())
        token = encoded.scene_object_tokens[0, query].detach().float().cpu().clone()
        center = encoded.instance.centers_world[0, query].detach().float().cpu().clone()
        support = float(encoded.target_prompt_support[0, query].detach().cpu())
        margin = float(encoded.target_selection_margin[0].detach().cpu())
        self.state = TargetIdentityState(
            token=token,
            center_world=center,
            category_id=int(category_id),
            query_id=query,
            selection_margin=margin,
            prompt_support=support,
        )
        return self.state

    def task_inputs(self) -> dict[str, Tensor]:
        if self.state is None:
            return {}
        return {
            "target_reid_token": self.state.token[None].clone(),
            "target_reid_center": self.state.center_world[None].clone(),
            "target_reid_valid": torch.ones(1, dtype=torch.bool),
        }
