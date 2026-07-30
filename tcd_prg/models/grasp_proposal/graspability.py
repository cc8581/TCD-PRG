"""State-level adaptive task-graspability prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StateGraspabilityHead(nn.Module):
    """Predict whether the state meets its required reliable-grasp count.

    ``verified_count_prediction`` is learned from state labels and is available
    at deployment without reading HDF5 truth.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU()
        )
        self.graspable = nn.Linear(dim, 1)
        self.count_log = nn.Linear(dim, 1)

    def forward(self, global_token: Tensor, target_token: Tensor, task_token: Tensor) -> dict[str, Tensor]:
        feature = self.shared(torch.cat((global_token, target_token, task_token), -1))
        count_log = torch.nn.functional.softplus(self.count_log(feature).squeeze(-1))
        return {
            "graspable_logit": self.graspable(feature).squeeze(-1),
            "verified_count_log_prediction": count_log,
            "verified_count_prediction": torch.expm1(count_log).clamp_min(0.0),
        }
