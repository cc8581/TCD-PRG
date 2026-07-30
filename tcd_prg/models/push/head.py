"""Fixed-distance PUSH object/contact/direction with potential/risk auxiliaries."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PushHead(nn.Module):
    RISK_NAMES = ("unstable", "out_of_workspace", "other_invalid")

    def __init__(self, dim: int = 256, direction_bins: int = 16, potential_dim: int = 5) -> None:
        super().__init__()
        self.direction_bins = direction_bins
        self.object_pointer = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.point = nn.Sequential(nn.Linear(4 * dim + 4, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.contact = nn.Linear(dim, 1)
        self.direction = nn.Linear(dim, direction_bins)
        self.direction_residual = nn.Linear(dim, 2)
        # Contact establishment and the fixed 0.15 m trajectory belong to the
        # deterministic execution layer.  The learned PUSH primitive predicts
        # only the acted object, contact and planar direction, with optional
        # potential/risk auxiliaries.
        self.potential = nn.Linear(dim, potential_dim)
        self.risk = nn.Linear(dim, len(self.RISK_NAMES))

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        instance_id: Tensor,
        point_mask: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        task_token: Tensor,
        target_token: Tensor,
        graph_context: Tensor,
        remaining_steps: Tensor,
    ) -> dict[str, Tensor]:
        task = task_token[:, None].expand_as(object_tokens)
        object_logits = self.object_pointer(torch.cat((object_tokens, graph_context, task), -1)).squeeze(-1)
        object_logits = object_logits.masked_fill(~object_mask, -30.0)
        row = torch.arange(xyz.shape[0], device=xyz.device)[:, None]
        safe_instance = instance_id.clamp(0, object_tokens.shape[1] - 1)
        point_object = graph_context[row, safe_instance]
        n = xyz.shape[1]
        remaining = remaining_steps[:, None, None].float().expand(-1, n, 1) / 5.0
        x = torch.cat(
            (
                point_features,
                point_object,
                task_token[:, None].expand(-1, n, -1),
                target_token[:, None].expand(-1, n, -1),
                xyz,
                remaining,
            ),
            -1,
        )
        x = self.point(x)
        contact = self.contact(x).squeeze(-1).masked_fill(~point_mask, -30.0)
        return {
            "object_logits": object_logits,
            "contact_logits": contact,
            "direction_logits": self.direction(x),
            "direction_residual": torch.tanh(self.direction_residual(x)),
            "potential_delta": self.potential(x),
            "risk_logits": self.risk(x),
        }
