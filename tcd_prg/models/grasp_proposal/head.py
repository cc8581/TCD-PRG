"""Task-conditioned 6D grasp proposal head compatible with GraspNet concepts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TaskGraspProposalHead(nn.Module):
    def __init__(self, dim: int = 256, rotation_bins: int = 12, depth_bins: int = 4) -> None:
        super().__init__()
        self.rotation_bins = rotation_bins
        self.depth_bins = depth_bins
        self.shared = nn.Sequential(nn.Linear(3 * dim + 1, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU())
        self.contact = nn.Linear(dim, 1)
        self.approach = nn.Linear(dim, 3)
        self.rotation = nn.Linear(dim, rotation_bins)
        self.depth = nn.Linear(dim, depth_bins)
        self.width = nn.Linear(dim, 1)
        self.confidence = nn.Linear(dim, 1)
        self.task_compatibility = nn.Linear(dim, 1)

    def forward(
        self,
        point_features: Tensor,
        target_token: Tensor,
        task_token: Tensor,
        region_probability: Tensor,
        target_mask: Tensor,
        generic_remove: bool = False,
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        task_context = torch.zeros_like(task_token) if generic_remove else task_token
        x = torch.cat(
            (
                point_features,
                target_token[:, None].expand(-1, n, -1),
                task_context[:, None].expand(-1, n, -1),
                region_probability.unsqueeze(-1),
            ),
            -1,
        )
        x = self.shared(x)
        contact = self.contact(x).squeeze(-1).masked_fill(~target_mask, -30.0)
        approach = torch.nn.functional.normalize(self.approach(x), dim=-1, eps=1e-6)
        return {
            "contact_logits": contact,
            "approach_direction": approach,
            "rotation_logits": self.rotation(x),
            "depth_logits": self.depth(x),
            "width_raw": self.width(x).squeeze(-1),
            "proposal_confidence_logit": self.confidence(x).squeeze(-1),
            "task_compatibility_logit": self.task_compatibility(x).squeeze(-1),
        }

    @staticmethod
    def decode_width(width_raw: Tensor, min_width_m: float, max_width_m: float) -> Tensor:
        return min_width_m + torch.sigmoid(width_raw) * (max_width_m - min_width_m)

    def topk(self, output: dict[str, Tensor], xyz: Tensor, k: int) -> dict[str, Tensor]:
        score = (
            torch.sigmoid(output["contact_logits"])
            * torch.sigmoid(output["proposal_confidence_logit"])
            * torch.sigmoid(output["task_compatibility_logit"])
        )
        k = min(k, score.shape[1])
        values, index = score.topk(k, dim=1)
        row = torch.arange(score.shape[0], device=score.device)[:, None]
        return {
            "point_index": index,
            "score": values,
            "contact_point": xyz[row, index],
            "approach_direction": output["approach_direction"][row, index],
            "rotation_logits": output["rotation_logits"][row, index],
            "depth_logits": output["depth_logits"][row, index],
            "width_raw": output["width_raw"][row, index],
        }

