"""Task-conditioned functional-region segmentation on the predicted target instance."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TaskRegionHead(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        self.point_logit = nn.Linear(dim // 2, 1)
        self.visibility = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(
        self,
        point_features: Tensor,
        target_token: Tensor,
        task_token: Tensor,
        target_probability: Tensor,
        point_mask: Tensor,
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        context = torch.cat(
            (
                point_features,
                target_token[:, None].expand(-1, n, -1),
                task_token[:, None].expand(-1, n, -1),
            ),
            -1,
        )
        # Raw logits are supervised by region_target/region_valid on the loss side.
        # Prediction is softly constrained by the model-predicted target instance;
        # no GT target mask enters this forward pass.
        raw_logits = self.point_logit(self.decoder(context)).squeeze(-1)
        raw_logits = raw_logits.masked_fill(~point_mask, -30.0)
        raw_probability = torch.sigmoid(raw_logits)
        region_probability = (
            raw_probability
            * target_probability.clamp(0.0, 1.0)
            * point_mask.to(raw_probability.dtype)
        )
        visibility_logit = self.visibility(
            torch.cat((target_token, task_token), -1)
        ).squeeze(-1)
        return {
            "region_logits": raw_logits,
            "region_probability": region_probability,
            "raw_region_probability": raw_probability,
            "visibility_logit": visibility_logit,
        }
