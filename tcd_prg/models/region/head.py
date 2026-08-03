"""Visible functional-region segmentation on target-instance points only."""

import torch
from torch import Tensor, nn


class TaskRegionHead(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim // 2), nn.GELU())
        self.point_logit = nn.Linear(dim // 2, 1)
        self.visibility = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self, point_features: Tensor, target_token: Tensor, task_token: Tensor, target_mask: Tensor
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        context = torch.cat(
            (point_features, target_token[:, None].expand(-1, n, -1), task_token[:, None].expand(-1, n, -1)), -1
        )
        # 功能区域分割只在目标物体点上定义，其他点直接屏蔽而不是作为负样本监督。
        logits = self.point_logit(self.decoder(context)).squeeze(-1)
        logits = logits.masked_fill(~target_mask, -30.0)
        visibility_logit = self.visibility(torch.cat((target_token, task_token), -1)).squeeze(-1)
        return {"region_logits": logits, "region_probability": torch.sigmoid(logits), "visibility_logit": visibility_logit}
