"""Fixed-distance PUSH object/contact/direction and complete utility change."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PushHead(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        direction_bins: int = 16,
        direction_dim: int = 64,
        direction_layers: int = 1,
        direction_heads: int = 4,
    ) -> None:
        super().__init__()
        self.direction_bins = direction_bins
        self.object_pointer = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.point = nn.Sequential(nn.Linear(4 * dim + 4, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.contact = nn.Linear(dim, 1)
        self.direction_context = nn.Linear(dim, direction_dim)
        self.direction_embedding = nn.Embedding(direction_bins, direction_dim)
        layer = nn.TransformerEncoderLayer(
            direction_dim,
            direction_heads,
            4 * direction_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.direction_transformer = nn.TransformerEncoder(
            layer, direction_layers, norm=nn.LayerNorm(direction_dim)
        )
        self.direction_score = nn.Linear(direction_dim, 1)
        self.direction_residual = nn.Linear(direction_dim, 2)
        self.utility = nn.Linear(direction_dim, 1)

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
        # 每个接触点与全部方向 bin 组成条件 token，各 bin 独立预测分数、残差和 utility。
        direction_tokens = (
            self.direction_context(x)[:, :, None]
            + self.direction_embedding.weight[None, None]
        )
        direction_tokens = self.direction_transformer(
            direction_tokens.flatten(0, 1)
        ).reshape(x.shape[0], x.shape[1], self.direction_bins, -1)
        direction_logits = self.direction_score(direction_tokens).squeeze(-1)
        direction_logits = direction_logits.masked_fill(~point_mask[:, :, None], -30.0)
        return {
            "object_logits": object_logits,
            "contact_logits": contact,
            "direction_logits": direction_logits,
            "direction_residual": torch.tanh(self.direction_residual(direction_tokens)),
            "utility_delta": self.utility(direction_tokens).squeeze(-1),
        }
