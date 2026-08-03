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
        direction_contact_topk: int = 32,
    ) -> None:
        super().__init__()
        self.direction_bins = direction_bins
        self.direction_contact_topk = direction_contact_topk
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
        forced_direction_points: tuple[Tensor, ...] | None = None,
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
        # 方向分支只处理高分接触点；训练真值点会被显式并入，监督语义保持不变。
        direction_logits_rows: list[Tensor] = []
        direction_residual_rows: list[Tensor] = []
        utility_rows: list[Tensor] = []
        direction_point_masks: list[Tensor] = []
        for batch_row in range(x.shape[0]):
            selected_per_object: list[Tensor] = []
            for object_index in torch.nonzero(
                object_mask[batch_row], as_tuple=False
            ).flatten():
                object_points = torch.nonzero(
                    point_mask[batch_row] & (instance_id[batch_row] == object_index),
                    as_tuple=False,
                ).flatten()
                count = min(self.direction_contact_topk, int(object_points.numel()))
                if count:
                    selected_per_object.append(object_points[
                        torch.topk(contact[batch_row, object_points], k=count).indices
                    ])
            selected = (
                torch.cat(selected_per_object)
                if selected_per_object
                else torch.empty(0, dtype=torch.long, device=x.device)
            )
            if forced_direction_points is not None:
                forced = forced_direction_points[batch_row].to(
                    device=x.device, dtype=torch.long
                )
                in_range = (forced >= 0) & (forced < n)
                forced = forced[in_range]
                forced = forced[point_mask[batch_row, forced]]
                selected = torch.unique(torch.cat((selected, forced)), sorted=True)

            selected_mask = torch.zeros(n, dtype=torch.bool, device=x.device)
            selected_mask[selected] = True
            direction_point_masks.append(selected_mask)
            logits = x.new_full((n, self.direction_bins), -30.0)
            residual = x.new_zeros((n, self.direction_bins, 2))
            utility = x.new_zeros((n, self.direction_bins))
            if selected.numel():
                direction_tokens = (
                    self.direction_context(x[batch_row, selected])[:, None]
                    + self.direction_embedding.weight[None]
                )
                direction_tokens = self.direction_transformer(direction_tokens)
                logits = logits.index_copy(
                    0, selected, self.direction_score(direction_tokens).squeeze(-1)
                )
                residual = residual.index_copy(
                    0, selected, torch.tanh(self.direction_residual(direction_tokens))
                )
                utility = utility.index_copy(
                    0, selected, self.utility(direction_tokens).squeeze(-1)
                )
            direction_logits_rows.append(logits)
            direction_residual_rows.append(residual)
            utility_rows.append(utility)

        direction_logits = torch.stack(direction_logits_rows)
        return {
            "object_logits": object_logits,
            "contact_logits": contact,
            "direction_logits": direction_logits,
            "direction_residual": torch.stack(direction_residual_rows),
            "utility_delta": torch.stack(utility_rows),
            "direction_point_mask": torch.stack(direction_point_masks),
        }
