"""Object-centric PUSH prediction using predicted instance probabilities."""
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
        self.object_pointer = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.point = nn.Sequential(
            nn.Linear(4 * dim + 4, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
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
        instance_probability: Tensor,
        point_mask: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        task_token: Tensor,
        target_token: Tensor,
        graph_context: Tensor,
        remaining_steps: Tensor,
    ) -> dict[str, Tensor]:
        task = task_token[:, None].expand_as(object_tokens)
        object_logits = self.object_pointer(
            torch.cat((object_tokens, graph_context, task), -1)
        ).squeeze(-1)
        object_logits = object_logits.masked_fill(~object_mask, -30.0)

        assignment = (
            instance_probability
            * object_mask[:, :, None].to(instance_probability.dtype)
        )
        assignment = assignment / assignment.sum(
            1, keepdim=True
        ).clamp_min(1e-6)
        point_object = torch.einsum(
            "bqn,bqd->bnd", assignment, graph_context
        )

        n = xyz.shape[1]
        remaining = (
            remaining_steps[:, None, None].float().expand(-1, n, 1) / 5.0
        )
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
        contact = self.contact(x).squeeze(-1).masked_fill(
            ~point_mask, -30.0
        )

        direction_logits_rows: list[Tensor] = []
        direction_residual_rows: list[Tensor] = []
        utility_rows: list[Tensor] = []
        direction_point_masks: list[Tensor] = []

        for batch_row in range(x.shape[0]):
            selected_per_object: list[Tensor] = []
            active_objects = torch.nonzero(
                object_mask[batch_row], as_tuple=False
            ).flatten()
            valid_point_count = int(point_mask[batch_row].sum())
            for object_index in active_objects:
                membership = assignment[batch_row, object_index]
                # Joint contact/membership ranking; selection is prediction-only.
                joint = contact[batch_row] + membership.clamp_min(1e-6).log()
                joint = joint.masked_fill(~point_mask[batch_row], -30.0)
                count = min(
                    self.direction_contact_topk,
                    valid_point_count,
                )
                if count:
                    selected_per_object.append(
                        torch.topk(joint, k=count).indices
                    )
            selected = (
                torch.unique(torch.cat(selected_per_object), sorted=True)
                if selected_per_object
                else torch.empty(0, dtype=torch.long, device=x.device)
            )
            selected_mask = torch.zeros(
                n, dtype=torch.bool, device=x.device
            )
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
                direction_tokens = self.direction_transformer(
                    direction_tokens
                )
                logits = logits.index_copy(
                    0, selected,
                    self.direction_score(direction_tokens).squeeze(-1),
                )
                residual = residual.index_copy(
                    0, selected,
                    torch.tanh(self.direction_residual(direction_tokens)),
                )
                utility = utility.index_copy(
                    0, selected,
                    self.utility(direction_tokens).squeeze(-1),
                )
            direction_logits_rows.append(logits)
            direction_residual_rows.append(residual)
            utility_rows.append(utility)

        return {
            "object_logits": object_logits,
            "contact_logits": contact,
            "direction_logits": torch.stack(direction_logits_rows),
            "direction_residual": torch.stack(direction_residual_rows),
            "utility_delta": torch.stack(utility_rows),
            "direction_point_mask": torch.stack(direction_point_masks),
            "point_object_probability": assignment,
        }
