"""Query-based complete 6D grasp set prediction heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcd_prg.geometry.se3 import rotation_6d_to_matrix


def _attention_heads(dim: int) -> int:
    for heads in (8, 4, 2):
        if dim % heads == 0:
            return heads
    return 1


class _CompleteGraspSetHead(nn.Module):
    """Decode fixed learned queries into complete ``(t, R, w, q)`` grasps."""

    def __init__(self, dim: int, queries: int, context_dim: int) -> None:
        super().__init__()
        if queries <= 0:
            raise ValueError("Complete grasp prediction requires at least one query")
        self.query_embedding = nn.Parameter(torch.empty(queries, dim))
        nn.init.normal_(self.query_embedding, std=0.02)
        self.context = nn.Sequential(nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory = nn.Sequential(nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cross_attention = nn.MultiheadAttention(
            dim, _attention_heads(dim), batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU()
        )
        self.translation_offset = nn.Linear(dim, 3)
        self.rotation_6d = nn.Linear(dim, 6)
        self.width = nn.Linear(dim, 1)
        self.quality = nn.Linear(dim, 1)

    def _decode(
        self, memory_input: Tensor, xyz: Tensor, point_domain: Tensor, context: Tensor,
        object_index: Tensor | None = None, object_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b = xyz.shape[0]
        memory = self.memory(memory_input)
        query = self.query_embedding[None].expand(b, -1, -1) + self.context(context)[:, None]
        padding = ~point_domain.bool()
        all_invalid = padding.all(-1)
        if all_invalid.any():
            padding = padding.clone()
            memory = memory.clone()
            padding[all_invalid, 0] = False
            memory[all_invalid, 0] = 0.0
        attended, weights = self.cross_attention(
            query, memory, memory, key_padding_mask=padding, need_weights=True
        )
        decoded = self.decoder(self.norm(query + attended))
        anchor = torch.bmm(weights, xyz)
        translation = anchor + 0.10 * torch.tanh(self.translation_offset(decoded))
        rotation_6d = self.rotation_6d(decoded)
        rotation = rotation_6d_to_matrix(rotation_6d)
        output = {
            "translation_world": translation,
            "rotation_matrix": rotation,
            "rotation_6d": rotation_6d,
            "width_raw": self.width(decoded).squeeze(-1),
            "quality_logit": self.quality(decoded).squeeze(-1),
            "attention_point_index": weights.argmax(-1),
        }
        if object_index is not None:
            if object_mask is None:
                raise ValueError("object_mask is required with object_index")
            object_count = object_mask.shape[1]
            valid_object_point = (
                point_domain & (object_index >= 0) & (object_index < object_count)
            )
            safe_object = object_index.clamp(0, object_count - 1)
            object_mass = weights.new_zeros((b, weights.shape[1], object_count))
            object_mass.scatter_add_(
                2,
                safe_object[:, None].expand(-1, weights.shape[1], -1),
                weights * valid_object_point[:, None],
            )
            visible_count = torch.zeros_like(object_mask, dtype=torch.long)
            visible_count.scatter_add_(1, safe_object, valid_object_point.long())
            valid_object = object_mask & (visible_count > 0)
            output["object_logits"] = torch.log(object_mass.clamp_min(1e-8)).masked_fill(
                ~valid_object[:, None], -30.0
            )
        return output

    @staticmethod
    def decode_width(width_raw: Tensor, min_width_m: float, max_width_m: float) -> Tensor:
        return min_width_m + torch.sigmoid(width_raw) * (max_width_m - min_width_m)


class TaskGraspProposalHead(_CompleteGraspSetHead):
    """Predict task-conditioned grasps from scene, task and functional region."""

    def __init__(self, dim: int = 256, queries: int = 32) -> None:
        super().__init__(dim, queries, 3 * dim + 1)

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        target_token: Tensor,
        task_token: Tensor,
        region_probability: Tensor,
        target_mask: Tensor,
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        memory = torch.cat((
            point_features,
            target_token[:, None].expand(-1, n, -1),
            task_token[:, None].expand(-1, n, -1),
            region_probability.unsqueeze(-1),
        ), -1)
        region_weight = region_probability * target_mask
        region_context = (
            point_features * region_weight.unsqueeze(-1)
        ).sum(1) / region_weight.sum(1, keepdim=True).clamp_min(1e-6)
        region_context = torch.where(
            region_weight.any(-1, keepdim=True), region_context, target_token
        )
        visibility = region_weight.sum(-1, keepdim=True) / target_mask.sum(
            -1, keepdim=True
        ).clamp_min(1)
        context = torch.cat((target_token, task_token, region_context, visibility), -1)
        return self._decode(memory, xyz, target_mask, context)


class GlobalGraspProposalHead(_CompleteGraspSetHead):
    """Predict a task-free scene grasp set using only neutral scene features."""

    VALID_INPUT_MODES = {"scene_only", "instance_assisted"}

    def __init__(self, dim: int = 256, queries: int = 64, input_mode: str = "scene_only") -> None:
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(f"Unsupported global grasp input mode: {input_mode}")
        super().__init__(dim, queries, 3 * dim)
        self.input_mode = input_mode

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        object_tokens: Tensor,
        global_scene_token: Tensor,
        instance_id: Tensor,
        point_domain: Tensor,
        object_mask: Tensor,
    ) -> dict[str, Tensor]:
        b, n, _ = point_features.shape
        if self.input_mode == "instance_assisted":
            row = torch.arange(b, device=point_features.device)[:, None]
            per_point_object = object_tokens[row, instance_id.clamp(0, object_tokens.shape[1] - 1)]
        else:
            per_point_object = torch.zeros_like(point_features)
        memory = torch.cat((
            point_features,
            per_point_object,
            global_scene_token[:, None].expand(-1, n, -1),
        ), -1)
        context = torch.cat((global_scene_token, global_scene_token, global_scene_token), -1)
        output = self._decode(
            memory, xyz, point_domain, context,
            object_index=instance_id, object_mask=object_mask,
        )
        output["point_domain"] = point_domain
        return output
