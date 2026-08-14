"""Query-based complete 6D grasp set prediction using predicted object instances."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from tcd_prg.geometry.se3 import rotation_6d_to_matrix


class M2T2GraspDecoder(nn.Module):
    def __init__(self, dim: int, layers: int = 3, heads: int = 8) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, layers, norm=nn.LayerNorm(dim)
        )

    def forward(
        self, query: Tensor, memory: Tensor, memory_padding_mask: Tensor
    ) -> Tensor:
        return self.decoder(
            query, memory, memory_key_padding_mask=memory_padding_mask
        )


class _CompleteGraspSetHead(nn.Module):
    def __init__(self, dim: int, queries: int, context_dim: int) -> None:
        super().__init__()
        if queries <= 0:
            raise ValueError("Complete grasp prediction requires at least one query")
        self.query_embedding = nn.Parameter(torch.empty(queries, dim))
        nn.init.normal_(self.query_embedding, std=0.02)
        self.context = nn.Sequential(
            nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.memory = nn.Sequential(
            nn.Linear(context_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.mask_query = nn.Linear(dim, dim)
        self.mask_memory = nn.Linear(dim, dim)
        self.translation_offset = nn.Linear(dim, 3)
        self.rotation_6d = nn.Linear(dim, 6)
        self.width = nn.Linear(dim, 1)
        self.quality = nn.Linear(dim, 1)

    def _decode(
        self,
        memory_input: Tensor,
        xyz: Tensor,
        point_domain: Tensor,
        context: Tensor,
        decoder: M2T2GraspDecoder,
        *,
        point_prior: Tensor | None = None,
        object_probability: Tensor | None = None,
        object_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b = xyz.shape[0]
        memory = self.memory(memory_input)
        query = (
            self.query_embedding[None].expand(b, -1, -1)
            + self.context(context)[:, None]
        )
        padding = ~point_domain.bool()
        all_invalid = padding.all(-1)
        if all_invalid.any():
            padding = padding.clone()
            memory = memory.clone()
            padding[all_invalid, 0] = False
            memory[all_invalid, 0] = 0.0

        decoded = decoder(query, memory, padding)
        mask_logits = torch.einsum(
            "bqd,bnd->bqn",
            self.mask_query(decoded),
            self.mask_memory(memory),
        ) * decoded.shape[-1] ** -0.5
        if point_prior is not None:
            prior = point_prior.clamp_min(1e-6).log()
            mask_logits = mask_logits + prior[:, None]
        mask_logits = mask_logits.masked_fill(padding[:, None], -30.0)
        weights = torch.softmax(mask_logits, -1)

        anchor = torch.bmm(weights, xyz)
        translation = anchor + 0.10 * torch.tanh(
            self.translation_offset(decoded)
        )
        rotation_6d = self.rotation_6d(decoded)
        rotation = rotation_6d_to_matrix(rotation_6d)
        output = {
            "translation_world": translation,
            "rotation_matrix": rotation,
            "rotation_6d": rotation_6d,
            "width_raw": self.width(decoded).squeeze(-1),
            "quality_logit": self.quality(decoded).squeeze(-1),
            "attention_point_index": weights.argmax(-1),
            "point_attention": weights,
        }

        if object_probability is not None:
            if object_mask is None:
                raise ValueError(
                    "object_mask is required with object_probability"
                )
            # Grasp-to-object association is derived from predicted instance masks.
            # weights: [B,K,N], object_probability: [B,Q,N] -> [B,K,Q]
            object_mass = torch.einsum(
                "bkn,bqn->bkq", weights, object_probability
            )
            object_mass = object_mass / object_mass.sum(
                -1, keepdim=True
            ).clamp_min(1e-8)
            output["object_logits"] = torch.log(
                object_mass.clamp_min(1e-8)
            ).masked_fill(~object_mask[:, None], -30.0)
        return output

    @staticmethod
    def decode_width(
        width_raw: Tensor, min_width_m: float, max_width_m: float
    ) -> Tensor:
        return min_width_m + torch.sigmoid(width_raw) * (
            max_width_m - min_width_m
        )


class TaskGraspProposalHead(_CompleteGraspSetHead):
    def __init__(self, dim: int = 256, queries: int = 32) -> None:
        super().__init__(dim, queries, 3 * dim + 1)

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        target_token: Tensor,
        task_token: Tensor,
        region_probability: Tensor,
        target_probability: Tensor,
        point_mask: Tensor,
        decoder: M2T2GraspDecoder,
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        memory = torch.cat(
            (
                point_features,
                target_token[:, None].expand(-1, n, -1),
                task_token[:, None].expand(-1, n, -1),
                region_probability.unsqueeze(-1),
            ),
            -1,
        )
        region_weight = (
            region_probability * target_probability * point_mask
        )
        region_context = (
            point_features * region_weight.unsqueeze(-1)
        ).sum(1) / region_weight.sum(1, keepdim=True).clamp_min(1e-6)
        region_context = torch.where(
            region_weight.any(-1, keepdim=True),
            region_context,
            target_token,
        )
        visibility = region_weight.sum(-1, keepdim=True) / (
            target_probability * point_mask
        ).sum(-1, keepdim=True).clamp_min(1e-6)
        context = torch.cat(
            (target_token, task_token, region_context, visibility), -1
        )
        return self._decode(
            memory,
            xyz,
            point_mask,
            context,
            decoder,
            point_prior=(
                target_probability
                * (0.25 + 0.75 * region_probability)
                * point_mask
            ),
        )


class GlobalGraspProposalHead(_CompleteGraspSetHead):
    """Task-free grasp prediction with predicted instance association."""

    VALID_INPUT_MODES = {"scene_only", "instance_assisted"}

    def __init__(
        self, dim: int = 256, queries: int = 64,
        input_mode: str = "scene_only",
    ) -> None:
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"Unsupported global grasp input mode: {input_mode}"
            )
        super().__init__(dim, queries, 3 * dim)
        self.input_mode = input_mode

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        object_tokens: Tensor,
        global_scene_token: Tensor,
        instance_probability: Tensor,
        point_domain: Tensor,
        object_mask: Tensor,
        decoder: M2T2GraspDecoder,
    ) -> dict[str, Tensor]:
        b, n, _ = point_features.shape
        if self.input_mode == "instance_assisted":
            assignment = (
                instance_probability
                * object_mask[:, :, None].to(instance_probability.dtype)
            )
            assignment = assignment / assignment.sum(
                1, keepdim=True
            ).clamp_min(1e-6)
            per_point_object = torch.einsum(
                "bqn,bqd->bnd", assignment, object_tokens
            )
        else:
            per_point_object = torch.zeros_like(point_features)

        memory = torch.cat(
            (
                point_features,
                per_point_object,
                global_scene_token[:, None].expand(-1, n, -1),
            ),
            -1,
        )
        context = torch.cat(
            (global_scene_token, global_scene_token, global_scene_token), -1
        )
        output = self._decode(
            memory,
            xyz,
            point_domain,
            context,
            decoder,
            object_probability=instance_probability,
            object_mask=object_mask,
        )
        output["point_domain"] = point_domain
        return output
