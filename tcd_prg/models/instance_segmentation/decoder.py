"""Query-based instance mask decoder for PTv3 point features.

PTv3 itself is kept as the official Pointcept encoder-decoder. This module is the
TCD-PRG-specific instance decoder that turns restored point features into unordered
object instances. It is intentionally deeper than the former two-layer query head
and exposes auxiliary predictions for deep supervision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class InstanceAuxOutput:
    mask_logits: Tensor
    objectness_logits: Tensor
    category_logits: Tensor


@dataclass(slots=True)
class InstanceQueryOutput:
    object_tokens: Tensor               # [B,Q,D]
    mask_logits: Tensor                 # [B,Q,N]
    mask_probability: Tensor            # [B,Q,N]
    objectness_logits: Tensor           # [B,Q]
    category_logits: Tensor             # [B,Q,C]
    object_mask: Tensor                 # [B,Q]
    centers_world: Tensor               # [B,Q,3]
    aux_outputs: tuple[InstanceAuxOutput, ...] = ()


class _DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_m = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim),
            nn.GELU(),
            nn.Linear(ffn_ratio * dim, dim),
        )

    def forward(
        self, query: Tensor, memory: Tensor, memory_padding: Tensor
    ) -> Tensor:
        q = self.cross_norm_q(query)
        m = self.cross_norm_m(memory)
        cross, _ = self.cross_attn(
            q, m, m, key_padding_mask=memory_padding, need_weights=False
        )
        query = query + cross
        q = self.self_norm(query)
        self_attn, _ = self.self_attn(q, q, q, need_weights=False)
        query = query + self_attn
        return query + self.ffn(self.ffn_norm(query))


class InstanceMaskDecoder(nn.Module):
    """Mask2Former-style object-query decoder on official PTv3 output features.

    The official PTv3 backbone is not reimplemented here. Only the task-specific
    instance decomposition decoder is learned by TCD-PRG.
    """

    def __init__(
        self,
        dim: int,
        queries: int,
        categories: int,
        layers: int = 6,
        heads: int = 8,
        objectness_threshold: float = 0.5,
        auxiliary_loss: bool = True,
    ) -> None:
        super().__init__()
        if queries <= 0:
            raise ValueError("instance_queries must be positive")
        if layers <= 0:
            raise ValueError("instance_decoder_layers must be positive")
        if dim % heads:
            raise ValueError("feature_dim must be divisible by instance_decoder_heads")
        self.queries = int(queries)
        self.objectness_threshold = float(objectness_threshold)
        self.auxiliary_loss = bool(auxiliary_loss)

        self.query_embed = nn.Parameter(torch.empty(queries, dim))
        self.query_pos = nn.Parameter(torch.empty(queries, dim))
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)

        self.memory_projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(_DecoderBlock(dim, heads) for _ in range(layers))
        self.output_norm = nn.LayerNorm(dim)

        self.mask_query = nn.Linear(dim, dim)
        self.mask_memory = nn.Linear(dim, dim)
        self.objectness = nn.Linear(dim, 1)
        self.category = nn.Linear(dim, categories)

    def _predict(
        self,
        query: Tensor,
        memory: Tensor,
        point_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dim = query.shape[-1]
        normalized = self.output_norm(query)
        mask_logits = torch.einsum(
            "bqd,bnd->bqn",
            self.mask_query(normalized),
            self.mask_memory(memory),
        ) * dim ** -0.5
        mask_logits = mask_logits.masked_fill(~point_mask[:, None], -30.0)
        return (
            mask_logits,
            self.objectness(normalized).squeeze(-1),
            self.category(normalized),
        )

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        point_mask: Tensor,
    ) -> InstanceQueryOutput:
        b, _, _ = point_features.shape
        memory = self.memory_projection(point_features)
        padding = ~point_mask.bool()
        all_invalid = padding.all(-1)
        if all_invalid.any():
            padding = padding.clone()
            memory = memory.clone()
            padding[all_invalid, 0] = False
            memory[all_invalid, 0] = 0.0

        query = (self.query_embed + self.query_pos)[None].expand(b, -1, -1)
        auxiliary: list[InstanceAuxOutput] = []
        for layer_index, block in enumerate(self.blocks):
            query = block(query, memory, padding)
            if self.auxiliary_loss and layer_index < len(self.blocks) - 1:
                mask, obj, cat = self._predict(query, memory, point_mask)
                auxiliary.append(InstanceAuxOutput(mask, obj, cat))

        object_tokens = self.output_norm(query)
        mask_logits, objectness_logits, category_logits = self._predict(
            query, memory, point_mask
        )
        mask_probability = (
            torch.sigmoid(mask_logits) * point_mask[:, None].to(mask_logits.dtype)
        )

        probability = torch.sigmoid(objectness_logits)
        object_mask = probability >= self.objectness_threshold
        fallback = probability.argmax(-1)
        empty = ~object_mask.any(-1)
        if empty.any():
            object_mask = object_mask.clone()
            object_mask[empty, fallback[empty]] = True

        mass = mask_probability.sum(-1, keepdim=True).clamp_min(1e-6)
        centers = torch.einsum("bqn,bnd->bqd", mask_probability, xyz) / mass
        return InstanceQueryOutput(
            object_tokens=object_tokens,
            mask_logits=mask_logits,
            mask_probability=mask_probability,
            objectness_logits=objectness_logits,
            category_logits=category_logits,
            object_mask=object_mask,
            centers_world=centers,
            aux_outputs=tuple(auxiliary),
        )
