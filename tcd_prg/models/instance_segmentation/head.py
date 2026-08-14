"""Lightweight class-agnostic instance queries on top of shared PTv3 point features.

The head predicts:
- unordered object queries,
- point-wise instance masks,
- objectness,
- closed-vocabulary category logits.

No GT instance/target mask is consumed here.  GT is used only by
`tcd_prg.losses.instance`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class InstanceQueryOutput:
    object_tokens: Tensor               # [B,Q,D]
    mask_logits: Tensor                 # [B,Q,N]
    mask_probability: Tensor            # [B,Q,N]
    objectness_logits: Tensor           # [B,Q]
    category_logits: Tensor             # [B,Q,C]
    object_mask: Tensor                 # [B,Q] runtime validity
    centers_world: Tensor               # [B,Q,3]


class InstanceQueryHead(nn.Module):
    """Small Mask2Former/DETR-style query decoder reusing scene point features."""

    def __init__(
        self,
        dim: int,
        queries: int,
        categories: int,
        layers: int = 2,
        heads: int = 8,
        objectness_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if queries <= 0:
            raise ValueError("instance_queries must be positive")
        if dim % heads:
            raise ValueError("feature_dim must be divisible by instance_decoder_heads")
        self.queries = int(queries)
        self.objectness_threshold = float(objectness_threshold)
        self.query = nn.Parameter(torch.empty(queries, dim))
        nn.init.normal_(self.query, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.mask_query = nn.Linear(dim, dim)
        self.mask_memory = nn.Linear(dim, dim)
        self.objectness = nn.Linear(dim, 1)
        self.category = nn.Linear(dim, categories)

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        point_mask: Tensor,
    ) -> InstanceQueryOutput:
        b, n, dim = point_features.shape
        query = self.query[None].expand(b, -1, -1)
        padding = ~point_mask.bool()

        # Defensive path for padded/empty rows. Formal observations must contain
        # at least one valid sensor point, but keeping this finite simplifies tests.
        all_invalid = padding.all(-1)
        memory = point_features
        if all_invalid.any():
            padding = padding.clone()
            memory = memory.clone()
            padding[all_invalid, 0] = False
            memory[all_invalid, 0] = 0.0

        object_tokens = self.decoder(
            query, memory, memory_key_padding_mask=padding
        )
        mask_logits = torch.einsum(
            "bqd,bnd->bqn",
            self.mask_query(object_tokens),
            self.mask_memory(memory),
        ) * dim ** -0.5
        mask_logits = mask_logits.masked_fill(~point_mask[:, None], -30.0)
        mask_probability = torch.sigmoid(mask_logits) * point_mask[:, None].to(mask_logits.dtype)

        objectness_logits = self.objectness(object_tokens).squeeze(-1)
        category_logits = self.category(object_tokens)

        probability = torch.sigmoid(objectness_logits)
        # Downstream Graph/Push only process confident predicted objects. Stage A
        # trains instance/objectness alone before these heads are enabled, so this
        # avoids Q*topk PUSH expansion without relying on any GT object count.
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
        )
