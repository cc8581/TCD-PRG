"""Masked pooling primitives for points, objects and candidates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    mask = mask.bool()
    minimum = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, minimum)
    probabilities = torch.softmax(logits, dim=dim)
    probabilities = probabilities * mask.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class MaskedAttentionPool(nn.Module):
    """Query-conditioned attention pooling over ``[B,N,C]`` tokens."""

    def __init__(self, dim: int, query_dim: int | None = None) -> None:
        super().__init__()
        query_dim = query_dim or dim
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(query_dim, dim, bias=False)
        self.scale = dim**-0.5

    def forward(self, tokens: Tensor, mask: Tensor, query: Tensor) -> Tensor:
        logits = torch.einsum("bnc,bc->bn", self.key(tokens), self.query(query)) * self.scale
        weights = masked_softmax(logits, mask, dim=-1)
        return torch.einsum("bn,bnc->bc", weights, self.value(tokens))

