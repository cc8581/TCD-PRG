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

    def forward_grouped(self, tokens: Tensor, mask: Tensor, query: Tensor) -> Tensor:
        """Pool several masked groups while projecting point keys/values only once.

        ``tokens`` is ``[B,N,C]``, ``mask`` is ``[B,G,N]`` and ``query`` may be
        either ``[B,Cq]`` (shared by all groups) or ``[B,G,Cq]``.  This is
        numerically equivalent to calling :meth:`forward` once per group, but
        avoids repeating the expensive point-wise key/value projections.
        """

        if mask.ndim != 3 or mask.shape[0] != tokens.shape[0] or mask.shape[2] != tokens.shape[1]:
            raise ValueError("Grouped pool expects tokens [B,N,C] and mask [B,G,N]")
        if query.ndim == 2:
            query = query[:, None].expand(-1, mask.shape[1], -1)
        elif query.ndim != 3 or query.shape[:2] != mask.shape[:2]:
            raise ValueError("Grouped pool query must be [B,Cq] or [B,G,Cq]")
        key = self.key(tokens)
        value = self.value(tokens)
        projected_query = self.query(query)
        logits = torch.einsum("bnc,bgc->bgn", key, projected_query) * self.scale
        weights = masked_softmax(logits, mask, dim=-1)
        return torch.einsum("bgn,bnc->bgc", weights, value)

