"""NaN-safe masked objectives. Invalid targets never enter arithmetic."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    safe = torch.where(mask, value, torch.zeros_like(value))
    return safe.sum() / mask.to(value.dtype).sum().clamp_min(1.0)


def safe_bce_with_logits(logits: Tensor, target: Tensor, valid: Tensor, **kwargs: float) -> Tensor:
    valid = valid.bool() & torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none", **kwargs)
    return masked_mean(loss, valid)


def safe_cross_entropy(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    valid = valid.bool() & (target >= 0)
    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
    loss = F.cross_entropy(logits.movedim(-1, 1), safe_target, reduction="none")
    return masked_mean(loss, valid)


def safe_smooth_l1(prediction: Tensor, target: Tensor, valid: Tensor, beta: float = 1.0) -> Tensor:
    while valid.ndim < target.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.bool() & torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = F.smooth_l1_loss(prediction, safe_target, reduction="none", beta=beta)
    return masked_mean(loss, valid)


def multi_positive_listwise_loss(logits: Tensor, positive: Tensor, valid: Tensor) -> Tensor:
    """Negative log probability mass assigned to the complete positive set."""

    valid = valid.bool()
    positive = positive.bool() & valid
    masked_logits = logits.masked_fill(~valid, -torch.inf)
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    log_all = torch.logsumexp(masked_logits, dim=-1)
    log_positive = torch.logsumexp(positive_logits, dim=-1)
    # A positive-only row has numerator == denominator and exactly zero
    # gradient.  Count/train only rows containing a known positive and at
    # least one known negative competitor.
    negative = valid & ~positive
    row_valid = positive.any(-1) & negative.any(-1)
    loss = -(log_positive - log_all)
    return masked_mean(loss, row_valid)
