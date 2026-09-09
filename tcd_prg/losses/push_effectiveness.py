"""Binary Stage-C PUSH-improvement objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PushImprovementLoss(nn.Module):
    """BCE-only supervision for ``P(improve | state, action)``."""

    def __init__(self, *, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("PUSH improvement pos_weight must be positive")
        self.pos_weight = pos_weight

    def forward(self, prediction: dict[str, Tensor], *, improvement_target: Tensor,
                improvement_valid: Tensor) -> dict[str, Tensor]:
        logit = prediction["improvement_logit"]
        if logit.shape != improvement_target.shape or improvement_valid.shape != logit.shape:
            raise ValueError("PUSH improvement logit/target/mask shapes must align")
        valid = improvement_valid.bool()
        if not bool(valid.any()):
            loss = logit.sum() * 0.0
        else:
            target = improvement_target[valid]
            if not bool(torch.isfinite(target).all()) or bool(((target != 0) & (target != 1)).any()):
                raise ValueError("valid PUSH improvement targets must be binary")
            weight = None if self.pos_weight is None else logit.new_tensor(self.pos_weight)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit[valid], target, pos_weight=weight
            )
        return {"push_improvement": loss, "push_improvement_bce": loss.detach(),
                "push_improvement_supervised_count": valid.sum().detach().float()}
