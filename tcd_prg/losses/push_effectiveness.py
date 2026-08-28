"""Binary objective for evaluated complete PUSH actions."""

from __future__ import annotations

from torch import Tensor, nn

from tcd_prg.constants import CandidateStatus

from .masked import safe_bce_with_logits


class PushEffectivenessLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        effective_logit: Tensor,
        evaluation_status: Tensor,
        action_improves_state: Tensor,
    ) -> dict[str, Tensor]:
        valid = evaluation_status != int(CandidateStatus.UNKNOWN_UNTESTED)
        target = action_improves_state.to(effective_logit.dtype)
        kwargs = {}
        if self.pos_weight is not None:
            kwargs["pos_weight"] = effective_logit.new_tensor(self.pos_weight)
        loss = safe_bce_with_logits(effective_logit, target, valid, **kwargs)
        return {
            "push_effectiveness": loss,
            "push_effectiveness_evaluated_count": valid.sum().float(),
            "push_effectiveness_positive_count": (valid & action_improves_state.bool())
            .sum()
            .float(),
        }
