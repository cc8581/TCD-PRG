"""Independent outcome-Critic objectives over executed Offline Bank actions."""

from __future__ import annotations

from torch import Tensor, nn

from tcd_prg.constants import CandidateStatus

from .masked import safe_bce_with_logits


class PushOutcomeCriticLoss(nn.Module):
    def forward(
        self,
        effective_logit: Tensor,
        robustness_logit: Tensor,
        status: Tensor,
        robust_success_count: Tensor,
        robust_trial_count: Tensor,
    ) -> dict[str, Tensor]:
        executed = status != int(CandidateStatus.UNKNOWN_UNTESTED)
        effective_target = (status == int(CandidateStatus.POSITIVE)).to(
            effective_logit.dtype
        )
        robust_valid = executed & (robust_trial_count > 0)
        robust_target = robust_success_count.to(robustness_logit.dtype) / robust_trial_count.clamp_min(1).to(robustness_logit.dtype)
        effective = safe_bce_with_logits(
            effective_logit, effective_target, executed
        )
        robustness = safe_bce_with_logits(
            robustness_logit, robust_target, robust_valid
        )
        return {
            "push_critic_effective": effective,
            "push_critic_robustness": robustness,
            "push_critic": effective + robustness,
            "push_critic_executed_count": executed.sum().float(),
            "push_critic_robust_count": robust_valid.sum().float(),
        }
