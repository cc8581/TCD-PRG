"""PUSH module losses with action-specific validity masks."""

from torch import Tensor, nn

from .masked import (
    multi_positive_listwise_loss,
    safe_bce_with_logits,
    safe_cross_entropy,
    safe_smooth_l1,
)


class PushLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        direction_bin = safe_cross_entropy(
            output["direction_logits"], labels["direction_bin"], labels["direction_valid"]
        )
        direction_residual = safe_smooth_l1(
            output["direction_residual"], labels["direction_residual"], labels["direction_valid"]
        )
        return {
            "push_object": multi_positive_listwise_loss(
                output["object_logits"], labels["object_positive"], labels["object_valid_mask"]
            ),
            "push_contact": safe_bce_with_logits(
                output["contact_logits"], labels["contact_target"].float(), labels["contact_valid"]
            ),
            "push_direction": direction_bin + direction_residual,
            "push_direction_bin_diagnostic": direction_bin,
            "push_direction_residual_diagnostic": direction_residual,
            "push_potential": safe_smooth_l1(
                output["utility_delta"], labels["utility_delta"], labels["utility_valid"]
            ),
        }
