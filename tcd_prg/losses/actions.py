"""PUSH and PICK_REMOVE losses with action-specific validity masks."""

from torch import Tensor, nn

from .masked import (
    multi_positive_listwise_loss,
    safe_bce_with_logits,
    safe_cross_entropy,
    safe_smooth_l1,
)


class PushLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        result = {
            "push_object": multi_positive_listwise_loss(
                output["object_logits"], labels["object_positive"], labels["object_valid_mask"]
            ),
            "push_contact": safe_bce_with_logits(
                output["contact_logits"], labels["contact_target"].float(), labels["contact_valid"]
            ),
            "push_direction_bin": safe_cross_entropy(
                output["direction_logits"], labels["direction_bin"], labels["direction_valid"]
            ),
            "push_direction_residual": safe_smooth_l1(
                output["direction_residual"], labels["direction_residual"], labels["direction_valid"]
            ),
        }
        if labels.get("use_potential", True):
            result["push_potential"] = safe_smooth_l1(
                output["potential_delta"], labels["potential_delta"], labels["potential_after_valid"]
            )
        if labels.get("use_risk", True):
            result["push_risk"] = safe_bce_with_logits(
                output["risk_logits"], labels["risk_target"].float(), labels["risk_valid"]
            )
        return result


class PickRemoveLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        result = {
            "remove_object": multi_positive_listwise_loss(
                output["object_logits"], labels["object_positive"], labels["object_valid_mask"]
            )
        }
        if "candidate_logits" in output:
            result["remove_candidate"] = multi_positive_listwise_loss(
                output["candidate_logits"], labels["candidate_positive"], labels["candidate_valid"]
            )
        return result
