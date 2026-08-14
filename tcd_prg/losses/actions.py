"""PUSH module losses with action-specific validity masks."""

from torch import Tensor, nn

from .masked import multi_positive_listwise_loss, safe_bce_with_logits, safe_smooth_l1


class PushLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        direction_bin = multi_positive_listwise_loss(
            output["direction_logits"],
            labels["direction_positive"],
            labels["direction_evaluated"],
        )
        direction_residual = safe_smooth_l1(
            output["direction_residual"],
            labels["direction_residual_target"],
            labels["direction_residual_valid"],
        )
        object_positive = labels["object_positive"].bool() & labels["object_valid_mask"].bool()
        object_negative = labels["object_valid_mask"].bool() & ~object_positive
        object_effective = object_positive.any(-1) & object_negative.any(-1)
        known_per_state = labels["object_valid_mask"].sum(-1).float()
        direction_positive = (
            labels["direction_positive"].bool() & labels["direction_evaluated"].bool()
        )
        direction_negative = labels["direction_evaluated"].bool() & ~direction_positive
        direction_effective = direction_positive.any(-1) & direction_negative.any(-1)
        contact_positive = labels["contact_valid"].bool() & (labels["contact_target"] > 0)
        contact_negative = labels["contact_valid"].bool() & ~contact_positive
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
            "push_object_effective_rows": object_effective.sum().float(),
            "push_known_objects_per_state": known_per_state.mean(),
            "push_multiobject_states": (known_per_state >= 2).sum().float(),
            "push_positive_objects": object_positive.sum().float(),
            "push_negative_objects": object_negative.sum().float(),
            "push_contact_positive_points": contact_positive.sum().float(),
            "push_contact_negative_points": contact_negative.sum().float(),
            "push_contact_valid_points": labels["contact_valid"].sum().float(),
            "push_direction_effective_rows": direction_effective.sum().float(),
            "push_direction_residual_targets": labels["direction_residual_valid"].sum().float(),
            "push_potential_valid_candidates": labels["utility_valid"].sum().float(),
            "push_potential": safe_smooth_l1(
                output["utility_delta"], labels["utility_delta"], labels["utility_valid"]
            ),
        }
