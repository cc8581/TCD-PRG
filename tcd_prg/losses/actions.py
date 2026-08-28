"""PUSH module losses with action-specific validity masks."""

from torch import Tensor, nn

from .masked import multi_positive_listwise_loss, safe_bce_with_logits, safe_smooth_l1


class PushLoss(nn.Module):
    def __init__(self, object_topk: int = 4) -> None:
        super().__init__()
        if object_topk <= 0:
            raise ValueError("PushLoss object_topk must be positive")
        self.object_topk = int(object_topk)

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        direction_bce = safe_bce_with_logits(
            output["direction_logits"],
            labels["direction_positive"].float(),
            labels["direction_evaluated"],
        )
        direction_rank = multi_positive_listwise_loss(
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
        valid_object_rows = object_positive.any(-1)

        def object_hits(k: int) -> Tensor:
            count = min(k, output["object_logits"].shape[-1])
            selected = (
                output["object_logits"]
                .masked_fill(~labels["object_deployment_valid"].bool(), float("-inf"))
                .topk(count, -1)
                .indices
            )
            hit = object_positive.gather(1, selected).any(-1)
            return (hit & valid_object_rows).sum().float()

        object_bce = safe_bce_with_logits(
            output["object_logits"], labels["object_positive"].float(), labels["object_valid_mask"]
        )
        object_rank = multi_positive_listwise_loss(
            output["object_logits"], labels["object_positive"], labels["object_valid_mask"]
        )
        return {
            "push_object": object_bce + object_rank,
            "push_object_bce_diagnostic": object_bce,
            "push_object_rank_diagnostic": object_rank,
            "push_object_positive_rows_count": valid_object_rows.sum().float(),
            "push_object_positive_hits_at_1_count": object_hits(1),
            "push_object_positive_hits_at_4_count": object_hits(4),
            "push_object_positive_hits_at_deployment_k_count": object_hits(self.object_topk),
            "push_positive_actions_total_count": labels["positive_evaluated_push"].sum().float(),
            "push_positive_actions_direction_covered_count": labels["positive_direction_covered"]
            .sum()
            .float(),
            "push_positive_actions_utility_covered_count": labels["positive_utility_covered"]
            .sum()
            .float(),
            "push_positive_actions_utility_eligible_count": labels["positive_utility_eligible"]
            .sum()
            .float(),
            "push_object_bce_active_rows_count": labels["object_valid_mask"].any(-1).sum().float(),
            "push_object_rank_active_rows_count": object_effective.sum().float(),
            "push_contact": safe_bce_with_logits(
                output["contact_logits"], labels["contact_target"].float(), labels["contact_valid"]
            ),
            "push_direction": direction_bce + direction_rank + direction_residual,
            "push_direction_bce_diagnostic": direction_bce,
            "push_direction_rank_diagnostic": direction_rank,
            "push_direction_bce_active_rows_count": labels["direction_evaluated"]
            .any(-1)
            .sum()
            .float(),
            "push_direction_rank_active_rows_count": direction_effective.sum().float(),
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
