"""Set-valued hierarchical policy supervision."""

import torch
from torch import Tensor, nn

from .masked import multi_positive_listwise_loss, safe_smooth_l1


class HierarchicalSetPolicyLoss(nn.Module):
    def forward(
        self,
        output: object,
        candidate_type: Tensor,
        candidate_object: Tensor,
        successful: Tensor,
        evaluated: Tensor,
        remaining_steps_target: Tensor | None = None,
        remaining_steps_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        valid = output.candidate_valid_mask & evaluated
        candidate_loss = multi_positive_listwise_loss(output.candidate_logits, successful, valid)
        type_positive = torch.stack(
            [(successful & (candidate_type == kind)).any(-1) for kind in range(3)], -1
        )
        type_loss = multi_positive_listwise_loss(output.action_type_logits, type_positive, output.type_valid_mask)
        b, _, o = output.object_logits.shape
        object_loss = output.object_logits.sum() * 0.0
        rows = 0
        for kind in range(3):
            positive = torch.zeros((b, o), dtype=torch.bool, device=candidate_type.device)
            for object_index in range(o):
                positive[:, object_index] = (
                    successful & (candidate_type == kind) & (candidate_object == object_index)
                ).any(-1)
            if positive.any():
                object_loss = object_loss + multi_positive_listwise_loss(
                    output.object_logits[:, kind], positive, output.object_valid_mask[:, kind]
                )
                rows += 1
        if rows:
            object_loss = object_loss / rows
        result = {
            "policy_type": type_loss,
            "policy_object": object_loss,
            "policy_candidate": candidate_loss,
        }
        if remaining_steps_target is not None and remaining_steps_valid is not None:
            result["policy_remaining_steps"] = safe_smooth_l1(
                output.remaining_steps_prediction,
                remaining_steps_target,
                remaining_steps_valid,
            )
        return result
