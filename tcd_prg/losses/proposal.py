"""Task-conditioned grasp proposal supervision."""

import torch
from torch import Tensor, nn

from .masked import masked_mean, safe_bce_with_logits, safe_cross_entropy, safe_smooth_l1


class GraspProposalLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        valid = labels["proposal_valid"].bool()
        contact = safe_bce_with_logits(output["contact_logits"], labels["contact_target"].float(), valid)
        positive = labels["mode_valid"].bool()
        approach_target = torch.nn.functional.normalize(
            torch.nan_to_num(labels["approach_target"].float()), dim=-1, eps=1e-6
        )
        cosine = 1 - (output["approach_direction"] * approach_target).sum(-1)
        approach = masked_mean(cosine, positive & torch.isfinite(labels["approach_target"]).all(-1))
        rotation = safe_cross_entropy(output["rotation_logits"], labels["rotation_bin"], positive)
        width = safe_smooth_l1(output["width_m"], labels["width_target_m"], positive & labels["width_valid"])
        center_offset = safe_smooth_l1(
            output["center_offset_m"], labels["center_offset_target_m"],
            labels["center_offset_valid"].unsqueeze(-1).expand_as(output["center_offset_m"]),
        )
        confidence = safe_bce_with_logits(
            output["proposal_confidence_logit"], labels["confidence_target"].float(), valid
        )
        compatibility = safe_bce_with_logits(
            output["task_compatibility_logit"],
            labels["compatibility_target"].float(),
            labels.get("compatibility_valid", valid),
        )
        return {
            "proposal_contact": contact,
            "proposal_approach": approach,
            "proposal_rotation": rotation,
            "proposal_width": width,
            "proposal_center_offset": center_offset,
            "proposal_confidence": confidence,
            "proposal_task_compatibility": compatibility,
        }


class StateGraspabilityLoss(nn.Module):
    """Binary adaptive gate plus calibrated reliable-candidate count."""

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        valid = torch.ones_like(labels["graspable_target"], dtype=torch.bool)
        return {
            "proposal_state_graspable": safe_bce_with_logits(
                output["graspable_logit"], labels["graspable_target"].float(), valid
            ),
            "proposal_verified_count": safe_smooth_l1(
                output["verified_count_log_prediction"],
                torch.log1p(labels["verified_count_target"].float()),
                valid,
            ),
        }
