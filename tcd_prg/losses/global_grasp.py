"""Set-matched multimodal global-grasp supervision."""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn

from .masked import masked_mean, safe_bce_with_logits, safe_cross_entropy, safe_smooth_l1


class GlobalGraspLoss(nn.Module):
    """Match unordered target modes to per-point predictions before regression."""

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        contact = safe_bce_with_logits(
            output["contact_logits"], labels["contact_target"], labels["contact_valid"]
        )
        predicted_modes = output["approach_direction"].shape[2]
        assignment = torch.full_like(labels["mode_valid"], -1, dtype=torch.long)
        target_counts = labels["mode_valid"].sum(-1)
        for count in range(1, predicted_modes + 1):
            points = torch.nonzero(target_counts == count, as_tuple=False)
            if not len(points):
                continue
            batch_rows, point_rows = points[:, 0], points[:, 1]
            predicted = output["approach_direction"][batch_rows, point_rows]
            target = labels["approach_target"][batch_rows, point_rows, :count]
            approach_cost = 1.0 - torch.einsum("smc,stc->smt", predicted, target)
            width_cost = torch.abs(
                output["width_m"][batch_rows, point_rows, :, None]
                - labels["width_target_m"][batch_rows, point_rows, None, :count]
            ) / 0.02
            rotation_probability = torch.log_softmax(
                output["rotation_logits"][batch_rows, point_rows], -1
            )
            rotation_target = labels["rotation_bin"][batch_rows, point_rows, :count]
            rotation_cost = -rotation_probability.gather(
                2, rotation_target[:, None].expand(-1, predicted_modes, -1)
            )
            cost = (approach_cost + width_cost + rotation_cost).detach()
            permutations = torch.tensor(
                list(itertools.permutations(range(predicted_modes), count)),
                dtype=torch.long, device=cost.device,
            )
            permutation_cost = torch.zeros(
                (len(points), len(permutations)), dtype=cost.dtype, device=cost.device
            )
            for target_index in range(count):
                permutation_cost += cost[:, permutations[:, target_index], target_index]
            best = permutations[permutation_cost.argmin(-1)]
            assignment[batch_rows, point_rows, :count] = best

        matched = assignment >= 0
        rows = torch.nonzero(matched, as_tuple=True)
        if rows[0].numel():
            prediction_index = assignment[rows]
            pred_approach = output["approach_direction"][rows[0], rows[1], prediction_index]
            pred_rotation = output["rotation_logits"][rows[0], rows[1], prediction_index]
            pred_width = output["width_m"][rows[0], rows[1], prediction_index]
            pred_center_offset = output["center_offset_m"][rows[0], rows[1], prediction_index]
            target_approach = labels["approach_target"][rows]
            target_rotation = labels["rotation_bin"][rows]
            target_width = labels["width_target_m"][rows]
            target_center_offset = labels["center_offset_target_m"][rows]
            geometry = labels["geometry_valid"][rows]
            approach = masked_mean(1.0 - (pred_approach * target_approach).sum(-1), geometry)
            rotation = safe_cross_entropy(pred_rotation, target_rotation, geometry)
            width = safe_smooth_l1(pred_width, target_width, geometry & labels["width_valid"][rows])
            center_offset = safe_smooth_l1(
                pred_center_offset, target_center_offset,
                (geometry & labels["center_offset_valid"][rows]).unsqueeze(-1).expand_as(pred_center_offset),
            )
        else:
            zero = output["contact_logits"].sum() * 0.0
            approach = rotation = width = center_offset = zero

        # DETR-style no-grasp supervision: every unmatched prediction mode at
        # a supervised point is an explicit confidence negative. Scene
        # confidence uses this negative only when every matched target at that
        # point has a known scene certification label.
        predicted_matched = torch.zeros_like(output["scene_confidence_logit"], dtype=torch.bool)
        intrinsic_target = torch.zeros_like(output["intrinsic_confidence_logit"])
        scene_target = torch.zeros_like(output["scene_confidence_logit"])
        intrinsic_valid = torch.zeros_like(predicted_matched)
        scene_valid = torch.zeros_like(predicted_matched)
        supervised_point = labels["mode_valid"].any(-1)
        scene_complete = supervised_point & (labels["scene_valid"] | ~labels["mode_valid"]).all(-1)
        if rows[0].numel():
            prediction_index = assignment[rows]
            predicted_matched[rows[0], rows[1], prediction_index] = True
            intrinsic_target[rows[0], rows[1], prediction_index] = labels["intrinsic_target"][rows]
            intrinsic_valid[rows[0], rows[1], prediction_index] = labels["intrinsic_valid"][rows]
            scene_target[rows[0], rows[1], prediction_index] = labels["scene_target"][rows]
            scene_valid[rows[0], rows[1], prediction_index] = labels["scene_valid"][rows]
        unmatched = supervised_point.unsqueeze(-1) & ~predicted_matched
        intrinsic_valid |= unmatched
        scene_valid |= unmatched & scene_complete.unsqueeze(-1)
        intrinsic_confidence = safe_bce_with_logits(
            output["intrinsic_confidence_logit"], intrinsic_target, intrinsic_valid
        )
        scene_confidence = safe_bce_with_logits(
            output["scene_confidence_logit"], scene_target, scene_valid
        )
        return {
            "global_contact": contact,
            "global_approach": approach,
            "global_rotation": rotation,
            "global_width": width,
            "global_center_offset": center_offset,
            "global_scene_confidence": scene_confidence,
            "global_intrinsic_confidence": intrinsic_confidence,
        }
