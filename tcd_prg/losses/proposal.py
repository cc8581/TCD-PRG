"""Hungarian set loss for complete 6D grasp candidates."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from tcd_prg.geometry.se3 import (
    parallel_jaw_rotation_chordal_loss,
    parallel_jaw_rotation_distance,
)


class CompleteGraspSetLoss(nn.Module):
    """One module objective with internal pose, width and quality terms."""

    def __init__(
        self, translation_weight: float = 1.0, rotation_weight: float = 1.0,
        width_weight: float = 0.5, quality_weight: float = 1.0,
        object_weight: float = 1.0,
        negative_translation_m: float = 0.01,
        negative_rotation_deg: float = 12.0,
        negative_width_m: float = 0.005,
    ) -> None:
        super().__init__()
        self.weights = (
            translation_weight, rotation_weight, width_weight, quality_weight, object_weight
        )
        self.negative_thresholds = (
            float(negative_translation_m), math.radians(float(negative_rotation_deg)),
            float(negative_width_m),
        )

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        prediction_t = output["translation_world"]
        prediction_r = output["rotation_matrix"]
        prediction_w = output["width_m"]
        quality_logit = output["quality_logit"]
        sample_valid = labels["sample_valid"].bool()
        matched_prediction: list[Tensor] = []
        matched_target: list[Tensor] = []
        quality_target = torch.zeros_like(quality_logit)
        unmatched_quality_valid = labels.get(
            "unmatched_quality_valid", torch.zeros_like(sample_valid)
        ).bool()
        quality_valid = unmatched_quality_valid[:, None].expand_as(quality_logit).clone()
        object_logits = output.get("object_logits")
        object_target = labels.get("object_index")
        matched_query = torch.zeros_like(quality_logit, dtype=torch.bool)
        for row in range(prediction_t.shape[0]):
            targets = torch.nonzero(labels["target_valid"][row], as_tuple=False).flatten()
            if not bool(sample_valid[row]) or not len(targets):
                continue
            translation_cost = torch.cdist(
                prediction_t[row], labels["translation_world"][row, targets], p=1
            ) / 0.02
            pred_rotation = prediction_r[row, :, None].expand(-1, len(targets), -1, -1)
            target_rotation = labels["rotation_matrix"][row, targets][None].expand(
                prediction_r.shape[1], -1, -1, -1
            )
            rotation_cost = parallel_jaw_rotation_distance(pred_rotation, target_rotation)
            width_cost = torch.abs(
                prediction_w[row, :, None] - labels["width_m"][row, targets][None]
            ) / 0.02
            confidence_cost = -F.logsigmoid(quality_logit[row])[:, None]
            cost = translation_cost + rotation_cost + 0.5 * width_cost + confidence_cost
            if object_logits is not None and object_target is not None:
                object_cost = -F.log_softmax(object_logits[row], -1)[:, object_target[row, targets]]
                cost = cost + object_cost
            pred_np, target_np = linear_sum_assignment(
                np.asarray(cost.detach().float().cpu())
            )
            pred_index = torch.as_tensor(pred_np, dtype=torch.long, device=prediction_t.device)
            target_index = targets[torch.as_tensor(target_np, dtype=torch.long, device=targets.device)]
            matched_prediction.append(torch.stack((
                torch.full_like(pred_index, row), pred_index,
            ), -1))
            matched_target.append(torch.stack((
                torch.full_like(target_index, row), target_index,
            ), -1))
            quality_target[row, pred_index] = labels["quality_target"][row, target_index]
            quality_valid[row, pred_index] = labels["quality_valid"][row, target_index]
            matched_query[row, pred_index] = True
        negative_valid = labels.get("negative_valid")
        if negative_valid is not None:
            translation_threshold, rotation_threshold, width_threshold = self.negative_thresholds
            negative_object = labels.get("negative_object_index")
            for row in range(prediction_t.shape[0]):
                if not bool(sample_valid[row]):
                    continue
                queries = torch.nonzero(~matched_query[row], as_tuple=False).flatten()
                negatives = torch.nonzero(negative_valid[row], as_tuple=False).flatten()
                if not len(queries) or not len(negatives):
                    continue
                translation_close = torch.cdist(
                    prediction_t[row, queries],
                    labels["negative_translation_world"][row, negatives],
                ) <= translation_threshold
                pred_rotation = prediction_r[row, queries, None].expand(
                    -1, len(negatives), -1, -1
                )
                target_rotation = labels["negative_rotation_matrix"][row, negatives][None].expand(
                    len(queries), -1, -1, -1
                )
                rotation_close = parallel_jaw_rotation_distance(
                    pred_rotation, target_rotation
                ) <= rotation_threshold
                width_close = (
                    prediction_w[row, queries, None]
                    - labels["negative_width_m"][row, negatives][None]
                ).abs() <= width_threshold
                associated = translation_close & rotation_close & width_close
                if object_logits is not None and negative_object is not None:
                    predicted_object = object_logits[row, queries].argmax(-1)
                    associated &= (
                        predicted_object[:, None] == negative_object[row, negatives][None]
                    )
                quality_valid[row, queries[associated.any(-1)]] = True
        if matched_prediction:
            prediction_index = torch.cat(matched_prediction)
            target_index = torch.cat(matched_target)
            pr, pq = prediction_index.unbind(-1)
            tr, tq = target_index.unbind(-1)
            translation = F.smooth_l1_loss(
                prediction_t[pr, pq], labels["translation_world"][tr, tq], beta=0.01
            )
            rotation = parallel_jaw_rotation_chordal_loss(
                prediction_r[pr, pq], labels["rotation_matrix"][tr, tq]
            ).mean()
            width = F.smooth_l1_loss(
                prediction_w[pr, pq], labels["width_m"][tr, tq], beta=0.005
            )
            if object_logits is not None and object_target is not None:
                object_assignment = F.cross_entropy(
                    object_logits[pr, pq], object_target[tr, tq]
                )
            else:
                object_assignment = translation.new_zeros(())
        else:
            zero = prediction_t.sum() * 0.0
            translation = rotation = width = object_assignment = zero
        safe_quality_target = torch.where(quality_valid, quality_target, torch.zeros_like(quality_target))
        quality_raw = F.binary_cross_entropy_with_logits(
            quality_logit, safe_quality_target, reduction="none"
        )
        quality = (quality_raw * quality_valid).sum() / quality_valid.sum().clamp_min(1)
        wt, wr, ww, wq, wo = self.weights
        total = (
            wt * translation + wr * rotation + ww * width + wq * quality
            + wo * object_assignment
        )
        return {
            "loss": total,
            "grasp_translation": translation,
            "grasp_rotation": rotation,
            "grasp_width": width,
            "grasp_quality": quality,
            "grasp_object": object_assignment,
        }


GraspProposalLoss = CompleteGraspSetLoss
