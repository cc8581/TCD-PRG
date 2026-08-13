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
    """One equal-budget grasp objective with explicit positive/negative matching."""

    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        width_weight: float = 0.5,
        quality_weight: float = 1.0,
        object_weight: float = 1.0,
        negative_translation_m: float = 0.01,
        negative_rotation_deg: float = 12.0,
        negative_width_m: float = 0.005,
        translation_scale_m: float = 0.02,
        width_scale_m: float = 0.02,
    ) -> None:
        super().__init__()
        self.weights = (
            translation_weight,
            rotation_weight,
            width_weight,
            quality_weight,
            object_weight,
        )
        self.negative_thresholds = (
            float(negative_translation_m),
            math.radians(float(negative_rotation_deg)),
            float(negative_width_m),
        )
        self.translation_scale_m = float(translation_scale_m)
        self.width_scale_m = float(width_scale_m)
        if min(self.translation_scale_m, self.width_scale_m) <= 0:
            raise ValueError("Grasp loss scales must be positive")

    @staticmethod
    def _hungarian(cost: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
        return CompleteGraspSetLoss._hungarian_many([cost], device)[0]

    @staticmethod
    def _hungarian_many(
        costs: list[Tensor], device: torch.device
    ) -> list[tuple[Tensor, Tensor]]:
        """Exact SciPy Hungarian with one device-to-host sync per stage.

        Every matrix is still passed independently to ``linear_sum_assignment``
        with the same float32 values and original shape.  Only the transfer of
        those matrices is coalesced.
        """

        if not costs:
            return []
        shapes = [tuple(int(value) for value in cost.shape) for cost in costs]
        sizes = [int(cost.numel()) for cost in costs]
        flat = torch.cat(
            [cost.detach().float().reshape(-1) for cost in costs], dim=0
        ).cpu().numpy()
        assignments: list[tuple[Tensor, Tensor]] = []
        offset = 0
        for shape, size in zip(shapes, sizes, strict=True):
            matrix = np.asarray(flat[offset : offset + size]).reshape(shape)
            row, column = linear_sum_assignment(matrix)
            assignments.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=device),
                    torch.as_tensor(column, dtype=torch.long, device=device),
                )
            )
            offset += size
        return assignments

    def _geometry_cost(
        self,
        prediction_t: Tensor,
        prediction_r: Tensor,
        prediction_w: Tensor,
        target_t: Tensor,
        target_r: Tensor,
        target_w: Tensor,
    ) -> Tensor:
        translation = torch.cdist(prediction_t, target_t, p=1) / self.translation_scale_m
        pred_rotation = prediction_r[:, None].expand(-1, len(target_r), -1, -1)
        target_rotation = target_r[None].expand(len(prediction_r), -1, -1, -1)
        rotation = parallel_jaw_rotation_distance(pred_rotation, target_rotation)
        width = torch.abs(prediction_w[:, None] - target_w[None]) / self.width_scale_m
        return translation + rotation + 0.5 * width

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
        negative_matched_query = torch.zeros_like(matched_query)

        positive_jobs: list[tuple[int, Tensor, Tensor]] = []
        for row in range(prediction_t.shape[0]):
            targets = torch.nonzero(labels["target_valid"][row], as_tuple=False).flatten()
            if not bool(sample_valid[row]) or not len(targets):
                continue
            cost = self._geometry_cost(
                prediction_t[row],
                prediction_r[row],
                prediction_w[row],
                labels["translation_world"][row, targets],
                labels["rotation_matrix"][row, targets],
                labels["width_m"][row, targets],
            )
            cost = cost - F.logsigmoid(quality_logit[row])[:, None]
            if object_logits is not None and object_target is not None:
                cost = cost - F.log_softmax(object_logits[row], -1)[:, object_target[row, targets]]
            positive_jobs.append((row, targets, cost.detach().float()))

        positive_assignments = self._hungarian_many(
            [job[2] for job in positive_jobs], prediction_t.device
        )
        for (row, targets, _), (pred_index, target_local) in zip(
            positive_jobs, positive_assignments, strict=True
        ):
            target_index = targets[target_local]
            matched_prediction.append(
                torch.stack((torch.full_like(pred_index, row), pred_index), -1)
            )
            matched_target.append(
                torch.stack((torch.full_like(target_index, row), target_index), -1)
            )
            quality_target[row, pred_index] = labels["quality_target"][row, target_index].to(
                quality_target.dtype
            )
            quality_valid[row, pred_index] = labels["quality_valid"][row, target_index]
            matched_query[row, pred_index] = True

        negative_valid = labels.get("negative_valid")
        if negative_valid is not None:
            negative_object = labels.get("negative_object_index")
            translation_threshold, rotation_threshold, width_threshold = self.negative_thresholds
            negative_jobs: list[tuple[int, Tensor, Tensor, Tensor]] = []
            for row in range(prediction_t.shape[0]):
                if not bool(sample_valid[row]):
                    continue
                queries = torch.nonzero(~matched_query[row], as_tuple=False).flatten()
                negatives = torch.nonzero(negative_valid[row], as_tuple=False).flatten()
                if not len(queries) or not len(negatives):
                    continue
                negative_cost = self._geometry_cost(
                    prediction_t[row, queries],
                    prediction_r[row, queries],
                    prediction_w[row, queries],
                    labels["negative_translation_world"][row, negatives],
                    labels["negative_rotation_matrix"][row, negatives],
                    labels["negative_width_m"][row, negatives],
                )
                if object_logits is not None and negative_object is not None:
                    negative_cost = (
                        negative_cost
                        - F.log_softmax(object_logits[row, queries], -1)[
                            :, negative_object[row, negatives]
                        ]
                    )
                negative_jobs.append(
                    (row, queries, negatives, negative_cost.detach().float())
                )

            negative_assignments = self._hungarian_many(
                [job[3] for job in negative_jobs], prediction_t.device
            )
            for (row, queries, negatives, _), (query_local, negative_local) in zip(
                negative_jobs, negative_assignments, strict=True
            ):
                candidate_queries = queries[query_local]
                candidate_negatives = negatives[negative_local]
                translation_ok = (
                    torch.linalg.vector_norm(
                        prediction_t[row, candidate_queries]
                        - labels["negative_translation_world"][row, candidate_negatives],
                        dim=-1,
                    )
                    <= translation_threshold
                )
                rotation_ok = (
                    parallel_jaw_rotation_distance(
                        prediction_r[row, candidate_queries],
                        labels["negative_rotation_matrix"][row, candidate_negatives],
                    )
                    <= rotation_threshold
                )
                width_ok = (
                    prediction_w[row, candidate_queries]
                    - labels["negative_width_m"][row, candidate_negatives]
                ).abs() <= width_threshold
                object_ok = torch.ones_like(translation_ok, dtype=torch.bool)
                if object_logits is not None and negative_object is not None:
                    object_ok = (
                        object_logits[row, candidate_queries].argmax(-1)
                        == negative_object[row, candidate_negatives]
                    )
                accepted = translation_ok & rotation_ok & width_ok & object_ok
                accepted_queries = candidate_queries[accepted]
                quality_target[row, accepted_queries] = 0.0
                quality_valid[row, accepted_queries] = True
                # A rejected Hungarian pair remains UNKNOWN and must remain
                # eligible for the all-negative threshold association below.
                matched_query[row, accepted_queries] = True
                negative_matched_query[row, accepted_queries] = True

                remaining = torch.nonzero(~matched_query[row], as_tuple=False).flatten()
                if not len(remaining):
                    continue
                translation_close = (
                    torch.cdist(
                        prediction_t[row, remaining],
                        labels["negative_translation_world"][row, negatives],
                    )
                    <= translation_threshold
                )
                pred_rotation = prediction_r[row, remaining, None].expand(
                    -1, len(negatives), -1, -1
                )
                target_rotation = labels["negative_rotation_matrix"][row, negatives][None].expand(
                    len(remaining), -1, -1, -1
                )
                rotation_close = (
                    parallel_jaw_rotation_distance(pred_rotation, target_rotation)
                    <= rotation_threshold
                )
                width_close = (
                    prediction_w[row, remaining, None]
                    - labels["negative_width_m"][row, negatives][None]
                ).abs() <= width_threshold
                associated = translation_close & rotation_close & width_close
                if object_logits is not None and negative_object is not None:
                    predicted_object = object_logits[row, remaining].argmax(-1)
                    associated &= predicted_object[:, None] == negative_object[row, negatives][None]
                quality_valid[row, remaining[associated.any(-1)]] = True

        if matched_prediction:
            prediction_index = torch.cat(matched_prediction)
            target_index = torch.cat(matched_target)
            pr, pq = prediction_index.unbind(-1)
            tr, tq = target_index.unbind(-1)
            translation_error = (
                prediction_t[pr, pq] - labels["translation_world"][tr, tq]
            ) / self.translation_scale_m
            translation = F.smooth_l1_loss(
                translation_error, torch.zeros_like(translation_error), beta=0.5
            )
            rotation = parallel_jaw_rotation_chordal_loss(
                prediction_r[pr, pq], labels["rotation_matrix"][tr, tq]
            ).mean()
            width_error = (prediction_w[pr, pq] - labels["width_m"][tr, tq]) / self.width_scale_m
            width = F.smooth_l1_loss(width_error, torch.zeros_like(width_error), beta=0.25)
            if object_logits is not None and object_target is not None:
                object_assignment = F.cross_entropy(object_logits[pr, pq], object_target[tr, tq])
                object_active = True
            else:
                object_assignment = translation.new_zeros(())
                object_active = False
            pose_active = True
        else:
            zero = prediction_t.sum() * 0.0
            translation = rotation = width = object_assignment = zero
            pose_active = object_active = False

        safe_quality_target = torch.where(
            quality_valid, quality_target, torch.zeros_like(quality_target)
        )
        quality_raw = F.binary_cross_entropy_with_logits(
            quality_logit, safe_quality_target, reduction="none"
        )
        quality = (quality_raw * quality_valid).sum() / quality_valid.sum().clamp_min(1)
        quality_active = bool(quality_valid.any())
        eligible_queries = sample_valid[:, None].expand_as(quality_valid)
        quality_positive = quality_valid & (quality_target > 0.5)
        quality_negative = quality_valid & ~quality_positive
        ignored_queries = eligible_queries & ~quality_valid
        wt, wr, ww, wq, wo = self.weights
        numerator = (
            wt * translation * float(pose_active)
            + wr * rotation * float(pose_active)
            + ww * width * float(pose_active)
            + wq * quality * float(quality_active)
            + wo * object_assignment * float(object_active)
        )
        denominator = (
            wt * float(pose_active)
            + wr * float(pose_active)
            + ww * float(pose_active)
            + wq * float(quality_active)
            + wo * float(object_active)
        )
        total = numerator / max(denominator, 1e-12)
        return {
            "loss": total,
            "grasp_translation": translation,
            "grasp_rotation": rotation,
            "grasp_width": width,
            "grasp_quality": quality,
            "grasp_object": object_assignment,
            "grasp_matched_positive_queries": (
                matched_query.sum().float() - negative_matched_query.sum().float()
            ),
            "grasp_matched_negative_queries": negative_matched_query.sum().float(),
            "grasp_quality_valid_queries": quality_valid.sum().float(),
            "grasp_quality_positive_queries": quality_positive.sum().float(),
            "grasp_quality_negative_queries": quality_negative.sum().float(),
            "grasp_ignored_unmatched_queries": ignored_queries.sum().float(),
            "grasp_supervised_rows": sample_valid.sum().float(),
        }


GraspProposalLoss = CompleteGraspSetLoss
