"""Proposal-level supervision for the task-region residual grasp scorer."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance
from tcd_prg.constants import CandidateStatus


def _rotation_error_deg(first: Tensor, second: Tensor) -> Tensor:
    # first [K,3,3], second [M,3,3] -> [K,M]
    return torch.rad2deg(
        parallel_jaw_rotation_distance(first[:, None], second[None])
    )


class TaskGraspScoringLoss(nn.Module):
    """Match frozen GraspNet proposals to task-grasp labels, then learn ranking.

    Identity uses translation plus parallel-jaw-symmetric rotation only. Explicit
    wrong-region labels are the sole task negatives; every unmatched proposal stays
    UNKNOWN because the stored grasp sets are intentionally incomplete.
    """

    def __init__(
        self,
        translation_m: float = 0.02,
        rotation_deg: float = 20.0,
        width_m: float | None = None,
        bce_weight: float = 1.0,
        ranking_weight: float = 1.0,
        listwise_weight: float = 1.0,
        hard_weight: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        # Compatibility-only argument: width is intentionally not proposal identity.
        del width_m
        self.translation_m = float(translation_m)
        self.rotation_deg = float(rotation_deg)
        self.bce_weight = float(bce_weight)
        self.ranking_weight = float(ranking_weight)
        self.listwise_weight = float(listwise_weight)
        self.hard_weight = float(hard_weight)
        self.temperature = float(temperature)

    def _match_set(
        self,
        translation: Tensor,
        rotation: Tensor,
        valid: Tensor,
        target_translation: Tensor,
        target_rotation: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        result = torch.zeros_like(valid)
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        targets = torch.nonzero(target_valid, as_tuple=False).flatten()
        if not len(candidates) or not len(targets):
            return result
        t = torch.cdist(
            translation[candidates].float(), target_translation[targets].float()
        )
        r = _rotation_error_deg(rotation[candidates].float(), target_rotation[targets].float())
        compatible = (
            (t <= self.translation_m)
            & (r <= self.rotation_deg)
        )
        result[candidates] = compatible.any(-1)
        return result

    def forward(
        self,
        prediction: dict[str, Tensor],
        labels: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        logits = prediction["quality_logit"]
        valid = prediction.get("valid", torch.ones_like(logits, dtype=torch.bool)).bool()
        positive = torch.zeros_like(valid)
        negative = torch.zeros_like(valid)
        positive_wrong_region_overlap = torch.zeros_like(valid)

        direct_status = labels.get("proposal_status")
        if direct_status is not None:
            positive = valid & (direct_status == int(CandidateStatus.POSITIVE))
            negative = valid & (direct_status == int(CandidateStatus.NEGATIVE))
        for row in range(logits.shape[0]):
            if direct_status is not None:
                continue
            positive[row] = self._match_set(
                prediction["translation_world"][row],
                prediction["rotation_matrix"][row],
                valid[row],
                labels["translation_world"][row],
                labels["rotation_matrix"][row],
                labels["target_valid"][row],
            )
            if "wrong_region_valid" in labels:
                raw_negative = self._match_set(
                    prediction["translation_world"][row],
                    prediction["rotation_matrix"][row],
                    valid[row],
                    labels["wrong_region_translation_world"][row],
                    labels["wrong_region_rotation_matrix"][row],
                    labels["wrong_region_valid"][row],
                )
                positive_wrong_region_overlap[row] = raw_negative & positive[row]
                # Explicit negative evidence wins conflicts.
                negative[row] = raw_negative
                positive[row] = positive[row] & ~raw_negative

        known = valid & (positive | negative)
        target = positive.to(logits.dtype)
        loss_logits = logits.float()
        if known.any():
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                loss_logits[known], target[known].float()
            )
        else:
            bce = loss_logits.sum() * 0.0

        row_effective = positive.any(-1) & negative.any(-1)
        ranking_terms = []
        listwise_terms = []
        for row in torch.nonzero(row_effective, as_tuple=False).flatten().tolist():
            # Deployment selects the highest-scoring proposal.  Set-level
            # logsumexp is count-sensitive and can be small even when the top
            # known negative outranks every positive.  Optimize the same hard
            # ordering used by candidate selection while leaving UNKNOWN
            # proposals ignored rather than silently treating them as negatives.
            pos = loss_logits[row][positive[row]].max()
            neg = loss_logits[row][negative[row]].max()
            ranking_terms.append(torch.nn.functional.softplus(neg - pos))
            known_logits = loss_logits[row][known[row]] / self.temperature
            known_positive = positive[row][known[row]]
            listwise_terms.append(
                -(torch.logsumexp(known_logits[known_positive], dim=0)
                  - torch.logsumexp(known_logits, dim=0))
            )
        ranking = (
            torch.stack(ranking_terms).mean()
            if ranking_terms
            else loss_logits.sum() * 0.0
        )
        listwise = (
            torch.stack(listwise_terms).mean()
            if listwise_terms else loss_logits.sum() * 0.0
        )
        loss = (
            self.bce_weight * bce
            + self.listwise_weight * listwise
            + self.hard_weight * self.ranking_weight * ranking
        )

        # Proposal-recall diagnostics are measured in frozen GraspNet score order.
        base_logit = prediction.get("graspnet_quality_logit", prediction["quality_logit"])
        recalls: dict[str, Tensor] = {}
        for amount in (16, 32, 64):
            translation_hit = logits.new_zeros((logits.shape[0],))
            translation_rotation_hit = logits.new_zeros((logits.shape[0],))
            applicable = (
                positive.any(-1)
                if direct_status is not None
                else labels["target_valid"].any(-1)
            )
            for row in range(logits.shape[0]):
                candidates = torch.nonzero(valid[row], as_tuple=False).flatten()
                if not len(candidates) or not bool(applicable[row]):
                    continue
                order = candidates[
                    base_logit[row, candidates].argsort(descending=True, stable=True)
                ][:amount]
                if direct_status is not None:
                    translation_hit[row] = positive[row, order].any().to(
                        translation_hit.dtype
                    )
                else:
                    targets = torch.nonzero(
                        labels["target_valid"][row], as_tuple=False
                    ).flatten()
                    distance = torch.cdist(
                        prediction["translation_world"][row, order].float(),
                        labels["translation_world"][row, targets].float(),
                    )
                    translation_hit[row] = (distance <= self.translation_m).any().to(
                        translation_hit.dtype
                    )
                translation_rotation_hit[row] = positive[row, order].any().to(
                    translation_rotation_hit.dtype
                )
            denom = applicable.float().sum().clamp_min(1.0)
            recalls[f"proposal_translation_recall_at_{amount}"] = (
                (translation_hit * applicable.float()).sum() / denom
            )
            recalls[f"proposal_translation_rotation_recall_at_{amount}"] = (
                (translation_rotation_hit * applicable.float()).sum() / denom
            )
            recalls[f"task_proposal_recall_at_{amount}"] = recalls[
                f"proposal_translation_rotation_recall_at_{amount}"
            ]

        top1_positive = logits.new_zeros((logits.shape[0],))
        top1_known_positive = logits.new_zeros((logits.shape[0],))
        top1_unknown = logits.new_zeros((logits.shape[0],))
        top1_known_negative = logits.new_zeros((logits.shape[0],))
        has_candidate = valid.any(-1)
        if has_candidate.any():
            masked = logits.masked_fill(~valid, -30.0)
            top1 = masked.argmax(-1)
            rows = torch.arange(logits.shape[0], device=logits.device)
            top1_positive = positive[rows, top1].float()
        supervised = known.any(-1)
        unknown = valid & ~known
        if supervised.any():
            known_top1 = logits.masked_fill(~known, -30.0).argmax(-1)
            top1_known_positive = positive[rows, known_top1].float()
            top1_unknown = unknown[rows, top1].float()
            top1_known_negative = negative[rows, top1].float()
        top1_metric = (
            (top1_positive * supervised.float()).sum()
            / supervised.float().sum().clamp_min(1.0)
        )
        top1_known_metric = (
            (top1_known_positive * supervised.float()).sum()
            / supervised.float().sum().clamp_min(1.0)
        )
        top1_unknown_fraction = (
            (top1_unknown * supervised.float()).sum()
            / supervised.float().sum().clamp_min(1.0)
        )
        top1_known_negative_fraction = (
            (top1_known_negative * supervised.float()).sum()
            / supervised.float().sum().clamp_min(1.0)
        )

        positive_count = positive.float().sum().detach()
        wrong_region_count = negative.float().sum().detach()
        unknown_count = unknown.float().sum().detach()
        supervised_rows = supervised.float().sum().detach()
        effective_rows = row_effective.float().sum().detach()
        row_count = max(1, logits.shape[0])
        proposal_count = (known.float().sum() + unknown_count).clamp_min(1.0)
        return {
            "loss": loss,
            "task_grasp_score_bce": bce.detach(),
            "task_grasp_score_ranking": ranking.detach(),
            "task_grasp_score_listwise": listwise.detach(),
            "task_grasp_effective_rows": effective_rows,
            "task_grasp_effective_fraction": effective_rows / row_count,
            "task_grasp_supervised_rows": supervised_rows,
            "task_grasp_positive_proposals": positive_count,
            "task_grasp_wrong_region_negative_proposals": wrong_region_count,
            "task_grasp_negative_proposals": wrong_region_count,
            "task_grasp_known_proposals": known.float().sum().detach(),
            "task_grasp_unknown_proposals": unknown_count,
            "task_grasp_unknown_fraction": unknown_count / proposal_count,
            "task_grasp_effective_ranking_rows": effective_rows,
            # Stable protocol names used by training dashboards.
            "task_positive_proposals": positive_count,
            "task_wrong_region_negative_proposals": wrong_region_count,
            "task_unknown_proposals": unknown_count,
            "task_supervised_rows": supervised_rows,
            "task_effective_ranking_rows": effective_rows,
            "positive_wrong_region_overlap": (
                positive_wrong_region_overlap.float().sum().detach()
            ),
            "collision_excluded_from_task_score": labels.get(
                "collision_excluded_from_task_score", logits.new_zeros(logits.shape[0])
            ).float().sum().detach(),
            "approach_excluded_from_task_score": labels.get(
                "approach_excluded_from_task_score", logits.new_zeros(logits.shape[0])
            ).float().sum().detach(),
            "task_grasp_top1_positive": top1_metric.detach(),
            "task_grasp_top1_known_positive": top1_known_metric.detach(),
            "task_grasp_top1_unknown": top1_unknown_fraction.detach(),
            "task_grasp_top1_known_negative": top1_known_negative_fraction.detach(),
            "task_top1_positive": top1_metric.detach(),
            **{key: value.detach() for key, value in recalls.items()},
        }
