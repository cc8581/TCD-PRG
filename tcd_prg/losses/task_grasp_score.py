"""Proposal-level supervision for the task-region residual grasp scorer."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _rotation_error_deg(first: Tensor, second: Tensor) -> Tensor:
    # first [K,3,3], second [M,3,3] -> [K,M]
    relative = torch.einsum("kij,mjl->kmil", first.transpose(-1, -2), second)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


class TaskGraspScoringLoss(nn.Module):
    """Match frozen GraspNet proposals to task-grasp labels, then learn ranking.

    Unknown unmatched proposals remain ignored unless the producer explicitly marks
    the label set complete. Physically plausible but task-incompatible labeled
    grasps therefore become useful hard negatives without inventing negatives.
    """

    def __init__(
        self,
        translation_m: float = 0.02,
        rotation_deg: float = 20.0,
        width_m: float = 0.01,
        bce_weight: float = 1.0,
        ranking_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.translation_m = float(translation_m)
        self.rotation_deg = float(rotation_deg)
        self.width_m = float(width_m)
        self.bce_weight = float(bce_weight)
        self.ranking_weight = float(ranking_weight)

    def _match_set(
        self,
        translation: Tensor,
        rotation: Tensor,
        width: Tensor,
        valid: Tensor,
        target_translation: Tensor,
        target_rotation: Tensor,
        target_width: Tensor,
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
        w = (width[candidates, None] - target_width[targets][None]).abs()
        compatible = (
            (t <= self.translation_m)
            & (r <= self.rotation_deg)
            & (w <= self.width_m)
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

        for row in range(logits.shape[0]):
            positive[row] = self._match_set(
                prediction["translation_world"][row],
                prediction["rotation_matrix"][row],
                prediction["width_m"][row],
                valid[row],
                labels["translation_world"][row],
                labels["rotation_matrix"][row],
                labels["width_m"][row],
                labels["target_valid"][row],
            )
            if "negative_valid" in labels:
                negative[row] = self._match_set(
                    prediction["translation_world"][row],
                    prediction["rotation_matrix"][row],
                    prediction["width_m"][row],
                    valid[row] & ~positive[row],
                    labels["negative_translation_world"][row],
                    labels["negative_rotation_matrix"][row],
                    labels["negative_width_m"][row],
                    labels["negative_valid"][row],
                )
            if bool(labels.get("label_set_complete", torch.zeros((), device=logits.device))[row]):
                negative[row] |= valid[row] & ~positive[row]

        known = valid & (positive | negative)
        target = positive.to(logits.dtype)
        if known.any():
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[known], target[known]
            )
        else:
            bce = logits.sum() * 0.0

        row_effective = positive.any(-1) & negative.any(-1)
        ranking_terms = []
        for row in torch.nonzero(row_effective, as_tuple=False).flatten().tolist():
            pos = torch.logsumexp(logits[row][positive[row]], dim=0)
            neg = torch.logsumexp(logits[row][negative[row]], dim=0)
            ranking_terms.append(torch.nn.functional.softplus(neg - pos))
        ranking = (
            torch.stack(ranking_terms).mean()
            if ranking_terms
            else logits.sum() * 0.0
        )
        loss = self.bce_weight * bce + self.ranking_weight * ranking

        # Proposal-recall diagnostics are measured in frozen GraspNet score order.
        base_logit = prediction.get("graspnet_quality_logit", prediction["quality_logit"])
        recalls: dict[str, Tensor] = {}
        for amount in (16, 32, 64):
            hit = logits.new_zeros((logits.shape[0],))
            applicable = labels["target_valid"].any(-1)
            for row in range(logits.shape[0]):
                candidates = torch.nonzero(valid[row], as_tuple=False).flatten()
                if not len(candidates) or not bool(applicable[row]):
                    continue
                order = candidates[
                    base_logit[row, candidates].argsort(descending=True, stable=True)
                ][:amount]
                hit[row] = positive[row, order].any().to(hit.dtype)
            denom = applicable.float().sum().clamp_min(1.0)
            recalls[f"task_proposal_recall_at_{amount}"] = (
                (hit * applicable.float()).sum() / denom
            )

        top1_positive = logits.new_zeros((logits.shape[0],))
        has_candidate = valid.any(-1)
        if has_candidate.any():
            masked = logits.masked_fill(~valid, -30.0)
            top1 = masked.argmax(-1)
            rows = torch.arange(logits.shape[0], device=logits.device)
            top1_positive = positive[rows, top1].float()
        supervised = known.any(-1)
        top1_metric = (
            (top1_positive * supervised.float()).sum()
            / supervised.float().sum().clamp_min(1.0)
        )

        return {
            "loss": loss,
            "task_grasp_score_bce": bce.detach(),
            "task_grasp_score_ranking": ranking.detach(),
            "task_grasp_effective_rows": row_effective.float().sum().detach(),
            "task_grasp_supervised_rows": supervised.float().sum().detach(),
            "task_grasp_positive_proposals": positive.float().sum().detach(),
            "task_grasp_negative_proposals": negative.float().sum().detach(),
            "task_grasp_known_proposals": known.float().sum().detach(),
            "task_grasp_top1_positive": top1_metric.detach(),
            **{key: value.detach() for key, value in recalls.items()},
        }
