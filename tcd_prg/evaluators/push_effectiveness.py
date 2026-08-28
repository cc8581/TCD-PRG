"""Metrics for complete-action PUSH effectiveness and within-state ranking."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.constants import ActionType, CandidateStatus


def push_effectiveness_metrics(
    probability: Tensor,
    target: Tensor,
    state_id: Tensor | None = None,
) -> dict[str, Tensor]:
    probability = probability.flatten()
    target = target.flatten().bool()
    order = probability.argsort(descending=True, stable=True)
    ranked = target[order].to(probability.dtype)
    positive_count = ranked.sum()
    precision = ranked.cumsum(0) / torch.arange(
        1, len(ranked) + 1, device=probability.device, dtype=probability.dtype
    )
    auprc = (precision * ranked).sum() / positive_count.clamp_min(1)
    negative_count = (~target).sum()
    # Pairwise definition handles ties deterministically and avoids sklearn dependency.
    positive_score = probability[target]
    negative_score = probability[~target]
    if len(positive_score) and len(negative_score):
        comparison = positive_score[:, None] - negative_score[None]
        auroc = ((comparison > 0).float() + 0.5 * (comparison == 0).float()).mean()
    else:
        auroc = probability.new_tensor(float("nan"))
    result = {
        "push_evaluator_auprc": auprc,
        "push_evaluator_auroc": auroc,
        "push_evaluator_positive_count": positive_count,
        "push_evaluator_negative_count": negative_count.to(probability.dtype),
    }
    if state_id is None:
        return result
    hit1: list[Tensor] = []
    recall5: list[Tensor] = []
    precision1: list[Tensor] = []
    for state in torch.unique(state_id):
        mask = state_id == state
        if not bool(target[mask].any()):
            continue
        local_order = probability[mask].argsort(descending=True, stable=True)
        local_target = target[mask][local_order]
        hit1.append(local_target[:1].any().to(probability.dtype))
        recall5.append(local_target[:5].any().to(probability.dtype))
        precision1.append(local_target[:1].float().mean())
    empty = probability.new_tensor(float("nan"))
    result.update(
        {
            "push_evaluator_hit_at_1": torch.stack(hit1).mean() if hit1 else empty,
            "push_evaluator_recall_at_5": torch.stack(recall5).mean() if recall5 else empty,
            "push_evaluator_precision_at_1": torch.stack(precision1).mean()
            if precision1
            else empty,
        }
    )
    return result


def proposal_positive_match_masks(
    rows: list[dict[str, Tensor]],
    batch: dict[str, Tensor],
    *,
    contact_threshold_m: float,
    direction_threshold_deg: float,
) -> list[Tensor]:
    """Mark decoded candidates matching at least one known positive PUSH."""
    parameters = batch["action_parameters"]
    positive = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
        & batch["action_improves_state"].bool()
        & torch.isfinite(parameters["push_contact_world"]).all(-1)
        & torch.isfinite(parameters["push_direction_world"]).all(-1)
    )
    cosine_threshold = math.cos(math.radians(float(direction_threshold_deg)))
    result: list[Tensor] = []
    for row_index, decoded in enumerate(rows):
        matched = torch.zeros(
            len(decoded["object"]), dtype=torch.bool, device=decoded["object"].device
        )
        if len(decoded["object"]):
            predicted_direction = torch.nn.functional.normalize(
                decoded["direction_world"][:, :2], dim=-1
            )
            for action_index in (
                torch.nonzero(positive[row_index], as_tuple=False).flatten().tolist()
            ):
                same_object = decoded["object"] == batch["acted_object"][row_index, action_index]
                contact_distance = torch.linalg.vector_norm(
                    decoded["contact_world"]
                    - parameters["push_contact_world"][row_index, action_index],
                    dim=-1,
                )
                gt_direction = torch.nn.functional.normalize(
                    parameters["push_direction_world"][row_index, action_index, :2],
                    dim=-1,
                )
                direction_match = (predicted_direction * gt_direction[None]).sum(
                    -1
                ) >= cosine_threshold
                matched |= (
                    same_object & (contact_distance <= float(contact_threshold_m)) & direction_match
                )
        result.append(matched)
    return result


def push_candidate_ranking_counts(
    rows: list[dict[str, Tensor]], positive_masks: list[Tensor]
) -> dict[str, Tensor]:
    """Count layer-2 ranking only when Proposal retrieved a positive."""
    if len(rows) != len(positive_masks):
        raise ValueError("rows and positive_masks must have identical lengths")
    reference = next(
        (row["proposal_score"] for row in rows if "proposal_score" in row),
        torch.tensor(0.0),
    )
    candidate_sets = reference.new_zeros(())
    hit1 = reference.new_zeros(())
    hit5 = reference.new_zeros(())
    for row, positive in zip(rows, positive_masks, strict=True):
        if len(positive) != len(row["object"]):
            raise ValueError("positive mask length does not match candidate row")
        if not bool(positive.any()):
            continue
        probability = row.get("effective_probability")
        if probability is None or len(probability) != len(positive):
            raise ValueError("candidate row is missing effective_probability")
        if not bool(torch.isfinite(probability).all()):
            raise ValueError("effective_probability contains non-finite values")
        order = probability.argsort(descending=True, stable=True)
        candidate_sets += 1.0
        hit1 += positive[order[:1]].any().to(reference.dtype)
        hit5 += positive[order[:5]].any().to(reference.dtype)
    return {
        "push_evaluator_candidate_set_count": candidate_sets,
        "push_evaluator_hit_at_1_count": hit1,
        "push_evaluator_recall_at_5_count": hit5,
    }
