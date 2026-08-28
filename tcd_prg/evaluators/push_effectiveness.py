"""Metrics for complete-action PUSH effectiveness and within-state ranking."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.constants import ActionType, CandidateStatus


def _binary_auroc_rank(probability: Tensor, target: Tensor) -> Tensor:
    """Exact tie-aware AUROC with O(N log N) memory."""
    target = target.bool()
    positive_count = target.sum()
    negative_count = (~target).sum()
    if not int(positive_count) or not int(negative_count):
        return probability.new_tensor(float("nan"))
    order = probability.argsort(descending=False, stable=True)
    sorted_score = probability[order]
    sorted_target = target[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    average_rank = (starts.to(probability.dtype) + 1.0 + ends.to(probability.dtype)) / 2.0
    group_index = torch.repeat_interleave(
        torch.arange(len(counts), device=probability.device), counts
    )
    positive_rank_sum = average_rank[group_index][sorted_target].sum()
    positive = positive_count.to(probability.dtype)
    negative = negative_count.to(probability.dtype)
    mann_whitney = positive_rank_sum - positive * (positive + 1.0) / 2.0
    return mann_whitney / (positive * negative)


def push_effectiveness_metrics(
    probability: Tensor,
    target: Tensor,
    state_id: Tensor | None = None,
) -> dict[str, Tensor]:
    probability = probability.flatten()
    target = target.flatten().bool()
    if probability.numel() != target.numel():
        raise ValueError("probability and target must have identical lengths")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("PUSH evaluator probability contains non-finite values")
    order = probability.argsort(descending=True, stable=True)
    ranked = target[order].to(probability.dtype)
    positive_count = ranked.sum()
    precision = ranked.cumsum(0) / torch.arange(
        1, len(ranked) + 1, device=probability.device, dtype=probability.dtype
    )
    auprc = (
        (precision * ranked).sum() / positive_count
        if int(positive_count)
        else probability.new_tensor(float("nan"))
    )
    negative_count = (~target).sum()
    auroc = _binary_auroc_rank(probability, target)
    result = {
        "push_evaluator_auprc": auprc,
        "push_evaluator_auroc": auroc,
        "push_evaluator_positive_count": positive_count,
        "push_evaluator_negative_count": negative_count.to(probability.dtype),
    }
    if state_id is None:
        return result
    state_id = state_id.flatten()
    if state_id.numel() != probability.numel():
        raise ValueError("state_id must align with probability")
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


def proposal_known_outcome_masks(
    rows: list[dict[str, Tensor]],
    batch: dict[str, Tensor],
    *,
    contact_threshold_m: float,
    direction_threshold_deg: float,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Match proposals to unambiguous known positive/negative outcomes."""
    parameters = batch["action_parameters"]
    evaluated = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
        & torch.isfinite(parameters["push_contact_world"]).all(-1)
        & torch.isfinite(parameters["push_direction_world"]).all(-1)
    )
    improves = batch["action_improves_state"].bool()
    cosine_threshold = math.cos(math.radians(float(direction_threshold_deg)))
    positive_rows: list[Tensor] = []
    negative_rows: list[Tensor] = []
    known_rows: list[Tensor] = []
    for row_index, decoded in enumerate(rows):
        positive_match = torch.zeros(
            len(decoded["object"]), dtype=torch.bool, device=decoded["object"].device
        )
        negative_match = torch.zeros_like(positive_match)
        if len(decoded["object"]):
            predicted_direction = torch.nn.functional.normalize(
                decoded["direction_world"][:, :2], dim=-1
            )
            for action_index in (
                torch.nonzero(evaluated[row_index], as_tuple=False).flatten().tolist()
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
                matched = (
                    same_object & (contact_distance <= float(contact_threshold_m)) & direction_match
                )
                if bool(improves[row_index, action_index]):
                    positive_match |= matched
                else:
                    negative_match |= matched
        ambiguous = positive_match & negative_match
        positive_match &= ~ambiguous
        negative_match &= ~ambiguous
        positive_rows.append(positive_match)
        negative_rows.append(negative_match)
        known_rows.append(positive_match | negative_match)
    return positive_rows, negative_rows, known_rows


def proposal_positive_match_masks(
    rows: list[dict[str, Tensor]],
    batch: dict[str, Tensor],
    *,
    contact_threshold_m: float,
    direction_threshold_deg: float,
) -> list[Tensor]:
    positive, _, _ = proposal_known_outcome_masks(
        rows,
        batch,
        contact_threshold_m=contact_threshold_m,
        direction_threshold_deg=direction_threshold_deg,
    )
    return positive


def push_candidate_ranking_counts(
    rows: list[dict[str, Tensor]],
    positive_masks: list[Tensor],
    known_masks: list[Tensor] | None = None,
) -> dict[str, Tensor]:
    """Evaluate partial-label ranking without treating UNKNOWN as negative."""
    if known_masks is None:
        known_masks = positive_masks
    if not (len(rows) == len(positive_masks) == len(known_masks)):
        raise ValueError("rows and outcome masks must have identical lengths")
    reference = next(
        (row["proposal_score"] for row in rows if "proposal_score" in row),
        torch.tensor(0.0),
    )
    positive_sets = reference.new_zeros(())
    top1_evaluable = reference.new_zeros(())
    top5_evaluable = reference.new_zeros(())
    hit1 = reference.new_zeros(())
    hit5 = reference.new_zeros(())
    known_candidate_count = reference.new_zeros(())
    total_candidate_count = reference.new_zeros(())
    for row, positive, known in zip(rows, positive_masks, known_masks, strict=True):
        if len(positive) != len(row["object"]) or len(known) != len(row["object"]):
            raise ValueError("outcome mask length does not match candidate row")
        total_candidate_count += float(len(known))
        known_candidate_count += known.sum().to(reference.dtype)
        if not bool(positive.any()):
            continue
        probability = row.get("effective_probability")
        if probability is None or len(probability) != len(positive):
            raise ValueError("candidate row is missing effective_probability")
        if not bool(torch.isfinite(probability).all()):
            raise ValueError("effective_probability contains non-finite values")
        order = probability.argsort(descending=True, stable=True)
        positive_sets += 1.0
        top1 = order[:1]
        if bool(known[top1].all()):
            top1_evaluable += 1.0
            hit1 += positive[top1].any().to(reference.dtype)
        top5 = order[: min(5, len(order))]
        if bool(positive[top5].any()):
            top5_evaluable += 1.0
            hit5 += 1.0
        elif len(top5) and bool(known[top5].all()):
            top5_evaluable += 1.0
    return {
        "push_evaluator_positive_candidate_set_count": positive_sets,
        "push_evaluator_top1_evaluable_count": top1_evaluable,
        "push_evaluator_top5_evaluable_count": top5_evaluable,
        "push_evaluator_hit_at_1_count": hit1,
        "push_evaluator_recall_at_5_count": hit5,
        "push_evaluator_known_candidate_count": known_candidate_count,
        "push_evaluator_total_candidate_count": total_candidate_count,
    }
