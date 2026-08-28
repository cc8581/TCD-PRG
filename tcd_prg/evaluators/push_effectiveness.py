"""Metrics for complete-action PUSH effectiveness and within-state ranking."""

from __future__ import annotations

import torch
from torch import Tensor


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
