"""Evaluation-only A+C PUSH proposal metrics with Hungarian query association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import TCDPRGConfig
from tcd_prg.losses.instance import InstanceSetLoss, build_instance_targets
from tcd_prg.planners.push_decoder import (
    decode_push_candidates,
    proposal_recall_counts,
)


@torch.no_grad()
def integrated_push_proposal_counts(
    stage_a: nn.Module,
    stage_c: nn.Module,
    batch: Mapping[str, Any],
    config: TCDPRGConfig,
) -> dict[str, Tensor]:
    """Run predicted PushCondition; use Hungarian matching only for metrics."""
    perception = stage_a(batch, forward_mode="perception")
    stage_c_batch = dict(batch)
    stage_c_batch["push_condition"] = perception["push_condition"]
    proposal = stage_c(stage_c_batch, forward_mode="push")
    pre_nms, final = decode_push_candidates(
        proposal["sensor"],
        proposal["push_condition"],
        proposal["push"],
        config.model,
        use_push_potential=config.ablation.use_push_potential,
    )

    targets = build_instance_targets(dict(batch), config.model.instance_queries)
    matcher = InstanceSetLoss(
        matching_points=config.model.instance_matching_points
    )
    match = matcher.match(perception["instance"], targets)

    def associated(rows: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        result: list[dict[str, Tensor]] = []
        for row_index, row in enumerate(rows):
            converted = dict(row)
            query = row["object"].long()
            mapped = match.query_to_gt[row_index, query] if len(query) else query
            converted["object"] = mapped
            # An unmatched predicted query can never satisfy same-GT-object.
            result.append(converted)
        return result

    pre_hits, total = proposal_recall_counts(
        associated(pre_nms),
        dict(batch),
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    final_hits, final_total = proposal_recall_counts(
        associated(final),
        dict(batch),
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    if not bool(torch.equal(total, final_total)):
        raise RuntimeError("Integrated PUSH denominator changed across NMS")
    return {
        "integrated_push_proposal_positive_total_count": total,
        "integrated_push_proposal_positive_pre_nms_hits_count": pre_hits,
        "integrated_push_proposal_positive_final_hits_count": final_hits,
    }
