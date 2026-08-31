"""Evaluation-only A+C PUSH proposal metrics with Hungarian query association."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import TCDPRGConfig
from tcd_prg.evaluators.push_effectiveness import (
    proposal_known_outcome_masks,
    push_candidate_ranking_counts,
)
from tcd_prg.losses.instance import InstanceSetLoss, build_instance_targets
from tcd_prg.planners.push_decoder import (
    decode_push_candidates,
    proposal_recall_counts,
)


def deterministic_target_prompt_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Synthesize the deterministic observable target click used by validation."""
    result = dict(batch)
    existing = batch.get("task_inputs")
    if isinstance(existing, Mapping) and all(
        key in existing
        for key in ("target_prompt_xyz", "target_prompt_label", "target_prompt_valid")
    ):
        return result

    xyz = batch["xyz"]
    valid_target = batch["point_mask"].bool() & batch["target_mask"].bool()
    batch_size = xyz.shape[0]
    prompt_xyz = xyz.new_zeros((batch_size, 1, 3))
    prompt_label = torch.ones((batch_size, 1), dtype=torch.long, device=xyz.device)
    prompt_valid = torch.zeros((batch_size, 1), dtype=torch.bool, device=xyz.device)
    for row in range(batch_size):
        candidates = torch.nonzero(valid_target[row], as_tuple=False).flatten()
        if not len(candidates):
            continue
        points = xyz[row, candidates]
        centroid = points.mean(0)
        choice = candidates[torch.linalg.vector_norm(points - centroid, dim=-1).argmin()]
        prompt_xyz[row, 0] = xyz[row, choice]
        prompt_valid[row, 0] = True

    task_inputs = dict(existing) if isinstance(existing, Mapping) else {}
    task_inputs.update(
        task_category_id=batch["task_category_id"],
        task_region_id=batch["task_region_id"],
        target_prompt_xyz=prompt_xyz,
        target_prompt_label=prompt_label,
        target_prompt_valid=prompt_valid,
    )
    result["task_inputs"] = task_inputs
    return result


@torch.no_grad()
def integrated_push_proposal_counts(
    stage_a: nn.Module,
    stage_c: nn.Module,
    batch: Mapping[str, Any],
    config: TCDPRGConfig,
) -> dict[str, Tensor]:
    """Run deployment-equivalent prompted Stage A+C; matching is metrics-only."""
    perception = stage_a(deterministic_target_prompt_batch(batch), forward_mode="perception")
    stage_c_batch = dict(batch)
    stage_c_batch["push_condition"] = perception["push_condition"]
    stage_c_batch["geometry_feature"] = perception["encoded"].scene_point_features.detach()
    proposal = stage_c(stage_c_batch, forward_mode="push")
    pre_nms, final = decode_push_candidates(
        proposal["sensor"],
        proposal["push_condition"],
        proposal["push"],
        config.model,
    )
    wrapped_stage_c = stage_c.module if hasattr(stage_c, "module") else stage_c
    push_evaluator = getattr(wrapped_stage_c, "push_evaluator", None)
    if push_evaluator is None or not bool(
        getattr(wrapped_stage_c, "push_evaluator_ready", False)
    ):
        raise RuntimeError(
            "Integrated PUSH evaluation requires a loaded PushEffectivenessEvaluator"
        )
    targets = build_instance_targets(dict(batch), config.model.instance_queries)
    matcher = InstanceSetLoss(matching_points=config.model.instance_matching_points)
    match = matcher.match(perception["instance"], targets)
    predicted_mask = perception["instance"].mask_probability >= 0.5
    target_mask = targets["mask"].bool()
    valid_query_to_gt = match.query_to_gt.clone()
    for row_index in range(valid_query_to_gt.shape[0]):
        for query_index in (
            torch.nonzero(valid_query_to_gt[row_index] >= 0, as_tuple=False).flatten().tolist()
        ):
            gt_index = int(valid_query_to_gt[row_index, query_index])
            intersection = (
                (predicted_mask[row_index, query_index] & target_mask[row_index, gt_index])
                .sum()
                .float()
            )
            union = (
                (predicted_mask[row_index, query_index] | target_mask[row_index, gt_index])
                .sum()
                .float()
            )
            iou = intersection / union.clamp_min(1.0)
            if float(iou) < config.evaluation.instance_match_iou_threshold:
                valid_query_to_gt[row_index, query_index] = -1

    def associated(rows: list[dict[str, Tensor]]) -> list[dict[str, Tensor]]:
        result: list[dict[str, Tensor]] = []
        for row_index, row in enumerate(rows):
            converted = dict(row)
            query = row["object"].long()
            mapped = valid_query_to_gt[row_index, query] if len(query) else query
            converted["object"] = mapped
            # An unmatched predicted query can never satisfy same-GT-object.
            result.append(converted)
        return result

    associated_pre = associated(pre_nms)
    associated_final = associated(final)
    pre_hits, total = proposal_recall_counts(
        associated_pre,
        dict(batch),
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    final_hits, final_total = proposal_recall_counts(
        associated_final,
        dict(batch),
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    if not bool(torch.equal(total, final_total)):
        raise RuntimeError("Integrated PUSH denominator changed across NMS")
    positive_masks, _, known_masks = proposal_known_outcome_masks(
        associated_final,
        dict(batch),
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    ranking = push_candidate_ranking_counts(final, positive_masks, known_masks)
    return {
        "integrated_push_proposal_positive_total_count": total,
        "integrated_push_proposal_positive_pre_nms_hits_count": pre_hits,
        "integrated_push_proposal_positive_final_hits_count": final_hits,
        "integrated_push_evaluator_positive_candidate_set_count": ranking[
            "push_evaluator_positive_candidate_set_count"
        ],
        "integrated_push_evaluator_top1_evaluable_count": ranking[
            "push_evaluator_top1_evaluable_count"
        ],
        "integrated_push_evaluator_top5_evaluable_count": ranking[
            "push_evaluator_top5_evaluable_count"
        ],
        "integrated_push_evaluator_hit_at_1_count": ranking["push_evaluator_hit_at_1_count"],
        "integrated_push_evaluator_recall_at_5_count": ranking["push_evaluator_recall_at_5_count"],
        "integrated_push_evaluator_known_candidate_count": ranking[
            "push_evaluator_known_candidate_count"
        ],
        "integrated_push_evaluator_total_candidate_count": ranking[
            "push_evaluator_total_candidate_count"
        ],
    }
