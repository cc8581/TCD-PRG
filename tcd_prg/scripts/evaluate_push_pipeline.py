"""Evaluate Oracle and A+C two-layer PUSH proposal/evaluator protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators.push_effectiveness import (
    proposal_known_outcome_masks,
    push_candidate_ranking_counts,
)
from tcd_prg.evaluators.push_integrated import integrated_push_proposal_counts
from tcd_prg.models import StandalonePushModel, TCDPRGModel, push_condition_from_gt
from tcd_prg.models.staged_checkpoint import (
    load_perception_stage,
    load_push_evaluator,
    load_push_stage,
)
from tcd_prg.planners.push_decoder import decode_push_candidates, proposal_recall_counts
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter


def _device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    return value


@torch.no_grad()
def oracle_push_pipeline_counts(stage_c, batch, config) -> dict[str, torch.Tensor]:
    stage_c_batch = dict(batch)
    stage_c_batch["push_condition"] = push_condition_from_gt(batch, config.model.instance_queries)
    proposal = stage_c(stage_c_batch, forward_mode="push")
    pre_nms, final = decode_push_candidates(
        proposal["sensor"],
        proposal["push_condition"],
        proposal["push"],
        config.model,
        use_push_potential=config.ablation.use_push_potential,
    )
    pre_hits, total = proposal_recall_counts(
        pre_nms,
        batch,
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    final_hits, final_total = proposal_recall_counts(
        final,
        batch,
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    if not bool(torch.equal(total, final_total)):
        raise RuntimeError("Oracle PUSH denominator changed across NMS")
    for row_index, row in enumerate(final):
        if len(row["point_index"]):
            logits = stage_c.push_evaluator(proposal["push"], row, batch_index=row_index)
            row["effective_logit"] = logits
            row["effective_probability"] = torch.sigmoid(logits)
    positive_masks, _, known_masks = proposal_known_outcome_masks(
        final,
        batch,
        contact_threshold_m=config.evaluation.push_match_contact_m,
        direction_threshold_deg=config.evaluation.push_match_direction_deg,
    )
    ranking = push_candidate_ranking_counts(final, positive_masks, known_masks)
    return {
        "oracle_push_proposal_positive_total_count": total,
        "oracle_push_proposal_positive_pre_nms_hits_count": pre_hits,
        "oracle_push_proposal_positive_final_hits_count": final_hits,
        "oracle_push_evaluator_positive_candidate_set_count": ranking[
            "push_evaluator_positive_candidate_set_count"
        ],
        "oracle_push_evaluator_top1_evaluable_count": ranking[
            "push_evaluator_top1_evaluable_count"
        ],
        "oracle_push_evaluator_top5_evaluable_count": ranking[
            "push_evaluator_top5_evaluable_count"
        ],
        "oracle_push_evaluator_hit_at_1_count": ranking["push_evaluator_hit_at_1_count"],
        "oracle_push_evaluator_recall_at_5_count": ranking["push_evaluator_recall_at_5_count"],
        "oracle_push_evaluator_known_candidate_count": ranking[
            "push_evaluator_known_candidate_count"
        ],
        "oracle_push_evaluator_total_candidate_count": ranking[
            "push_evaluator_total_candidate_count"
        ],
    }


def _ratio(payload: dict[str, float], numerator: str, denominator: str) -> float | None:
    value = payload.get(denominator, 0.0)
    return payload.get(numerator, 0.0) / value if value else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-c-checkpoint", required=True)
    parser.add_argument("--push-evaluator-checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--output", default="outputs/push_pipeline_metrics.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split=args.split,
        max_groups=config.evaluation.max_groups,
        global_grasp_mode="never",
    )
    if not len(dataset):
        raise RuntimeError(f"No action groups exist for split={args.split}")
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.validation_num_workers,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    stage_a = TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet).to(
        device
    )
    stage_c = StandalonePushModel(config.model).to(device)
    load_perception_stage(stage_a, args.stage_a_checkpoint, config)
    load_push_stage(stage_c, args.stage_c_checkpoint, config)
    load_push_evaluator(
        stage_c,
        args.push_evaluator_checkpoint,
        proposal_checkpoint=args.stage_c_checkpoint,
    )
    stage_a.eval()
    stage_c.eval()
    totals: dict[str, float] = {}
    for cpu_batch in loader:
        batch = _device(cpu_batch, device)
        for result in (
            oracle_push_pipeline_counts(stage_c, batch, config),
            integrated_push_proposal_counts(stage_a, stage_c, batch, config),
        ):
            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    metrics = dict(totals)
    for prefix in ("oracle", "integrated"):
        metrics[f"{prefix}_push_proposal_positive_recall_pre_nms_at_32"] = _ratio(
            totals,
            f"{prefix}_push_proposal_positive_pre_nms_hits_count",
            f"{prefix}_push_proposal_positive_total_count",
        )
        metrics[f"{prefix}_push_proposal_positive_recall_final_at_32"] = _ratio(
            totals,
            f"{prefix}_push_proposal_positive_final_hits_count",
            f"{prefix}_push_proposal_positive_total_count",
        )
        metrics[f"{prefix}_push_evaluator_hit_at_1"] = _ratio(
            totals,
            f"{prefix}_push_evaluator_hit_at_1_count",
            f"{prefix}_push_evaluator_top1_evaluable_count",
        )
        metrics[f"{prefix}_push_evaluator_recall_at_5"] = _ratio(
            totals,
            f"{prefix}_push_evaluator_recall_at_5_count",
            f"{prefix}_push_evaluator_top5_evaluable_count",
        )
        metrics[f"{prefix}_push_evaluator_top1_evaluable_rate"] = _ratio(
            totals,
            f"{prefix}_push_evaluator_top1_evaluable_count",
            f"{prefix}_push_evaluator_positive_candidate_set_count",
        )
        metrics[f"{prefix}_push_evaluator_top5_evaluable_rate"] = _ratio(
            totals,
            f"{prefix}_push_evaluator_top5_evaluable_count",
            f"{prefix}_push_evaluator_positive_candidate_set_count",
        )
        metrics[f"{prefix}_push_evaluator_known_candidate_coverage"] = _ratio(
            totals,
            f"{prefix}_push_evaluator_known_candidate_count",
            f"{prefix}_push_evaluator_total_candidate_count",
        )
    payload = {
        "split": args.split,
        "group_count": len(dataset),
        "metrics": metrics,
        "protocol": {
            "push_distance_m": 0.15,
            "unknown_is_negative": False,
            "partial_label_warning": (
                "Report evaluable rates and known-candidate coverage with Hit@1/Recall@5"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
