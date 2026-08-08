"""Internal Task/Global Grasp convergence diagnostics.

These values are written only to ``validation_grasp_diagnostics.jsonl``.  They
must never enter the standard-only evaluator, metrics.json or paper tables.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from tcd_prg.geometry.grasp_nms import grasp_nms
from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance
from tcd_prg.losses.labels import build_global_grasp_labels, build_grasp_proposal_labels


def _thresholds(
    evaluation_config: Any, prefix: str
) -> tuple[float, float, float, float, float, float]:
    if prefix == "task":
        return (
            float(evaluation_config.task_translation_threshold_m),
            float(evaluation_config.task_rotation_threshold_deg),
            float(evaluation_config.task_width_threshold_m),
            float(evaluation_config.task_nms_translation_m),
            float(evaluation_config.task_nms_rotation_deg),
            float(evaluation_config.task_nms_width_m),
        )
    if prefix == "global":
        return (
            float(evaluation_config.global_translation_threshold_m),
            float(evaluation_config.global_rotation_threshold_deg),
            float(evaluation_config.global_width_threshold_m),
            float(evaluation_config.global_nms_translation_m),
            float(evaluation_config.global_nms_rotation_deg),
            float(evaluation_config.global_nms_width_m),
        )
    raise ValueError(f"Unsupported grasp diagnostic prefix={prefix}")


@torch.no_grad()
def grasp_diagnostic_record(
    output: dict[str, Tensor],
    labels: dict[str, Tensor],
    row: int,
    *,
    prefix: str,
    evaluation_config: Any,
    topk: tuple[int, ...] = (1, 5, 10, 64),
) -> dict[str, float]:
    """Compute one positive-labelled row's diagnostic-only measurements.

    ``quality_hit_at_K`` uses quality ranking after diagnostic NMS.
    ``oracle_hit_at_K`` ignores quality and ranks raw queries by normalized
    nearest-GT geometry error.  Oracle compatibility still requires every
    matching threshold, including object identity for Global Grasp.
    """

    if not bool(labels["sample_valid"][row]):
        return {}
    target_index = torch.nonzero(labels["target_valid"][row], as_tuple=False).flatten()
    if not len(target_index):
        return {}

    (
        translation_threshold,
        rotation_threshold_deg,
        width_threshold,
        nms_translation,
        nms_rotation_deg,
        nms_width,
    ) = _thresholds(evaluation_config, prefix)
    rotation_threshold = math.radians(rotation_threshold_deg)

    prediction_t = output["translation_world"][row].detach().float()
    prediction_r = output["rotation_matrix"][row].detach().float()
    prediction_w = output["width_m"][row].detach().float()
    quality = torch.sigmoid(output["quality_logit"][row].detach().float())
    target_t = labels["translation_world"][row, target_index].detach().float()
    target_r = labels["rotation_matrix"][row, target_index].detach().float()
    target_w = labels["width_m"][row, target_index].detach().float()

    translation = torch.cdist(prediction_t, target_t)
    rotation = parallel_jaw_rotation_distance(
        prediction_r[:, None].expand(-1, len(target_index), -1, -1),
        target_r[None].expand(len(prediction_t), -1, -1, -1),
    )
    width = (prediction_w[:, None] - target_w[None]).abs()

    geometry_cost = (
        translation / max(translation_threshold, 1e-12)
        + rotation / max(rotation_threshold, 1e-12)
        + width / max(width_threshold, 1e-12)
    )
    compatible = (
        (translation <= translation_threshold)
        & (rotation <= rotation_threshold)
        & (width <= width_threshold)
    )

    predicted_object = None
    if prefix == "global":
        object_target = labels.get("object_index")
        if object_target is None or "object_logits" not in output:
            raise KeyError("Global grasp diagnostics require object labels/logits")
        predicted_object = output["object_logits"][row].detach().argmax(-1)
        target_object = object_target[row, target_index]
        compatible &= predicted_object[:, None] == target_object[None]

    nearest_cost, nearest_target = geometry_cost.min(-1)
    oracle_order = torch.argsort(nearest_cost, stable=True)
    nms_order = grasp_nms(
        prediction_t,
        prediction_r,
        prediction_w,
        quality,
        translation_threshold_m=nms_translation,
        rotation_threshold_deg=nms_rotation_deg,
        width_threshold_m=nms_width,
        object_index=predicted_object,
    )

    record: dict[str, float] = {}
    requested = tuple(sorted(set(int(k) for k in (*topk, 64) if int(k) > 0)))
    for k in requested:
        oracle_used = min(k, len(oracle_order))
        quality_used = min(k, len(nms_order))
        record[f"{prefix}_oracle_hit_at_{k}"] = float(
            oracle_used > 0 and bool(compatible[oracle_order[:oracle_used]].any())
        )
        record[f"{prefix}_quality_hit_at_{k}"] = float(
            quality_used > 0 and bool(compatible[nms_order[:quality_used]].any())
        )

    # Highest-quality, NMS-surviving prediction; all component errors use the
    # same nearest normalized-geometry GT so the failure mode is interpretable.
    if len(nms_order):
        q = int(nms_order[0])
        t = int(nearest_target[q])
        record[f"{prefix}_top1_translation_error_m"] = float(translation[q, t])
        record[f"{prefix}_top1_rotation_error_deg"] = math.degrees(float(rotation[q, t]))
        record[f"{prefix}_top1_width_error_m"] = float(width[q, t])
        if prefix == "global":
            target_object = labels["object_index"][row, target_index]
            record["global_top1_object_correct"] = float(
                int(predicted_object[q]) == int(target_object[t])
            )

    # Best-of-Q geometry ignores quality and object assignment.  Object accuracy
    # is therefore diagnosable separately rather than contaminating pose error.
    flat = int(torch.argmin(geometry_cost))
    _, target_count = geometry_cost.shape
    q = flat // target_count
    t = flat % target_count
    record[f"{prefix}_best_translation_error_m"] = float(translation[q, t])
    record[f"{prefix}_best_rotation_error_deg"] = math.degrees(float(rotation[q, t]))
    record[f"{prefix}_best_width_error_m"] = float(width[q, t])
    return record


class GraspDiagnosticAccumulator:
    """Accumulate diagnostic-only scalar sums/counts across validation batches."""

    def __init__(self, model_config: Any, evaluation_config: Any) -> None:
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.records: list[dict[str, float | int]] = []

    def _add(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            if math.isfinite(float(value)):
                self.sums[key] += float(value)
                self.counts[key] += 1

    @torch.no_grad()
    def update(self, batch: dict[str, Any], output: dict[str, Any]) -> None:
        task_labels = (
            build_grasp_proposal_labels(batch, self.model_config)
            if "task_grasp" in output else None
        )
        global_labels = (
            build_global_grasp_labels(batch, self.model_config)
            if "global_grasp" in output else None
        )
        for row in range(batch["xyz"].shape[0]):
            sample = batch["samples"][row]
            sample_record: dict[str, float | int] = {
                "scene_id": int(sample.observation.scene_id),
                "state_id": int(sample.observation.state_id),
                "task_index": int(sample.observation.task_index),
            }
            diagnostic_values: dict[str, float] = {}
            if task_labels is not None:
                diagnostic_values.update(
                    grasp_diagnostic_record(
                        output["task_grasp"],
                        task_labels,
                        row,
                        prefix="task",
                        evaluation_config=self.evaluation_config,
                        topk=tuple(self.evaluation_config.ranking_topk),
                    )
                )
            if global_labels is not None:
                diagnostic_values.update(
                    grasp_diagnostic_record(
                        output["global_grasp"],
                        global_labels,
                        row,
                        prefix="global",
                        evaluation_config=self.evaluation_config,
                        topk=tuple(self.evaluation_config.ranking_topk),
                    )
                )
            if diagnostic_values:
                self._add(diagnostic_values)
                sample_record.update(diagnostic_values)
                self.records.append(sample_record)

    def payload(self) -> dict[str, dict[str, float | int]]:
        return {"sums": dict(self.sums), "counts": dict(self.counts)}
