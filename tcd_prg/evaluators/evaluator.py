"""Unified metric aggregation and reproducible JSON/CSV exports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_METRICS = (
    "task_success_rate_h0", "task_success_rate_h1", "task_success_rate_h3",
    "task_success_rate_h5", "correct_functional_region_grasp_rate",
    "direct_grasp_false_positive", "average_preparation_steps",
    "closed_loop_recovery_rate", "planning_time_s", "execution_time_s",
    "region_miou", "region_dice", "invisible_region_false_positive",
    "task_grasp_recall_at_1", "task_grasp_recall_at_5", "task_grasp_recall_at_10",
    "correct_region_contact_rate_at_10", "wrong_region_grasp_rate_at_10",
    "verifier_overall_auroc", "verifier_overall_f1",
    "physical_edge_f1", "task_blocking_edge_f1",
    "direct_blocker_recall_at_3", "indirect_blocker_f1",
    "actionable_blocker_accuracy", "push_acted_object_top1",
    "push_contact_distance_error_m", "push_direction_angle_error_deg",
    "push_candidate_ndcg",
    "push_utility_delta_mae", "action_type_accuracy",
    "acted_object_accuracy", "successful_action_set_recall", "candidate_ndcg",
)


def bootstrap_confidence_interval(
    values: np.ndarray, samples: int = 1_000, confidence: float = 0.95, seed: int = 2026
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


class Evaluator:
    """Stores per-task records so every aggregate remains auditable."""

    def __init__(self, bootstrap_samples: int = 1_000, confidence: float = 0.95) -> None:
        self.records: list[dict[str, Any]] = []
        self.bootstrap_samples = bootstrap_samples
        self.confidence = confidence

    def add(self, **record: Any) -> None:
        self.records.append(record)

    def summarize(self) -> dict[str, Any]:
        if not self.records:
            return {"count": 0, "metrics": {}}
        numeric: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            for key, value in record.items():
                if isinstance(value, (bool, int, float, np.number)) and np.isfinite(value):
                    numeric[key].append(float(value))
        metrics = {}
        for key, values in numeric.items():
            array = np.asarray(values)
            low, high = bootstrap_confidence_interval(
                array, self.bootstrap_samples, self.confidence, seed=2026
            )
            metrics[key] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "ci_low": low,
                "ci_high": high,
                "count": len(array),
            }
        for key in REQUIRED_METRICS:
            metrics.setdefault(key, {
                "mean": None, "std": None, "ci_low": None, "ci_high": None,
                "count": 0,
                "unavailable_reason": (
                    "No valid labels/evaluated candidates in this subset, or the metric "
                    "requires online execution rather than offline replay."
                ),
            })
        return {"count": len(self.records), "metrics": metrics}

    def grouped(self, key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, Evaluator] = {}
        for record in self.records:
            name = str(record.get(key, "unknown"))
            groups.setdefault(name, Evaluator(self.bootstrap_samples, self.confidence)).records.append(record)
        return {name: evaluator.summarize() for name, evaluator in groups.items()}

    def export(self, output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summarize(),
            "per_category": self.grouped("category"),
            "per_task_region": self.grouped("task_region"),
            "per_sequence_length": self.grouped("sequence_length"),
            "per_occlusion_level": self.grouped("occlusion_level"),
            "config": config or {},
        }
        (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        columns = sorted({key for record in self.records for key in record})
        with (output / "per_task.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.records)
