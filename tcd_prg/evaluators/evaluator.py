"""Auditable metric aggregation with scene-clustered confidence intervals."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .metrics import (
    binary_auroc,
    binary_average_precision,
    brier_score,
    confusion_metrics,
    expected_calibration_error,
)


METADATA_KEYS = {
    "scene_id", "state_id", "task_index", "category", "task_region",
    "sequence_length", "occlusion_level", "selected_action_type",
}


def bootstrap_confidence_interval(
    values: np.ndarray, samples: int = 1_000, confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


class Evaluator:
    """Store per-state records and aggregate them by resampling whole scenes."""

    def __init__(
        self, bootstrap_samples: int = 1_000, confidence: float = 0.95,
        calibration_bins: int = 15,
    ) -> None:
        self.records: list[dict[str, Any]] = []
        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence = float(confidence)
        self.calibration_bins = int(calibration_bins)

    def add(self, **record: Any) -> None:
        self.records.append(record)

    def _cluster_bootstrap(
        self, records: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float]
    ) -> np.ndarray:
        by_scene: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for index, record in enumerate(records):
            by_scene[record.get("scene_id", ("record", index))].append(record)
        clusters = list(by_scene.values())
        if not clusters or self.bootstrap_samples <= 0:
            return np.empty(0, np.float64)
        rng = np.random.default_rng(2026)
        values = []
        for _ in range(self.bootstrap_samples):
            chosen = rng.integers(0, len(clusters), len(clusters))
            sample = [record for index in chosen for record in clusters[int(index)]]
            value = statistic(sample)
            if np.isfinite(value):
                values.append(value)
        return np.asarray(values, np.float64)

    def _payload(
        self, value: float, count: int, bootstrap: np.ndarray,
        observed: np.ndarray | None = None,
    ) -> dict[str, Any]:
        alpha = (1.0 - self.confidence) / 2.0
        return {
            "mean": float(value) if np.isfinite(value) else None,
            "std": (
                float(np.std(observed, ddof=1)) if observed is not None and len(observed) > 1
                else (float(np.std(bootstrap, ddof=1)) if len(bootstrap) > 1 else 0.0)
            ),
            "ci_low": float(np.quantile(bootstrap, alpha)) if len(bootstrap) else None,
            "ci_high": float(np.quantile(bootstrap, 1.0 - alpha)) if len(bootstrap) else None,
            "count": int(count),
            "cluster_unit": "scene",
        }

    @staticmethod
    def _confusion_stat(records: list[dict[str, Any]], key: str, metric: str) -> float:
        values = [record[key] for record in records if key in record]
        if not values:
            return float("nan")
        totals = np.asarray(values, dtype=np.int64).sum(0)
        return confusion_metrics(*totals.tolist())[metric]

    @staticmethod
    def _binary_arrays(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
        values = [record[key] for record in records if key in record]
        if not values:
            return np.empty(0), np.empty(0, dtype=bool)
        scores = np.concatenate([np.asarray(value[0], np.float64) for value in values])
        targets = np.concatenate([np.asarray(value[1], bool) for value in values])
        return scores, targets

    def summarize(self) -> dict[str, Any]:
        if not self.records:
            return {"count": 0, "scene_count": 0, "metrics": {}}
        metrics: dict[str, dict[str, Any]] = {}
        scalar_keys = sorted({
            key for record in self.records for key, value in record.items()
            if key not in METADATA_KEYS and not key.startswith("_")
            and isinstance(value, (bool, int, float, np.number))
        })
        for key in scalar_keys:
            selected = [
                record for record in self.records
                if key in record and np.isfinite(float(record[key]))
            ]
            values = np.asarray([float(record[key]) for record in selected], np.float64)
            if not len(values):
                continue
            bootstrap = self._cluster_bootstrap(
                selected,
                lambda rows, name=key: float(np.mean([
                    float(row[name]) for row in rows if name in row and np.isfinite(float(row[name]))
                ])),
            )
            metrics[key] = self._payload(float(values.mean()), len(values), bootstrap, values)

        confusion_keys = sorted({
            key for record in self.records for key in record if key.startswith("_confusion_")
        })
        for internal in confusion_keys:
            name = internal.removeprefix("_confusion_")
            count = sum(internal in record for record in self.records)
            for suffix in ("precision", "recall", "f1", "iou", "dice"):
                value = self._confusion_stat(self.records, internal, suffix)
                if not np.isfinite(value):
                    continue
                bootstrap = self._cluster_bootstrap(
                    [record for record in self.records if internal in record],
                    lambda rows, key=internal, metric=suffix: self._confusion_stat(rows, key, metric),
                )
                metrics[f"{name}_{suffix}"] = self._payload(value, count, bootstrap)

        binary_keys = sorted({
            key for record in self.records for key in record if key.startswith("_binary_")
        })
        functions = {
            "auroc": lambda score, target: binary_auroc(score, target),
            "average_precision": lambda score, target: binary_average_precision(score, target),
            "brier": lambda score, target: brier_score(score, target),
            "ece": lambda score, target: expected_calibration_error(
                score, target, bins=self.calibration_bins
            ),
        }
        for internal in binary_keys:
            name = internal.removeprefix("_binary_")
            selected = [record for record in self.records if internal in record]
            scores, targets = self._binary_arrays(selected, internal)
            for suffix, function in functions.items():
                value = function(scores, targets)
                if not np.isfinite(value):
                    continue
                bootstrap = self._cluster_bootstrap(
                    selected,
                    lambda rows, key=internal, fn=function: fn(*self._binary_arrays(rows, key)),
                )
                metrics[f"{name}_{suffix}"] = self._payload(value, len(targets), bootstrap)

        return {
            "count": len(self.records),
            "scene_count": len({record.get("scene_id") for record in self.records}),
            "metrics": metrics,
        }

    def grouped(self, key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, Evaluator] = {}
        for record in self.records:
            name = str(record.get(key, "unknown"))
            groups.setdefault(
                name, Evaluator(
                    self.bootstrap_samples, self.confidence, self.calibration_bins
                )
            ).records.append(record)
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
        (output / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        columns = sorted({
            key for record in self.records for key in record if not key.startswith("_")
        })
        with (output / "per_task.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: value for key, value in record.items() if key in columns})
