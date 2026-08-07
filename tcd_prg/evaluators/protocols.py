"""Kernels for externally comparable evaluation protocols only.

No TCD-specific proxy or diagnostic metric is defined here. If current labels do
not support a field's accepted protocol, that field produces no metric until the
required benchmark/physical execution data are supplied.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

GRASPNET_FRICTION_COEFFICIENTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
GRASPNET_TOP_K = 50


@dataclass(frozen=True, slots=True)
class MetricProtocol:
    task: str
    protocol: str
    note: str


def no_graph_constraint_relation_counts_at_k(
    scores: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-predicate hits/GT for no-graph-constraint relation Recall@K."""

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != target.shape or scores.shape != valid.shape:
        raise ValueError("scores, target and valid must have identical shapes")
    if scores.ndim < 1 or k <= 0:
        raise ValueError("relation scores need a predicate dimension and positive k")

    classes = scores.shape[-1]
    flat_score = scores.reshape(-1)
    flat_target = target.reshape(-1)
    flat_valid = valid.reshape(-1) & np.isfinite(flat_score)
    predicate = np.broadcast_to(np.arange(classes, dtype=np.int64), scores.shape).reshape(-1)

    total = np.bincount(predicate[flat_valid & flat_target], minlength=classes).astype(
        np.int64, copy=False
    )
    candidates = np.flatnonzero(flat_valid)
    if not len(candidates):
        return np.zeros(classes, np.int64), total
    order = candidates[np.argsort(-flat_score[candidates], kind="mergesort")[:k]]
    matched = order[flat_target[order]]
    hits = np.bincount(predicate[matched], minlength=classes).astype(np.int64, copy=False)
    return hits, total


def relation_recall_from_counts(hits: np.ndarray, total: np.ndarray) -> tuple[float, float]:
    """Return dataset R@K and predicate-macro mR@K from accumulated counts."""

    hits = np.asarray(hits, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    if hits.shape != total.shape:
        raise ValueError("hits and total must have identical shapes")
    recall = float(hits.sum() / total.sum()) if total.sum() > 0 else float("nan")
    present = total > 0
    mean_recall = float(np.mean(hits[present] / total[present])) if present.any() else float("nan")
    return recall, mean_recall


def graspnet_accuracy_matrix(
    friction_scores: np.ndarray,
    *,
    top_k: int = GRASPNET_TOP_K,
    friction_coefficients: Iterable[float] = GRASPNET_FRICTION_COEFFICIENTS,
) -> np.ndarray:
    """GraspNet Precision@K-by-friction matrix from official physical scores."""

    scores = np.asarray(friction_scores, dtype=np.float64).reshape(-1)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    frictions = np.asarray(tuple(float(value) for value in friction_coefficients), np.float64)
    if not len(frictions) or np.any(frictions <= 0):
        raise ValueError("friction_coefficients must contain positive values")

    result = np.zeros((top_k, len(frictions)), dtype=np.float64)
    for friction_index, friction in enumerate(frictions):
        for rank in range(top_k):
            prefix = scores if rank + 1 > len(scores) else scores[: rank + 1]
            success = (prefix <= friction) & (prefix > 0)
            result[rank, friction_index] = float(np.count_nonzero(success) / (rank + 1))
    return result


def graspnet_metrics_from_friction_scores(
    friction_scores: np.ndarray,
    *,
    top_k: int = GRASPNET_TOP_K,
    friction_coefficients: Iterable[float] = GRASPNET_FRICTION_COEFFICIENTS,
) -> dict[str, float]:
    """GraspNet AP/AP_mu values from official force-closure/collision scores."""

    frictions = tuple(float(value) for value in friction_coefficients)
    accuracy = graspnet_accuracy_matrix(
        friction_scores, top_k=top_k, friction_coefficients=frictions
    )
    result = {"standard_graspnet_AP": float(np.mean(accuracy))}
    for index, friction in enumerate(frictions):
        result[f"standard_graspnet_AP_mu_{friction:.1f}"] = float(np.mean(accuracy[:, index]))
    return result


def summarize_executed_episodes(
    episodes: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute only verified execution metrics.

    VPG protocol fields:
      completed, object_count, grasp_attempts, successful_grasps, total_actions.
    VPG grasp success and action efficiency are averaged over completed trials.

    Optional task-oriented-grasp field:
      task_grasp_trial, task_success.  If present, task success rate is the
      fraction of executed task-grasp trials that complete the requested task.
    """

    rows = [dict(item) for item in episodes]
    if not rows:
        raise ValueError("at least one episode is required")

    for index, row in enumerate(rows):
        for name in ("object_count", "grasp_attempts", "successful_grasps", "total_actions"):
            if name in row and int(row[name]) < 0:
                raise ValueError(f"episode {index}: {name} must be non-negative")
        if int(row.get("successful_grasps", 0)) > int(row.get("grasp_attempts", 0)):
            raise ValueError(f"episode {index}: successful_grasps exceeds grasp_attempts")

    completed = np.asarray([bool(row.get("completed", False)) for row in rows], dtype=bool)
    result: dict[str, float] = {
        "standard_vpg_completion_rate": float(completed.mean()),
    }

    completed_rows = [row for row, ok in zip(rows, completed.tolist(), strict=True) if ok]
    if completed_rows:
        grasp_rates = []
        action_efficiencies = []
        for row in completed_rows:
            attempts = int(row.get("grasp_attempts", 0))
            successes = int(row.get("successful_grasps", 0))
            actions = int(row.get("total_actions", 0))
            objects = int(row.get("object_count", 0))
            if attempts <= 0:
                raise ValueError("completed VPG trial must contain a grasp attempt")
            if actions <= 0:
                raise ValueError("completed VPG trial must contain an action")
            grasp_rates.append(successes / attempts)
            action_efficiencies.append(objects / actions)
        result["standard_vpg_grasp_success_rate"] = float(np.mean(grasp_rates))
        result["standard_vpg_action_efficiency"] = float(np.mean(action_efficiencies))
    else:
        result["standard_vpg_grasp_success_rate"] = float("nan")
        result["standard_vpg_action_efficiency"] = float("nan")

    task_trials = [row for row in rows if bool(row.get("task_grasp_trial", False))]
    if task_trials:
        result["standard_task_grasp_task_success_rate"] = float(
            np.mean([bool(row.get("task_success", False)) for row in task_trials])
        )
    return result


def metric_protocol(name: str) -> MetricProtocol:
    """Return the audited protocol for a standard metric; reject other names."""

    if not name.startswith("standard_"):
        raise ValueError(f"Non-standard metric is not exportable: {name}")
    if "graspnet" in name:
        return MetricProtocol(
            "global_grasp",
            "GraspNet Precision@K/AP_mu/AP",
            "Requires official GraspNet physical evaluation.",
        )
    if "relation" in name:
        return MetricProtocol(
            "dependency_graph",
            "SGG no-graph-constraint R@K/mR@K",
            "Multiple predicates per relation pair remain eligible.",
        )
    if "region" in name:
        return MetricProtocol(
            "task_region",
            "binary semantic segmentation IoU/mIoU",
            "Dataset-level per-class confusion aggregation.",
        )
    if "verifier" in name:
        return MetricProtocol(
            "verifier",
            "binary classification AP/AUROC/Precision/Recall/F1/Brier/ECE",
            "Candidate-level pooled predictions with explicit valid labels.",
        )
    if "vpg_" in name:
        return MetricProtocol(
            "push_policy",
            "VPG Completion/Grasp Success/Action Efficiency",
            "Requires executed closed-loop trials.",
        )
    if "task_grasp_task_success_rate" in name:
        return MetricProtocol(
            "task_grasp",
            "executed task-oriented grasp task success rate",
            "Requires real/simulated execution and task outcome.",
        )
    raise ValueError(f"Unknown standard metric protocol: {name}")


def protocol_audit_manifest() -> dict[str, Any]:
    """State exactly which protocols are available from each evaluation path."""

    return {
        "task_region": {
            "protocol": "binary segmentation IoU/mIoU",
            "offline_supported": True,
        },
        "dependency_graph": {
            "protocol": "SGG no-graph-constraint R@K/mR@K",
            "offline_supported": True,
        },
        "verifier": {
            "protocol": "binary classification AP/AUROC/Precision/Recall/F1/Brier/ECE",
            "offline_supported": True,
        },
        "global_grasp": {
            "protocol": "official GraspNet Precision@K/AP_mu/AP",
            "offline_supported": False,
            "entrypoint": "tcd-prg-eval-graspnet",
        },
        "task_grasp": {
            "protocol": "executed task success rate",
            "offline_supported": False,
            "entrypoint": "tcd-prg-eval-episodes",
        },
        "push_policy": {
            "protocol": "VPG Completion/Grasp Success/Action Efficiency",
            "offline_supported": False,
            "entrypoint": "tcd-prg-eval-episodes",
        },
    }
