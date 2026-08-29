from __future__ import annotations

import numpy as np
import pytest

from tcd_prg.evaluators.evaluator import Evaluator
from tcd_prg.evaluators.protocols import (
    graspnet_accuracy_matrix,
    graspnet_metrics_from_friction_scores,
    metric_protocol,
    no_graph_constraint_relation_counts_at_k,
    relation_recall_from_counts,
    summarize_executed_episodes,
)


def test_no_graph_constraint_relation_recall_keeps_multiple_predicates_per_pair() -> None:
    scores = np.array([[[0.9, 0.8, 0.1], [0.7, 0.2, 0.6]]])
    target = np.array([[[True, True, False], [False, False, True]]])
    valid = np.ones_like(target, dtype=bool)
    hits, total = no_graph_constraint_relation_counts_at_k(scores, target, valid, 3)
    assert total.tolist() == [1, 1, 1]
    assert hits.tolist() == [1, 1, 0]
    recall, mean_recall = relation_recall_from_counts(hits, total)
    assert recall == pytest.approx(2 / 3)
    assert mean_recall == pytest.approx(2 / 3)


def test_graspnet_accuracy_matches_topk_denominator_rule() -> None:
    scores = np.array([0.2, 0.6, -1.0])
    matrix = graspnet_accuracy_matrix(
        scores, top_k=4, friction_coefficients=(0.4, 0.6)
    )
    assert matrix[:, 0].tolist() == pytest.approx([1.0, 0.5, 1 / 3, 0.25])
    assert matrix[:, 1].tolist() == pytest.approx([1.0, 1.0, 2 / 3, 0.5])
    metrics = graspnet_metrics_from_friction_scores(
        scores, top_k=4, friction_coefficients=(0.4, 0.6)
    )
    assert metrics["standard_graspnet_AP"] == pytest.approx(matrix.mean())


def test_vpg_metrics_use_only_completed_trials_for_grasp_and_efficiency() -> None:
    metrics = summarize_executed_episodes([
        {
            "completed": True,
            "object_count": 10,
            "grasp_attempts": 5,
            "successful_grasps": 4,
            "total_actions": 10,
            "task_grasp_trial": True,
            "task_success": True,
        },
        {
            "completed": False,
            "object_count": 10,
            "grasp_attempts": 4,
            "successful_grasps": 1,
            "total_actions": 12,
            "task_grasp_trial": True,
            "task_success": False,
        },
    ])
    assert metrics["standard_vpg_completion_rate"] == pytest.approx(0.5)
    assert metrics["standard_vpg_grasp_success_rate"] == pytest.approx(0.8)
    assert metrics["standard_vpg_action_efficiency"] == pytest.approx(1.0)
    assert metrics["standard_task_grasp_task_success_rate"] == pytest.approx(0.5)
    assert all(key.startswith("standard_") for key in metrics)


def test_nonstandard_metric_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        metric_protocol("task_grasp_known_hit_at_1")
    with pytest.raises(ValueError):
        metric_protocol("selected_candidate_success")
    assert metric_protocol("standard_region_miou").task == "task_region"
    assert metric_protocol("standard_target_iou").task == "target_instance"


def test_evaluator_exports_only_standard_metrics() -> None:
    evaluator = Evaluator(bootstrap_samples=0)
    evaluator.add(
        scene_id=1,
        selected_candidate_success=1.0,
        task_grasp_known_hit_at_1=1.0,
        _confusion_region_foreground=(9, 1, 1, 9),
        _confusion_standard_region_foreground=(2, 1, 1, 6),
        _confusion_standard_region_background=(6, 1, 1, 2),
        _relation_counts_standard_task_relation_ng_at_50=([1, 0, 1], [1, 1, 1]),
        standard_task_relation_ng_recall_at_50=2 / 3,
    )
    summary = evaluator.summarize()["metrics"]
    assert "selected_candidate_success" not in summary
    assert "task_grasp_known_hit_at_1" not in summary
    assert "region_foreground_iou" not in summary
    assert "standard_region_miou" in summary
    assert "standard_task_relation_ng_recall_at_50" in summary
    assert "standard_task_relation_ng_mean_recall_at_50" in summary
