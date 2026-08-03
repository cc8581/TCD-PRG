from __future__ import annotations

import numpy as np
import pytest

from tcd_prg.evaluators.evaluator import Evaluator
from tcd_prg.evaluators.offline import OfflineModelEvaluator
from tcd_prg.constants import ActionType
from tcd_prg.evaluators.metrics import (
    binary_auroc,
    binary_average_precision,
    binary_confusion,
    confusion_metrics,
)


def test_binary_ranking_metrics_group_equal_scores() -> None:
    scores = np.array([0.5, 0.5])
    first = np.array([False, True])
    second = first[::-1]
    assert binary_auroc(scores, first) == pytest.approx(0.5)
    assert binary_auroc(scores, second) == pytest.approx(0.5)
    assert binary_average_precision(scores, first) == pytest.approx(0.5)
    assert binary_average_precision(scores, second) == pytest.approx(0.5)


def test_confusion_metrics_use_global_counts() -> None:
    counts = binary_confusion(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert counts == (1, 1, 1, 1)
    metrics = confusion_metrics(*counts)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1 / 3)


def test_evaluator_uses_scene_clusters_and_has_no_empty_metric_placeholders() -> None:
    evaluator = Evaluator(bootstrap_samples=50, confidence=0.95)
    evaluator.add(
        scene_id=1, state_id=0, selected_candidate_success=1.0,
        _confusion_region_foreground=(2, 1, 0, 4),
        _binary_verifier_overall=([0.9, 0.2], [True, False]),
    )
    evaluator.add(
        scene_id=1, state_id=1, selected_candidate_success=0.0,
        _confusion_region_foreground=(1, 0, 1, 4),
    )
    evaluator.add(
        scene_id=2, state_id=0, selected_candidate_success=1.0,
        _confusion_region_foreground=(1, 1, 0, 3),
        _binary_verifier_overall=([0.8, 0.1], [True, False]),
    )
    summary = evaluator.summarize()
    assert summary["count"] == 3
    assert summary["scene_count"] == 2
    assert summary["metrics"]["selected_candidate_success"]["mean"] == pytest.approx(2 / 3)
    assert summary["metrics"]["selected_candidate_success"]["cluster_unit"] == "scene"
    assert summary["metrics"]["region_foreground_precision"]["mean"] == pytest.approx(4 / 6)
    assert summary["metrics"]["verifier_overall_auroc"]["mean"] == pytest.approx(1.0)
    assert all(value["count"] > 0 for value in summary["metrics"].values())
    assert "task_grasp_recall_at_1" not in summary["metrics"]


def test_labelled_replay_is_not_named_online_task_success() -> None:
    evaluator = OfflineModelEvaluator(model_config=object(), bootstrap_samples=0)
    root = {"scene_id": 3, "state_id": 0, "task_index": 1}
    terminal = {"scene_id": 3, "state_id": 1, "task_index": 1}
    evaluator.evaluator.records = [root, terminal]
    evaluator.decisions = {
        (3, 0, 1): {
            "index": 0, "action_type": np.array([int(ActionType.PUSH)]),
            "evaluated": np.array([True]), "success": np.array([True]),
            "to_state": np.array([1]), "after_state_valid": np.array([True]),
            "depth": 0, "graspable": False, "record": root,
        },
        (3, 1, 1): {
            "index": 0, "action_type": np.array([int(ActionType.TASK_GRASP)]),
            "evaluated": np.array([True]), "success": np.array([True]),
            "to_state": np.array([-1]), "after_state_valid": np.array([False]),
            "depth": 1, "graspable": True, "record": terminal,
        },
    }
    evaluator.finalize_closed_loop_replay((0, 1))
    assert root["labelled_replay_task_success_h0"] == 0.0
    assert root["labelled_replay_task_success_h1"] == 1.0
    assert root["labelled_replay_preparation_actions"] == 1.0
    assert not any(key.startswith("task_success_rate") for key in root)
