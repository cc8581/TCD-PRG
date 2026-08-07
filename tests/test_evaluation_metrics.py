from __future__ import annotations

import numpy as np
import pytest

from tcd_prg.evaluators.evaluator import Evaluator
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


def test_evaluator_ignores_nonstandard_records() -> None:
    evaluator = Evaluator(bootstrap_samples=0)
    evaluator.add(
        scene_id=1,
        selected_candidate_success=1.0,
        _binary_verifier_overall=([0.9, 0.2], [True, False]),
        _binary_standard_verifier_overall=([0.9, 0.2], [True, False]),
        _confusion_standard_verifier_overall=(1, 0, 0, 1),
    )
    summary = evaluator.summarize()["metrics"]
    assert "selected_candidate_success" not in summary
    assert "verifier_overall_auroc" not in summary
    assert summary["standard_verifier_overall_auroc"]["mean"] == pytest.approx(1.0)
    assert summary["standard_verifier_overall_f1"]["mean"] == pytest.approx(1.0)
    assert "standard_verifier_overall_iou" not in summary
    assert "standard_verifier_overall_dice" not in summary
