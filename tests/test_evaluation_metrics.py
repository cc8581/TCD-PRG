from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tcd_prg.evaluators.evaluator import Evaluator
from tcd_prg.evaluators.offline import OfflineModelEvaluator
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


def test_offline_evaluator_reports_target_and_region_segmentation_metrics() -> None:
    observation = SimpleNamespace(
        scene_id=2500,
        state_id=0,
        task_index=0,
        object_category_id=np.asarray([3]),
        target_object=0,
        task_region_id=1,
    )
    sample = SimpleNamespace(
        observation=observation,
        state_labels=SimpleNamespace(sequence_depth=1, target_visible_ratio=1.0),
    )
    batch = {
        "xyz": torch.zeros((1, 4, 3)),
        "samples": [sample],
        "point_mask": torch.tensor([[True, True, True, True]]),
        "target_mask": torch.tensor([[True, True, False, False]]),
        "region_valid": torch.tensor([[True, True, True, True]]),
        "region_target": torch.tensor([[True, True, False, False]]),
    }
    output = {
        "encoded": SimpleNamespace(
            target_instance_probability=torch.tensor([[0.9, 0.2, 0.8, 0.1]])
        ),
        "region": {"region_probability": torch.tensor([[0.9, 0.2, 0.8, 0.1]])},
    }
    evaluator = OfflineModelEvaluator(model_config=None, bootstrap_samples=0)
    evaluator.update(batch, output)
    metrics = evaluator.summarize()["metrics"]

    assert metrics["standard_target_iou"]["mean"] == pytest.approx(1 / 3)
    assert metrics["standard_target_precision"]["mean"] == pytest.approx(0.5)
    assert metrics["standard_target_recall"]["mean"] == pytest.approx(0.5)
    assert metrics["standard_region_foreground_iou"]["mean"] == pytest.approx(1 / 3)
    assert metrics["standard_region_background_iou"]["mean"] == pytest.approx(1 / 3)
    assert metrics["standard_region_miou"]["mean"] == pytest.approx(1 / 3)
    assert metrics["standard_region_foreground_precision"]["mean"] == pytest.approx(0.5)
    assert metrics["standard_region_foreground_recall"]["mean"] == pytest.approx(0.5)
