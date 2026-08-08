from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcd_prg.diagnostics.grasp import grasp_diagnostic_record
from tcd_prg.geometry.grasp_nms import grasp_nms


def _cfg():
    return SimpleNamespace(
        task_translation_threshold_m=0.010,
        task_rotation_threshold_deg=12.0,
        task_width_threshold_m=0.005,
        global_translation_threshold_m=0.010,
        global_rotation_threshold_deg=15.0,
        global_width_threshold_m=0.005,
        task_nms_translation_m=0.010,
        task_nms_rotation_deg=12.0,
        task_nms_width_m=0.005,
        global_nms_translation_m=0.010,
        global_nms_rotation_deg=15.0,
        global_nms_width_m=0.005,
    )


def _labels(global_mode: bool = False):
    labels = {
        "sample_valid": torch.tensor([True]),
        "target_valid": torch.tensor([[True]]),
        "translation_world": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "width_m": torch.tensor([[0.04]]),
    }
    if global_mode:
        labels["object_index"] = torch.tensor([[1]])
    return labels


def _output(global_mode: bool = False):
    output = {
        "translation_world": torch.tensor([[[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]]),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "width_m": torch.tensor([[0.04, 0.08]]),
        "quality_logit": torch.tensor([[20.0, -20.0]]),
    }
    if global_mode:
        output["object_logits"] = torch.tensor([[[-20.0, 20.0, -20.0], [20.0, -20.0, -20.0]]])
    return output


def test_gt_as_prediction_is_exact_task_self_consistency():
    values = grasp_diagnostic_record(
        _output(False),
        _labels(False),
        0,
        prefix="task",
        evaluation_config=_cfg(),
        topk=(1, 5, 10, 64),
    )
    assert values["task_quality_hit_at_1"] == 1.0
    assert values["task_oracle_hit_at_64"] == 1.0
    assert values["task_top1_translation_error_m"] == pytest.approx(0.0, abs=1e-8)
    assert values["task_top1_rotation_error_deg"] == pytest.approx(0.0, abs=1e-6)
    assert values["task_top1_width_error_m"] == pytest.approx(0.0, abs=1e-8)


def test_gt_as_prediction_is_exact_global_self_consistency():
    values = grasp_diagnostic_record(
        _output(True),
        _labels(True),
        0,
        prefix="global",
        evaluation_config=_cfg(),
        topk=(1, 64),
    )
    assert values["global_quality_hit_at_1"] == 1.0
    assert values["global_oracle_hit_at_64"] == 1.0
    assert values["global_top1_object_correct"] == 1.0


def test_parallel_jaw_180_degree_symmetry_is_zero_error_and_hit():
    output = _output(False)
    symmetry = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    output["rotation_matrix"][0, 0] = output["rotation_matrix"][0, 0] @ symmetry
    values = grasp_diagnostic_record(
        output,
        _labels(False),
        0,
        prefix="task",
        evaluation_config=_cfg(),
        topk=(1, 64),
    )
    assert values["task_top1_rotation_error_deg"] == pytest.approx(0.0, abs=1e-5)
    assert values["task_quality_hit_at_1"] == 1.0


def test_nms_suppresses_only_same_object_close_grasps():
    t = torch.tensor([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.005, 0.0, 0.0]])
    r = torch.eye(3).repeat(3, 1, 1)
    w = torch.tensor([0.04, 0.04, 0.04])
    score = torch.tensor([0.9, 0.8, 0.7])
    obj = torch.tensor([0, 0, 1])
    selected = grasp_nms(
        t,
        r,
        w,
        score,
        translation_threshold_m=0.01,
        rotation_threshold_deg=15.0,
        width_threshold_m=0.005,
        object_index=obj,
    )
    assert selected.tolist() == [0, 2]
