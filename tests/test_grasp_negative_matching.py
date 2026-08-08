from __future__ import annotations

import pytest
import torch

from tcd_prg.losses.proposal import CompleteGraspSetLoss


def _base_output(x: float, *, global_mode: bool = False, predicted_object: int = 0):
    output = {
        "translation_world": torch.tensor([[[x, 0.0, 0.0]]], dtype=torch.float32),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "width_m": torch.tensor([[0.04]], dtype=torch.float32),
        "quality_logit": torch.tensor([[0.0]], dtype=torch.float32),
    }
    if global_mode:
        logits = torch.full((1, 1, 2), -10.0)
        logits[0, 0, predicted_object] = 10.0
        output["object_logits"] = logits
    return output


def _negative_labels(*, global_mode: bool = False, negative_object: int = 0):
    labels = {
        "sample_valid": torch.tensor([True]),
        "target_valid": torch.tensor([[False]]),
        "translation_world": torch.zeros((1, 1, 3)),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "width_m": torch.tensor([[0.04]]),
        "quality_target": torch.zeros((1, 1)),
        "quality_valid": torch.zeros((1, 1), dtype=torch.bool),
        "unmatched_quality_valid": torch.tensor([False]),
        "negative_valid": torch.tensor([[True]]),
        "negative_translation_world": torch.zeros((1, 1, 3)),
        "negative_rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "negative_width_m": torch.tensor([[0.04]]),
    }
    if global_mode:
        labels["object_index"] = torch.zeros((1, 1), dtype=torch.long)
        labels["negative_object_index"] = torch.tensor([[negative_object]], dtype=torch.long)
    return labels


def test_close_negative_hungarian_pair_is_accepted():
    loss = CompleteGraspSetLoss(
        negative_translation_m=0.01,
        negative_rotation_deg=12.0,
        negative_width_m=0.005,
    )
    values = loss(_base_output(0.005), _negative_labels())
    assert float(values["grasp_matched_negative_queries"]) == pytest.approx(1.0)
    assert float(values["grasp_quality_valid_queries"]) == pytest.approx(1.0)


def test_far_negative_hungarian_pair_remains_unknown():
    loss = CompleteGraspSetLoss(
        negative_translation_m=0.01,
        negative_rotation_deg=12.0,
        negative_width_m=0.005,
    )
    values = loss(_base_output(0.08), _negative_labels())
    assert float(values["grasp_matched_negative_queries"]) == pytest.approx(0.0)
    assert float(values["grasp_quality_valid_queries"]) == pytest.approx(0.0)


def test_global_wrong_object_negative_pair_remains_unknown():
    loss = CompleteGraspSetLoss(
        negative_translation_m=0.01,
        negative_rotation_deg=15.0,
        negative_width_m=0.005,
    )
    values = loss(
        _base_output(0.005, global_mode=True, predicted_object=1),
        _negative_labels(global_mode=True, negative_object=0),
    )
    assert float(values["grasp_matched_negative_queries"]) == pytest.approx(0.0)
    assert float(values["grasp_quality_valid_queries"]) == pytest.approx(0.0)
