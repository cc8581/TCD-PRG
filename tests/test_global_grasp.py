from __future__ import annotations

import numpy as np
import torch

from tcd_prg.baselines.base import GlobalGraspPrediction
from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig
from tcd_prg.datasets.types import GlobalGraspLabels
from tcd_prg.evaluators.global_grasp import GlobalGraspEvaluator
from tcd_prg.losses.global_grasp import GlobalGraspLoss
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners.candidate_generator import DenseCandidateGenerator


def _small_model() -> TCDPRGModel:
    config = ModelConfig(
        feature_dim=16, task_dim=8, num_categories=8, num_task_regions=8,
        global_grasp_modes_per_point=3, global_grasp_input_mode="scene_only",
        activation_checkpointing=False,
    )
    return TCDPRGModel(
        config, AblationConfig(), GraphConfig(layers=1, heads=4),
        RouterConfig(layers=1, heads=4), BackboneConfig(attention_points=8),
    ).eval()


def test_global_branch_is_invariant_to_task_and_target(tiny_batch) -> None:
    model = _small_model()
    first = {key: value.clone() for key, value in tiny_batch.items()}
    second = {key: value.clone() for key, value in tiny_batch.items()}
    second["target_mask"] = second["instance_id"] == 2
    second["target_object"] = torch.tensor([2])
    second["task_category_id"] = torch.tensor([7])
    second["task_region_id"] = torch.tensor([6])
    with torch.no_grad():
        output_first = model(first)["global_grasp"]
        output_second = model(second)["global_grasp"]
    for key in ("contact_logits", "approach_direction", "rotation_logits", "width_m",
                "scene_confidence_logit", "intrinsic_confidence_logit"):
        assert torch.equal(output_first[key], output_second[key]), key


def test_global_branch_has_multiple_modes_per_point(tiny_batch) -> None:
    output = _small_model()(tiny_batch)["global_grasp"]
    assert output["approach_direction"].shape == (1, 24, 3, 3)
    assert output["rotation_logits"].shape == (1, 24, 3, 12)
    assert output["width_m"].shape == (1, 24, 3)


def test_unknown_scene_executability_is_ignored() -> None:
    output = {
        "contact_logits": torch.zeros(1, 1, requires_grad=True),
        "approach_direction": torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]], requires_grad=True),
        "rotation_logits": torch.zeros(1, 1, 2, 4, requires_grad=True),
        "width_m": torch.full((1, 1, 2), 0.05, requires_grad=True),
        "scene_confidence_logit": torch.zeros(1, 1, 2, requires_grad=True),
        "intrinsic_confidence_logit": torch.zeros(1, 1, 2, requires_grad=True),
    }
    labels = {
        "contact_target": torch.ones(1, 1), "contact_valid": torch.ones(1, 1, dtype=torch.bool),
        "mode_valid": torch.tensor([[[True, False]]]),
        "geometry_valid": torch.tensor([[[True, False]]]),
        "approach_target": torch.tensor([[[[0.0, 0.0, 1.0], [float("nan")] * 3]]]),
        "rotation_bin": torch.tensor([[[0, -1]]]),
        "width_target_m": torch.tensor([[[0.05, float("nan")]]]),
        "width_valid": torch.tensor([[[True, False]]]),
        "scene_target": torch.zeros(1, 1, 2), "scene_valid": torch.zeros(1, 1, 2, dtype=torch.bool),
        "intrinsic_target": torch.tensor([[[1.0, 0.0]]]),
        "intrinsic_valid": torch.tensor([[[True, False]]]),
    }
    losses = GlobalGraspLoss()(output, labels)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert output["scene_confidence_logit"].grad is None or not output["scene_confidence_logit"].grad.any()


def test_global_prediction_domain_uses_present_not_active() -> None:
    config = ModelConfig(feature_dim=8, global_grasp_modes_per_point=2, candidate_topk=4)
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.zeros(1, 2, 3), "instance_id": torch.tensor([[0, 1]]),
        "point_mask": torch.ones(1, 2, dtype=torch.bool),
        "object_mask": torch.ones(1, 2, dtype=torch.bool),
        "object_present": torch.ones(1, 2, dtype=torch.bool),
        "object_active": torch.tensor([[False, True]]),
    }
    head = {
        "contact_logits": torch.zeros(1, 2),
        "scene_confidence_logit": torch.zeros(1, 2, 2),
        "intrinsic_confidence_logit": torch.zeros(1, 2, 2),
        "rotation_logits": torch.zeros(1, 2, 2, 12),
        "approach_direction": torch.nn.functional.normalize(torch.ones(1, 2, 2, 3), dim=-1),
        "width_m": torch.full((1, 2, 2), 0.05),
    }
    decoded = generator.global_predictions(batch, {"global_grasp": head})[0]
    assert set(decoded["object"].tolist()) == {0, 1}


def test_global_evaluator_accepts_parallel_jaw_symmetric_pose() -> None:
    labels = GlobalGraspLabels(
        object_index=np.array([0]), source_grasp_index=np.array([1]),
        contact_point_world=np.zeros((1, 3), np.float32),
        grasp_pose_world=np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        approach_direction_world=np.array([[0, 0, 1]], np.float32), width_m=np.array([0.05], np.float32),
        intrinsic_stable=np.array([True]), scene_executable=np.array([1], np.int8),
        valid_mask=np.array([True]), conversion_version="test",
    )
    prediction = GlobalGraspPrediction(
        0, np.zeros(3), np.array([0, 0, 0, 0, 0, 1, 0], np.float32),
        0.05, 1.0, 1.0, True, "test",
    )
    metrics = GlobalGraspEvaluator().evaluate([prediction], labels, certified=False, topk=(1,))
    assert metrics["raw_recall@1"] == 1.0
