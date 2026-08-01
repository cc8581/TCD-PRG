from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from tcd_prg.baselines.base import GlobalGraspPrediction
from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig
from tcd_prg.datasets.collate import _empty_global_grasps_like
from tcd_prg.datasets.task_oriented_clutter import _se3_diverse_rows
from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset
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
                "center_offset_m", "scene_confidence_logit", "intrinsic_confidence_logit"):
        assert torch.equal(output_first[key], output_second[key]), key


def test_scene_only_forward_is_strictly_invariant_to_instance_ids(tiny_batch) -> None:
    model = _small_model()
    first = {key: value.clone() for key, value in tiny_batch.items()}
    second = {key: value.clone() for key, value in tiny_batch.items()}
    second["instance_id"][:] = -1
    with torch.no_grad():
        output_first = model(first)["global_grasp"]
        output_second = model(second)["global_grasp"]
    for key, value in output_first.items():
        assert torch.equal(value, output_second[key]), key


def test_scene_only_topk_and_nms_do_not_use_instance_ids(tiny_batch) -> None:
    model = _small_model()
    generator = DenseCandidateGenerator(model.config)
    first = {key: value.clone() for key, value in tiny_batch.items()}
    second = {key: value.clone() for key, value in tiny_batch.items()}
    second["instance_id"][:] = -1
    with torch.no_grad():
        output = model(first)
        decoded_first = generator.global_predictions(first, output, topk=8)[0]
        decoded_second = generator.global_predictions(second, output, topk=8)[0]
    for key in ("point_index", "mode_index", "pose_world", "raw_score", "scene_score"):
        assert torch.equal(decoded_first[key], decoded_second[key]), key
    assert (decoded_second["object"] == -1).all()


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
        "center_offset_m": torch.zeros(1, 1, 2, 3, requires_grad=True),
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
        "center_offset_target_m": torch.tensor([[[[0.01, 0.0, 0.0], [float("nan")] * 3]]]),
        "center_offset_valid": torch.tensor([[[True, False]]]),
        "scene_target": torch.zeros(1, 1, 2), "scene_valid": torch.zeros(1, 1, 2, dtype=torch.bool),
        "intrinsic_target": torch.tensor([[[1.0, 0.0]]]),
        "intrinsic_valid": torch.tensor([[[True, False]]]),
    }
    losses = GlobalGraspLoss()(output, labels)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert output["scene_confidence_logit"].grad is None or not output["scene_confidence_logit"].grad.any()


def test_unmatched_global_modes_receive_no_grasp_confidence_gradient() -> None:
    output = {
        "contact_logits": torch.zeros(1, 1, requires_grad=True),
        "approach_direction": torch.tensor(
            [[[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]], requires_grad=True
        ),
        "rotation_logits": torch.zeros(1, 1, 2, 4, requires_grad=True),
        "width_m": torch.full((1, 1, 2), 0.05, requires_grad=True),
        "center_offset_m": torch.zeros(1, 1, 2, 3, requires_grad=True),
        "scene_confidence_logit": torch.full((1, 1, 2), 2.0, requires_grad=True),
        "intrinsic_confidence_logit": torch.full((1, 1, 2), 2.0, requires_grad=True),
    }
    labels = {
        "contact_target": torch.ones(1, 1),
        "contact_valid": torch.ones(1, 1, dtype=torch.bool),
        "mode_valid": torch.tensor([[[True, False]]]),
        "geometry_valid": torch.tensor([[[True, False]]]),
        "approach_target": torch.tensor([[[[0.0, 0.0, 1.0], [float("nan")] * 3]]]),
        "rotation_bin": torch.tensor([[[0, -1]]]),
        "width_target_m": torch.tensor([[[0.05, float("nan")]]]),
        "width_valid": torch.tensor([[[True, False]]]),
        "center_offset_target_m": torch.tensor([[[[0.0, 0.0, 0.0], [float("nan")] * 3]]]),
        "center_offset_valid": torch.tensor([[[True, False]]]),
        "scene_target": torch.tensor([[[1.0, 0.0]]]),
        "scene_valid": torch.tensor([[[True, False]]]),
        "intrinsic_target": torch.tensor([[[1.0, 0.0]]]),
        "intrinsic_valid": torch.tensor([[[True, False]]]),
    }
    losses = GlobalGraspLoss()(output, labels)
    (losses["global_intrinsic_confidence"] + losses["global_scene_confidence"]).backward()
    assert output["intrinsic_confidence_logit"].grad[0, 0, 1] > 0
    assert output["scene_confidence_logit"].grad[0, 0, 1] > 0


def test_global_prediction_domain_uses_present_not_active() -> None:
    config = ModelConfig(
        feature_dim=8, global_grasp_modes_per_point=2, candidate_topk=4,
        global_grasp_input_mode="instance_assisted",
    )
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
        "center_offset_m": torch.zeros(1, 2, 2, 3),
    }
    decoded = generator.global_predictions(batch, {"global_grasp": head})[0]
    assert set(decoded["object"].tolist()) == {0, 1}


def test_global_center_offset_reconstructs_grasp_translation() -> None:
    config = ModelConfig(
        feature_dim=8, global_grasp_modes_per_point=1, candidate_topk=1,
        global_grasp_input_mode="scene_only",
    )
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "instance_id": torch.tensor([[0]]),
        "point_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_present": torch.ones(1, 1, dtype=torch.bool),
    }
    head = {
        "contact_logits": torch.tensor([[10.0]]),
        "scene_confidence_logit": torch.tensor([[[10.0]]]),
        "intrinsic_confidence_logit": torch.tensor([[[10.0]]]),
        "rotation_logits": torch.zeros(1, 1, 1, 12),
        "approach_direction": torch.tensor([[[[0.0, 0.0, 1.0]]]]),
        "width_m": torch.full((1, 1, 1), 0.05),
        "center_offset_m": torch.tensor([[[[0.01, -0.02, 0.03]]]]),
    }
    decoded = generator.global_predictions(batch, {"global_grasp": head})[0]
    assert torch.allclose(decoded["pose_world"][0, :3], torch.tensor([0.11, 0.18, 0.33]))


def test_global_prediction_applies_se3_nms_to_duplicate_modes() -> None:
    config = ModelConfig(
        feature_dim=8, global_grasp_modes_per_point=2, candidate_topk=2,
        global_grasp_input_mode="scene_only",
    )
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.zeros(1, 1, 3), "instance_id": torch.tensor([[0]]),
        "point_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_present": torch.ones(1, 1, dtype=torch.bool),
    }
    head = {
        "contact_logits": torch.tensor([[10.0]]),
        "scene_confidence_logit": torch.full((1, 1, 2), 10.0),
        "intrinsic_confidence_logit": torch.tensor([[[10.0, 9.0]]]),
        "rotation_logits": torch.zeros(1, 1, 2, 12),
        "approach_direction": torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]]),
        "width_m": torch.full((1, 1, 2), 0.05),
        "center_offset_m": torch.zeros(1, 1, 2, 3),
    }
    decoded = generator.global_predictions(batch, {"global_grasp": head})[0]
    assert len(decoded["raw_score"]) == 1


def test_global_sampling_is_pose_diverse_and_respects_quota() -> None:
    transform = np.repeat(np.eye(4, dtype=np.float32)[None], 10, axis=0)
    transform[:, 0, 3] = np.linspace(0.0, 0.09, 10)
    library = SimpleNamespace(
        transform_object=transform,
        ag_width_m=np.linspace(0.03, 0.08, 10, dtype=np.float32),
        quality=np.ones(10, np.float32),
    )
    selected = _se3_diverse_rows(library, np.arange(10), 4)
    assert len(selected) == 4
    assert len(np.unique(selected)) == 4
    assert np.ptp(transform[selected, 0, 3]) >= 0.06


def test_global_supervision_has_one_representative_per_scene_state() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.flags: list[bool] = []

        @staticmethod
        def iter_action_groups(split=None):
            del split
            return iter(((1, 2, 0, 10), (1, 2, 1, 11), (1, 3, 0, 12)))

        def load_sample(self, *args, include_global_grasps=True):
            del args
            self.flags.append(include_global_grasps)
            return include_global_grasps

    adapter = Adapter()
    dataset = ActionStateGroupDataset(adapter, include_strata=False)  # type: ignore[arg-type]
    assert [dataset[index] for index in range(len(dataset))] == [True, False, True]


def test_global_evaluator_accepts_parallel_jaw_symmetric_pose() -> None:
    labels = GlobalGraspLabels(
        object_index=np.array([0]), source_grasp_index=np.array([1]),
        contact_point_world=np.zeros((1, 3), np.float32),
        grasp_pose_world=np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        approach_direction_world=np.array([[0, 0, 1]], np.float32), width_m=np.array([0.05], np.float32),
        intrinsic_stable=np.array([True]), scene_executable=np.array([1], np.int8),
        valid_mask=np.array([True]), anchor_visible_distance_m=np.array([0.0], np.float32),
        conversion_version="test",
    )
    prediction = GlobalGraspPrediction(
        0, np.zeros(3), np.array([0, 0, 0, 0, 0, 1, 0], np.float32),
        0.05, 1.0, 1.0, 1.0, True, "test",
    )
    metrics = GlobalGraspEvaluator().evaluate([prediction], labels, certified=False, topk=(1,))
    assert metrics["raw_recall@1"] == 1.0


def test_empty_global_grasp_placeholder_preserves_complete_contract() -> None:
    labels = GlobalGraspLabels(
        object_index=np.array([0]), source_grasp_index=np.array([1]),
        contact_point_world=np.zeros((1, 3), np.float32),
        grasp_pose_world=np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        approach_direction_world=np.array([[0, 0, 1]], np.float32),
        width_m=np.array([0.05], np.float32), intrinsic_stable=np.array([True]),
        scene_executable=np.array([1], np.int8), valid_mask=np.array([True]),
        anchor_visible_distance_m=np.array([0.0], np.float32),
        conversion_version="generic_grasp_v2",
    )
    empty = _empty_global_grasps_like(labels)
    empty.validate()
    assert empty.conversion_version == labels.conversion_version
    assert len(empty.object_index) == 0
