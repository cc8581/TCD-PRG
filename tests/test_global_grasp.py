from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tcd_prg.baselines.base import GlobalGraspPrediction
from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig
from tcd_prg.datasets.collate import _empty_global_grasps_like
from tcd_prg.datasets.task_oriented_clutter import _se3_diverse_rows
from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset
from tcd_prg.datasets.types import GlobalGraspLabels
from tcd_prg.evaluators.global_grasp import GlobalGraspEvaluator
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners.candidate_generator import DenseCandidateGenerator


def _small_model() -> TCDPRGModel:
    config = ModelConfig(
        feature_dim=16, task_dim=8, num_categories=8, num_task_regions=8,
        global_grasp_candidates=6, task_grasp_candidates=4,
        global_grasp_input_mode="scene_only", activation_checkpointing=False,
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
        one = model(first)["global_grasp"]
        two = model(second)["global_grasp"]
    for key in ("translation_world", "rotation_matrix", "width_m", "quality_logit"):
        assert torch.equal(one[key], two[key]), key


def test_scene_only_forward_is_strictly_invariant_to_instance_ids(tiny_batch) -> None:
    model = _small_model()
    first = {key: value.clone() for key, value in tiny_batch.items()}
    second = {key: value.clone() for key, value in tiny_batch.items()}
    second["instance_id"][:] = -1
    with torch.no_grad():
        one = model(first)["global_grasp"]
        two = model(second)["global_grasp"]
    for key, value in one.items():
        assert torch.equal(value, two[key]), key


def test_complete_global_branch_outputs_fixed_grasp_set(tiny_batch) -> None:
    output = _small_model()(tiny_batch)["global_grasp"]
    assert output["translation_world"].shape == (1, 6, 3)
    assert output["rotation_matrix"].shape == (1, 6, 3, 3)
    assert output["width_m"].shape == (1, 6)
    assert output["quality_logit"].shape == (1, 6)
    determinant = torch.det(output["rotation_matrix"])
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=1e-5)


def test_global_prediction_uses_complete_pose_and_se3_nms() -> None:
    config = ModelConfig(feature_dim=8, global_grasp_candidates=2, candidate_topk=2)
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.zeros(1, 1, 3), "instance_id": torch.tensor([[0]]),
        "point_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_present": torch.ones(1, 1, dtype=torch.bool),
    }
    rotation = torch.eye(3).expand(1, 2, 3, 3).clone()
    head = {
        "translation_world": torch.tensor([[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]]),
        "rotation_matrix": rotation,
        "width_m": torch.full((1, 2), 0.05),
        "quality_logit": torch.tensor([[10.0, 9.0]]),
    }
    decoded = generator.global_predictions(batch, {"global_grasp": head})[0]
    assert len(decoded["raw_score"]) == 1
    assert torch.allclose(decoded["pose_world"][0, :3], torch.tensor([0.1, 0.2, 0.3]))


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


def _labels() -> GlobalGraspLabels:
    return GlobalGraspLabels(
        object_index=np.array([0]), source_grasp_index=np.array([1]),
        contact_point_world=np.zeros((1, 3), np.float32),
        grasp_pose_world=np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        approach_direction_world=np.array([[0, 0, 1]], np.float32),
        width_m=np.array([0.05], np.float32), intrinsic_stable=np.array([True]),
        scene_executable=np.array([1], np.int8), valid_mask=np.array([True]),
        anchor_visible_distance_m=np.array([0.0], np.float32), conversion_version="test",
    )


def test_global_evaluator_accepts_parallel_jaw_symmetric_pose() -> None:
    prediction = GlobalGraspPrediction(
        0, np.zeros(3), np.array([0, 0, 0, 0, 0, 1, 0], np.float32),
        0.05, 1.0, 1.0, 1.0, True, "test",
    )
    assert GlobalGraspEvaluator().evaluate([prediction], _labels(), certified=False, topk=(1,))[
        "raw_recall@1"
    ] == 1.0


def test_empty_global_grasp_placeholder_preserves_complete_contract() -> None:
    empty = _empty_global_grasps_like(_labels())
    empty.validate()
    assert empty.conversion_version == "test"
    assert len(empty.object_index) == 0
