from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcd_prg.config import AblationConfig, GraspNetConfig, ModelConfig
from tcd_prg.constants import ActionType
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.push import PushHead
from tcd_prg.planners import DenseCandidateGenerator


pytestmark = pytest.mark.usefixtures("fake_graspnet")


def _config() -> ModelConfig:
    return ModelConfig(
        feature_dim=32,
        task_dim=16,
        num_categories=20,
        num_task_regions=20,
        activation_checkpointing=False,
    )


def test_fixed_seed_reproducibility(tiny_batch) -> None:
    torch.manual_seed(3)
    first = TCDPRGModel(_config()).eval()
    torch.manual_seed(3)
    second = TCDPRGModel(_config()).eval()
    with torch.no_grad():
        one = first(tiny_batch)["region"]["region_logits"]
        two = second(tiny_batch)["region"]["region_logits"]
    assert torch.equal(one, two)






def test_network_features_do_not_depend_on_simulation_object_pose(tiny_batch) -> None:
    model = TCDPRGModel(_config()).eval()
    changed = dict(tiny_batch)
    changed.pop("object_pose")
    with torch.no_grad():
        first = model(tiny_batch)["encoded"].point_features
        second = model(changed)["encoded"].point_features
    assert torch.equal(first, second)


def test_policy_heads_match_training_contract(tiny_batch) -> None:
    output = TCDPRGModel(_config()).eval()(tiny_batch)
    assert {
        "translation_world",
        "rotation_matrix",
        "width_m",
        "depth_m",
        "quality_logit",
        "graspnet_score",
        "attention_point_index",
        "object_logits",
        "valid",
        "task_valid_logit",
        "task_valid_probability",
        "graspnet_width_m",
        "target_grasp_valid",
        "target_crop_points",
    } <= set(output["task_grasp"])
    assert "object_logits" in output["global_grasp"]
    assert output["push"]["utility_delta"].shape[-1] == _config().num_direction_bins
    assert "approach_logits" not in output["push"]
    assert "risk_logits" not in output["push"]
    assert "pick_remove" not in output


def test_camera2_to_task_evaluator_to_dense_candidate_end_to_end(tiny_batch) -> None:
    config = _config()
    config.instance_objectness_threshold = 0.0
    config.target_prompt_min_support = 0.0
    config.target_prompt_min_margin = 0.0
    config.task_grasp_probability_threshold = 1e-6
    tiny_batch = dict(tiny_batch)
    tiny_batch["target_prompt_xyz"] = tiny_batch["xyz"][:, :1]
    tiny_batch["target_prompt_label"] = torch.ones(1, 1, dtype=torch.bool)
    tiny_batch["target_prompt_valid"] = torch.ones(1, 1, dtype=torch.bool)
    graspnet = GraspNetConfig(
        target_crop_probability=0.0,
        target_min_crop_points=1,
        target_proposals=8,
        global_proposals=8,
    )
    model = TCDPRGModel(config, graspnet_config=graspnet).eval()
    with torch.no_grad():
        output = model(tiny_batch)
        candidates = DenseCandidateGenerator(config).generate(
            model, tiny_batch, output
        )

    task = output["task_grasp"]
    assert task["target_grasp_valid"].all()
    assert task["valid"].any()
    assert torch.isfinite(task["task_valid_logit"]).all()
    assert torch.all((task["width_m"] >= 0.0) & (task["width_m"] <= 0.095))
    task_candidates = candidates["valid"] & (
        candidates["type"] == int(ActionType.TASK_GRASP)
    )
    assert task_candidates.any()
    assert torch.all(
        (candidates["width_m"][task_candidates] >= 0.0)
        & (candidates["width_m"][task_candidates] <= 0.095)
    )
