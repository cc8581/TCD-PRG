from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcd_prg.config import AblationConfig, ModelConfig
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.common import ActionCandidateEncoder
from tcd_prg.models.dependency_graph.hgt import derive_dependency_masks
from tcd_prg.models.grasp_verifier import GripperSceneTaskVerifier
from tcd_prg.models.policy import MaskedHierarchicalCandidateRouter
from tcd_prg.models.push import PushHead


def _config() -> ModelConfig:
    return ModelConfig(
        feature_dim=32,
        task_dim=16,
        num_categories=20,
        num_task_regions=20,
        activation_checkpointing=False,
    )


def test_point_gripper_joint_input_shapes() -> None:
    verifier = GripperSceneTaskVerifier(32, 32)
    output = verifier(
        torch.randn(1, 2, 12, 3),
        torch.randn(1, 2, 8, 3),
        torch.randn(1, 2, 12, 32),
        torch.ones(1, 2, 12, dtype=torch.bool),
        torch.rand(1, 2, 12),
        torch.randn(1, 32),
    )
    assert all(value.shape == (1, 2) for value in output.values())


def test_cached_verifier_only_runs_valid_candidates() -> None:
    config = _config()
    config.verifier_candidate_micro_batch = 1
    model = TCDPRGModel(config).eval()
    b, k, n, local, gripper = 1, 5, 8, 4, 3
    candidate_valid = torch.tensor([[True, False, False, True, False]])
    seen_candidates: list[int] = []

    def record_candidates(_module, inputs) -> None:
        seen_candidates.append(inputs[0].shape[0] * inputs[0].shape[1])

    handle = model.verifier.register_forward_pre_hook(record_candidates)
    output = model.verify_cached(
        {"target_mask": torch.ones(b, n, dtype=torch.bool)},
        {
            "encoded": SimpleNamespace(
                point_features=torch.randn(b, n, config.feature_dim),
                task_token=torch.randn(b, config.feature_dim),
                target_probability=torch.ones(b, n),
            ),
            "region": {"region_probability": torch.rand(b, n)},
        },
        {
            "candidate_valid": candidate_valid,
            "scene_point_index": torch.zeros(b, k, local, dtype=torch.long),
            "scene_xyz_grasp": torch.randn(b, k, local, 3),
            "scene_valid": torch.ones(b, k, local, dtype=torch.bool),
            "gripper_xyz_grasp": torch.randn(b, k, gripper, 3),
            "gripper_valid": torch.ones(b, k, gripper, dtype=torch.bool),
        },
    )
    handle.remove()
    assert sum(seen_candidates) == int(candidate_valid.sum())
    assert output["overall_logit"].shape == (b, k)
    assert torch.equal(
        output["overall_logit"][~candidate_valid],
        torch.full((k - int(candidate_valid.sum()),), -30.0),
    )
    output["overall_logit"][candidate_valid].sum().backward()
    assert model.verifier.overall.weight.grad is not None


def test_push_direction_sparse_points_are_prediction_only() -> None:
    torch.manual_seed(9)
    b, n, objects, dim = 1, 8, 2, 16
    head = PushHead(
        dim=dim,
        direction_bins=4,
        direction_dim=8,
        direction_layers=1,
        direction_heads=2,
        direction_contact_topk=2,
    ).eval()
    inputs = (
        torch.randn(b, n, dim),
        torch.randn(b, n, 3),
        torch.arange(n)[None] % objects,
        torch.ones(b, n, dtype=torch.bool),
        torch.randn(b, objects, dim),
        torch.ones(b, objects, dtype=torch.bool),
        torch.randn(b, dim),
        torch.randn(b, dim),
        torch.randn(b, objects, dim),
        torch.tensor([3]),
    )
    output = head(*inputs)
    assert output["direction_logits"].shape == (b, n, 4)
    assert output["direction_residual"].shape == (b, n, 4, 2)
    assert output["direction_point_mask"].sum().item() <= objects * 2
    assert (output["direction_logits"][~output["direction_point_mask"]] == -30.0).all()


def test_router_never_selects_invalid_candidate() -> None:
    router = MaskedHierarchicalCandidateRouter(32, layers=1)
    valid = torch.tensor([[False, True, False]])
    output = router(
        torch.randn(1, 32),
        torch.randn(1, 32),
        torch.randn(1, 2, 32),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 3, 32),
        torch.tensor([[0, 2, 1]]),
        torch.tensor([[0, 1, 0]]),
        valid,
        torch.tensor([5]),
    )
    selected = router.select(output, torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 0]]))
    assert selected.item() == 1


def test_router_all_invalid_is_finite_and_selects_nothing() -> None:
    router = MaskedHierarchicalCandidateRouter(32, layers=1)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    candidate_type = torch.tensor([[-1, -1]])
    candidate_object = torch.tensor([[-1, -1]])
    output = router(
        torch.randn(1, 32),
        torch.randn(1, 32),
        torch.randn(1, 2, 32),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 2, 32),
        candidate_type,
        candidate_object,
        valid,
        torch.tensor([5]),
    )
    assert torch.isfinite(output.candidate_logits).all()
    assert router.select(output, candidate_type, candidate_object).item() == -1


def test_dependency_closure_returns_only_topmost_actionable_object() -> None:
    physical = torch.full((1, 3, 3, 5), -20.0)
    task = torch.full((1, 3, 3), -20.0)
    task[0, 0, 0] = 20.0  # object 0 directly blocks TASK_GRASP
    physical[0, 0, 1, 2] = 20.0  # object 1 is prerequisite of object 0
    physical[0, 1, 2, 2] = 20.0  # object 2 is prerequisite of object 1
    direct, indirect, dependency, actionable = derive_dependency_masks(
        physical, task, torch.ones(1, 3, dtype=torch.bool)
    )
    assert direct.tolist() == [[True, False, False]]
    assert indirect.tolist() == [[False, True, True]]
    assert dependency.all()
    assert actionable.tolist() == [[False, False, True]]


def test_physical_neighborhood_does_not_create_direct_task_blocker() -> None:
    physical = torch.full((1, 2, 2, 5), -20.0)
    physical[0, 1, 0, 0] = 20.0
    task = torch.full((1, 2, 3), -20.0)
    direct, _, _, _ = derive_dependency_masks(
        physical, task, torch.ones(1, 2, dtype=torch.bool), target_object=torch.tensor([0])
    )
    assert not direct.any()


@pytest.mark.parametrize(
    "ablation",
    [
        AblationConfig(use_task_region_condition=False),
        AblationConfig(use_dependency_graph=False),
        AblationConfig(use_indirect_dependency_reasoning=False),
        AblationConfig(use_gripper_scene_verifier=False),
        AblationConfig(use_push_potential=False),
        AblationConfig(router_type="fixed_priority"),
        AblationConfig(router_type="flat_candidate_classifier"),
    ],
)
def test_every_ablation_can_forward(tiny_batch, ablation) -> None:
    model = TCDPRGModel(_config(), ablation)
    result = model(tiny_batch)
    assert result["region"]["region_logits"].shape == (1, 24)


def test_fixed_seed_reproducibility(tiny_batch) -> None:
    torch.manual_seed(3)
    first = TCDPRGModel(_config()).eval()
    torch.manual_seed(3)
    second = TCDPRGModel(_config()).eval()
    with torch.no_grad():
        one = first(tiny_batch)["region"]["region_logits"]
        two = second(tiny_batch)["region"]["region_logits"]
    assert torch.equal(one, two)


def test_model_routes_cached_generated_candidates_without_teacher_axis(tiny_batch) -> None:
    model = TCDPRGModel(_config()).eval()
    generated = {
        "type": torch.tensor([[0, 2]]),
        "object": torch.tensor([[0, 1]]),
        "contact_world": torch.tensor([[[0.0, 0.0, 0.0], [float("nan")] * 3]]),
        "direction_world": torch.tensor([[[1.0, 0.0, 0.0], [float("nan")] * 3]]),
        "pose_world": torch.tensor([[[float("nan")] * 7, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]),
        "destination_world": torch.full((1, 2, 3), float("nan")),
        "width_m": torch.tensor([[float("nan"), 0.05]]),
        "evidence": torch.zeros(1, 2, 7),
        "valid": torch.ones(1, 2, dtype=torch.bool),
        "label_status": torch.tensor([[0, 1]], dtype=torch.int8),
        "policy_success": torch.tensor([[False, True]]),
    }
    batch = {**tiny_batch, "generated_policy_candidates": generated}
    output = model(batch)
    assert output["generated_router"].candidate_logits.shape == (1, 2)
    assert torch.isfinite(output["generated_router"].candidate_logits).all()

    class ForbiddenHead(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("full geometry head ran during generated-policy forward")

    model.region_head = ForbiddenHead()
    fast = model(batch, forward_mode="generated_policy")
    assert fast["generated_router"].candidate_logits.shape == (1, 2)
    evaluation_batch = {
        **batch,
        "candidate_mask": torch.zeros(1, 1, dtype=torch.bool),
        "policy_success_mask": torch.zeros(1, 1, dtype=torch.bool),
        "evaluation_status": torch.full((1, 1), -1, dtype=torch.long),
        "action_type": torch.full((1, 1), -1, dtype=torch.long),
        "acted_object": torch.full((1, 1), -1, dtype=torch.long),
        "samples": [
            SimpleNamespace(
                observation=SimpleNamespace(
                    scene_id=1,
                    state_id=0,
                    task_index=0,
                    target_object=1,
                    object_category_id=[0, 1, 2],
                    task_region_id=2,
                ),
                state_labels=SimpleNamespace(
                    sequence_depth=0,
                    target_visible_ratio=1.0,
                    graspable=True,
                ),
            )
        ],
    }
    evaluator = OfflineModelEvaluator(_config(), bootstrap_samples=0)
    evaluator.update(evaluation_batch, fast)
    metrics = evaluator.summarize()["metrics"]
    assert "generated_positive_coverage" not in metrics
    assert "generated_effective_policy_row" not in metrics


def test_candidate_encoder_does_not_consume_pick_remove_destination() -> None:
    encoder = ActionCandidateEncoder(8).eval()
    object_tokens = torch.randn(1, 2, 8)
    common = {
        "object_tokens": object_tokens,
        "action_type": torch.tensor([[1]]),
        "acted_object": torch.tensor([[0]]),
        "contact_world": torch.full((1, 1, 3), float("nan")),
        "direction_world": torch.full((1, 1, 3), float("nan")),
        "pose_world": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]),
        "parameter_valid": torch.ones(1, 1, 5, dtype=torch.bool),
        "task_token": torch.randn(1, 8),
    }
    first = encoder(destination_world=torch.zeros(1, 1, 3), **common)
    second = encoder(destination_world=torch.full((1, 1, 3), 100.0), **common)
    assert torch.equal(first, second)


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
    assert set(output["task_grasp"]) == {
        "translation_world",
        "rotation_matrix",
        "rotation_6d",
        "width_raw",
        "width_m",
        "quality_logit",
        "attention_point_index",
        "point_attention",
    }
    assert "object_logits" in output["global_grasp"]
    assert output["push"]["utility_delta"].shape[-1] == _config().num_direction_bins
    assert "approach_logits" not in output["push"]
    assert "risk_logits" not in output["push"]
    assert "pick_remove" not in output


def test_default_task_query_capacity_has_certification_margin() -> None:
    config = ModelConfig()
    assert config.task_grasp_candidates == 64
    assert config.task_grasp_candidates >= 3 * config.default_required_grasp_count
