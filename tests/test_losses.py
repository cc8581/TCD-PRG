import torch

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.losses.labels import (
    _pack_grasp_set,
    build_global_grasp_labels,
    build_grasp_proposal_labels,
    build_push_supervision,
)
from tcd_prg.losses.masked import multi_positive_listwise_loss, safe_smooth_l1
from tcd_prg.losses.objective import TCDPRGObjective
from tcd_prg.losses.proposal import CompleteGraspSetLoss
from tcd_prg.losses.total import MultiTaskLoss


def test_nan_loss_is_masked() -> None:
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.5, float("nan")])
    loss = safe_smooth_l1(prediction, target, torch.tensor([True, False]))
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad[1] == 0


def test_multiple_correct_action_set_loss() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[False, True, True]])
    valid = torch.ones_like(positive)
    loss = multi_positive_listwise_loss(logits, positive, valid)
    assert 0 <= loss < 1
    loss.backward()


def test_unknown_candidate_excluded_from_listwise_denominator() -> None:
    logits = torch.tensor([[0.0, 1.0, 100.0]])
    positive = torch.tensor([[False, True, False]])
    evaluated = torch.tensor([[True, True, False]])
    loss = multi_positive_listwise_loss(logits, positive, evaluated)
    assert loss < 1


def test_sequence_topology_mask_only_masks_order_loss() -> None:
    from tcd_prg.losses.masked import safe_bce_with_logits

    logits = torch.tensor([[100.0]], requires_grad=True)
    topology_valid = torch.tensor([False])[:, None]
    topology = safe_bce_with_logits(logits, torch.zeros_like(logits), topology_valid)
    action = safe_bce_with_logits(logits, torch.zeros_like(logits), torch.ones_like(topology_valid))
    assert topology == 0
    assert action > 0


def test_multitask_total_uses_family_subtotal_only() -> None:
    loss = MultiTaskLoss(
        DatasetCapabilities(has_task_grasps=True), AblationConfig(), {"task_grasp": 3.0}
    )
    total, logged = loss({
        "task_grasp": {"loss": torch.tensor(2.0), "proposal_detail": torch.tensor(100.0)}
    })
    assert total.item() == 6.0
    assert logged["proposal_detail"].item() == 100.0


def test_family_subtotal_is_weighted_mean_of_active_children_only() -> None:
    objective = TCDPRGObjective(
        DatasetCapabilities(has_task_grasps=True), ModelConfig(), AblationConfig(),
        LossConfig(internal={"first": 2.0, "second": 1.0, "inactive": 100.0}),
    )
    first = torch.tensor(2.0, requires_grad=True)
    second = torch.tensor(4.0, requires_grad=True)
    inactive = torch.tensor(100.0, requires_grad=True)
    family = objective._subtotal(
        {"first": first, "second": second, "inactive": inactive},
        {"first": True, "second": True, "inactive": False},
    )
    assert torch.allclose(family["loss"], torch.tensor(8.0 / 3.0))
    family["loss"].backward()
    assert inactive.grad == 0


def test_exactly_eleven_paper_level_objectives() -> None:
    assert len(TCDPRGObjective.MODULE_OBJECTIVES) == 11
    assert tuple(MultiTaskLoss.DEFAULT_WEIGHTS) == TCDPRGObjective.MODULE_OBJECTIVES
    capabilities = DatasetCapabilities(
        has_task_regions=True, has_task_grasps=True, has_global_grasps=True,
        has_push_actions=True, has_pick_remove_actions=True, has_sequences=True,
        has_relation_graph=True,
    )
    aggregator = MultiTaskLoss(capabilities, AblationConfig())
    total, terms = aggregator({
        name: {"loss": torch.tensor(1.0)} for name in TCDPRGObjective.MODULE_OBJECTIVES
    })
    assert total == 11
    assert len([name for name in terms if name.startswith("loss_") and name != "loss_total"]) == 11


def test_complete_grasp_hungarian_matching_keeps_two_modes_distinct() -> None:
    identity = torch.eye(3)
    output = {
        "translation_world": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True),
        "rotation_matrix": identity.expand(1, 2, 3, 3).clone().requires_grad_(),
        "width_m": torch.tensor([[0.06, 0.04]], requires_grad=True),
        "quality_logit": torch.zeros(1, 2, requires_grad=True),
    }
    labels = {
        "translation_world": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "rotation_matrix": identity.expand(1, 2, 3, 3).clone(),
        "width_m": torch.tensor([[0.04, 0.06]]),
        "quality_target": torch.ones(1, 2),
        "target_valid": torch.ones(1, 2, dtype=torch.bool),
        "quality_valid": torch.ones(1, 2, dtype=torch.bool),
        "sample_valid": torch.tensor([True]),
    }
    losses = CompleteGraspSetLoss()(output, labels)
    assert losses["grasp_translation"] < 1e-7
    assert losses["grasp_width"] < 1e-7
    losses["loss"].backward()


def test_parallel_jaw_symmetric_rotation_has_zero_set_loss() -> None:
    identity = torch.eye(3)
    swapped = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    output = {
        "translation_world": torch.zeros(1, 1, 3, requires_grad=True),
        "rotation_matrix": swapped.expand(1, 1, 3, 3).clone().requires_grad_(),
        "width_m": torch.full((1, 1), 0.05, requires_grad=True),
        "quality_logit": torch.zeros(1, 1, requires_grad=True),
    }
    labels = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": identity.expand(1, 1, 3, 3).clone(),
        "width_m": torch.full((1, 1), 0.05),
        "quality_target": torch.ones(1, 1),
        "target_valid": torch.ones(1, 1, dtype=torch.bool),
        "quality_valid": torch.ones(1, 1, dtype=torch.bool),
        "sample_valid": torch.tensor([True]),
    }
    loss = CompleteGraspSetLoss()(output, labels)
    assert loss["grasp_rotation"] < 1e-7
    loss["loss"].backward()
    assert torch.isfinite(output["rotation_matrix"].grad).all()


def test_partial_grasp_set_does_not_make_unmatched_queries_negative() -> None:
    identity = torch.eye(3)
    output = {
        "translation_world": torch.tensor(
            [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], requires_grad=True
        ),
        "rotation_matrix": identity.expand(1, 2, 3, 3).clone().requires_grad_(),
        "width_m": torch.full((1, 2), 0.05, requires_grad=True),
        "quality_logit": torch.tensor([[0.0, 100.0]], requires_grad=True),
    }
    labels = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": identity.expand(1, 2, 3, 3).clone(),
        "width_m": torch.full((1, 2), 0.05),
        "quality_target": torch.tensor([[1.0, 0.0]]),
        "target_valid": torch.tensor([[True, False]]),
        "quality_valid": torch.tensor([[True, False]]),
        "sample_valid": torch.tensor([True]),
        "unmatched_quality_valid": torch.tensor([False]),
    }
    quality = CompleteGraspSetLoss()(output, labels)["grasp_quality"]
    assert torch.allclose(quality, torch.log(torch.tensor(2.0)))


def test_only_queries_near_explicit_negatives_receive_negative_supervision() -> None:
    identity = torch.eye(3)
    output = {
        "translation_world": torch.tensor([[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]
        ]]),
        "rotation_matrix": identity.expand(1, 3, 3, 3).clone(),
        "width_m": torch.full((1, 3), 0.05),
        # The far open-world query must be ignored despite its high score.
        "quality_logit": torch.tensor([[0.0, 0.0, 100.0]]),
    }
    labels = {
        "translation_world": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "rotation_matrix": identity.expand(1, 1, 3, 3).clone(),
        "width_m": torch.full((1, 1), 0.05),
        "quality_target": torch.ones(1, 1),
        "target_valid": torch.ones(1, 1, dtype=torch.bool),
        "quality_valid": torch.ones(1, 1, dtype=torch.bool),
        "sample_valid": torch.tensor([True]),
        "unmatched_quality_valid": torch.tensor([False]),
        "negative_translation_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "negative_rotation_matrix": identity.expand(1, 1, 3, 3).clone(),
        "negative_width_m": torch.full((1, 1), 0.05),
        "negative_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    quality = CompleteGraspSetLoss()(output, labels)["grasp_quality"]
    assert torch.allclose(quality, torch.log(torch.tensor(2.0)))


def test_unknown_task_grasp_marks_label_set_non_exhaustive() -> None:
    pose = torch.zeros(1, 2, 7)
    pose[..., 6] = 1.0
    batch = {
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "action_type": torch.full((1, 2), int(ActionType.TASK_GRASP)),
        "evaluation_status": torch.tensor([[
            int(CandidateStatus.POSITIVE), int(CandidateStatus.UNKNOWN_UNTESTED)
        ]]),
        "action_improves_state": torch.tensor([[True, False]]),
        "action_parameters": {
            "task_grasp_pose_world": pose,
            "grasp_width_m": torch.full((1, 2), 0.05),
        },
    }
    labels = build_grasp_proposal_labels(batch, ModelConfig(task_grasp_candidates=2))
    assert labels["sample_valid"].item()
    assert not labels["unmatched_quality_valid"].item()
    assert labels["target_valid"].sum().item() == 1


def test_sampled_global_labels_never_imply_complete_unmatched_negatives() -> None:
    pose = torch.zeros(1, 3, 7)
    pose[..., 6] = 1.0
    batch = {
        "global_loss_sample_valid": torch.tensor([True]),
        "global_grasp_labels": {
            "object_index": torch.zeros(1, 3, dtype=torch.long),
            "grasp_pose_world": pose,
            "width_m": torch.full((1, 3), 0.05),
            "scene_executable": torch.tensor([[1, 0, -1]], dtype=torch.int8),
            "valid_mask": torch.ones(1, 3, dtype=torch.bool),
            "label_set_complete": torch.tensor([False]),
        },
    }
    labels = build_global_grasp_labels(batch, ModelConfig(global_grasp_candidates=3))
    assert labels is not None
    assert labels["target_valid"].sum().item() == 1
    assert labels["negative_valid"].sum().item() == 1
    assert not labels["unmatched_quality_valid"].item()


def test_global_grasp_packing_balances_objects_before_truncation() -> None:
    pose = torch.zeros(1, 8, 7)
    pose[..., 6] = 1.0
    packed = _pack_grasp_set(
        pose, torch.full((1, 8), 0.05), torch.ones(1, 8, dtype=torch.bool),
        torch.ones(1, 8), torch.ones(1, 8, dtype=torch.bool), 4,
        object_index=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
    )
    assert packed["object_index"].tolist() == [[0, 1, 0, 1]]


def test_global_grasp_object_assignment_is_an_internal_set_term() -> None:
    identity = torch.eye(3)
    object_logits = torch.tensor(
        [[[10.0, -10.0], [-10.0, 10.0]]], requires_grad=True
    )
    output = {
        "translation_world": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], requires_grad=True
        ),
        "rotation_matrix": identity.expand(1, 2, 3, 3).clone().requires_grad_(),
        "width_m": torch.full((1, 2), 0.05, requires_grad=True),
        "quality_logit": torch.full((1, 2), 10.0, requires_grad=True),
        "object_logits": object_logits,
    }
    labels = {
        "translation_world": output["translation_world"].detach().clone(),
        "rotation_matrix": output["rotation_matrix"].detach().clone(),
        "width_m": output["width_m"].detach().clone(),
        "quality_target": torch.ones(1, 2),
        "target_valid": torch.ones(1, 2, dtype=torch.bool),
        "quality_valid": torch.ones(1, 2, dtype=torch.bool),
        "object_index": torch.tensor([[0, 1]]),
        "sample_valid": torch.tensor([True]),
    }
    loss = CompleteGraspSetLoss()(output, labels)
    assert loss["grasp_object"] < 1e-6
    loss["loss"].backward()
    assert torch.isfinite(object_logits.grad).all()


def test_push_utility_uses_ground_truth_direction_and_keeps_failed_transition() -> None:
    output = {
        "object_logits": torch.zeros(1, 1),
        "contact_logits": torch.zeros(1, 1),
        "direction_logits": torch.zeros(1, 1, 4),
        "direction_residual": torch.zeros(1, 1, 2),
        "utility_delta": torch.arange(4.0).reshape(1, 1, 4),
    }
    batch = {
        "xyz": torch.zeros(1, 1, 3),
        "point_mask": torch.ones(1, 1, dtype=torch.bool),
        "instance_id": torch.zeros(1, 1, dtype=torch.long),
        "candidate_mask": torch.ones(1, 1, dtype=torch.bool),
        "action_type": torch.zeros(1, 1, dtype=torch.long),
        "acted_object": torch.zeros(1, 1, dtype=torch.long),
        "object_mask": torch.ones(1, 1, dtype=torch.bool),
        "object_active": torch.ones(1, 1, dtype=torch.bool),
        "action_improves_state": torch.zeros(1, 1, dtype=torch.bool),
        "evaluation_status": torch.ones(1, 1, dtype=torch.long),
        "potential_delta": torch.zeros(1, 1, 5),
        "potential_after_valid": torch.zeros(1, 1, dtype=torch.bool),
        "action_parameters": {
            "push_contact_world": torch.zeros(1, 1, 3),
            "push_direction_world": torch.tensor([[[-1.0, 0.0, 0.0]]]),
            "risk_unstable": torch.ones(1, 1),
            "risk_out_of_workspace": torch.zeros(1, 1),
            "risk_other_invalid": torch.zeros(1, 1),
        },
    }
    gathered, labels = build_push_supervision(output, batch, ModelConfig())
    assert gathered["utility_delta"].item() == 2.0
    assert labels["utility_delta"].item() == -1.0
    assert labels["utility_valid"].item()
