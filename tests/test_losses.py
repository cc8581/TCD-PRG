import pytest
import torch

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.diagnostics import family_gradient_norms
from tcd_prg.losses.labels import (
    build_grasp_proposal_labels,
    build_push_supervision,
)
from tcd_prg.losses.masked import multi_positive_listwise_loss, safe_smooth_l1
from tcd_prg.losses.objective import TCDPRGObjective
from tcd_prg.losses.total import MultiTaskLoss
from tcd_prg.losses.verifier import GraspVerifierLoss


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


def test_positive_only_policy_row_has_no_effective_loss_or_gradient() -> None:
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    positive = torch.tensor([[True, True]])
    loss = multi_positive_listwise_loss(logits, positive, torch.ones_like(positive))
    assert loss == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


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


def test_exactly_eleven_proposal_v2_objectives() -> None:
    assert len(TCDPRGObjective.MODULE_OBJECTIVES) == 11
    assert tuple(MultiTaskLoss.DEFAULT_WEIGHTS) == TCDPRGObjective.MODULE_OBJECTIVES
    capabilities = DatasetCapabilities(
        has_instance_masks=True, has_task_regions=True, has_task_grasps=True,
        has_global_grasps=True,
        has_push_actions=True, has_pick_remove_actions=True, has_sequences=True,
        has_relation_graph=True,
    )
    aggregator = MultiTaskLoss(capabilities, AblationConfig())
    total, terms = aggregator({
        name: {"loss": torch.tensor(1.0)} for name in TCDPRGObjective.MODULE_OBJECTIVES
    })
    assert total == 11
    assert len([name for name in terms if name.startswith("loss_") and name != "loss_total"]) == 11


def test_verifier_reports_class_balance_and_prior_baseline() -> None:
    losses = GraspVerifierLoss()(
        {"overall_logit": torch.tensor([[2.0, -2.0], [-1.0, 3.0]])},
        {
            "overall_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "overall_valid": torch.ones(2, 2, dtype=torch.bool),
        },
    )
    assert losses["verifier_supervised_rows"] == 2
    assert losses["verifier_positive_candidates"] == 2
    assert losses["verifier_negative_candidates"] == 2
    assert losses["verifier_balanced_accuracy"] == 1
    assert losses["verifier_prior_bce"] == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert losses["verifier_auroc"] == 1
    assert losses["verifier_average_precision"] == 1
    assert losses["verifier_ranking_metrics_valid"] == 1


def test_verifier_marks_single_class_ranking_metrics_invalid() -> None:
    losses = GraspVerifierLoss()(
        {"overall_logit": torch.tensor([[1.0, 2.0]])},
        {
            "overall_target": torch.ones(1, 2),
            "overall_valid": torch.ones(1, 2, dtype=torch.bool),
        },
    )
    assert losses["verifier_ranking_metrics_valid"] == 0
    assert losses["verifier_auroc"] == 0
    assert losses["verifier_average_precision"] == 0


def test_activity_helpers_report_fraction_of_supervised_rows() -> None:
    valid = torch.tensor([[True, False], [False, False]])
    assert TCDPRGObjective._row_active(valid).float().mean() == 0.5

    positive = torch.tensor([[True, False], [True, False]])
    evaluated = torch.tensor([[True, True], [True, False]])
    active = TCDPRGObjective._listwise_active_rows(positive, evaluated)
    assert active.tolist() == [True, False]
    assert active.float().mean() == 0.5


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
            # 未执行/未验证的候选 valid=False，回退到 action_improves_state。
            "verifier_overall_valid": torch.zeros(1, 2, dtype=torch.bool),
            "verifier_overall_target": torch.zeros(1, 2),
            "grasp_confidence": torch.ones(1, 2),
        },
    }
    labels = build_grasp_proposal_labels(batch, ModelConfig(task_grasp_candidates=2))
    assert labels["sample_valid"].item()
    assert not labels["unmatched_quality_valid"].item()
    assert labels["target_valid"].sum().item() == 1


def test_grasp_proposal_labels_require_verifier_contract() -> None:
    pose = torch.zeros(1, 1, 7)
    pose[..., 6] = 1.0
    batch = {
        "candidate_mask": torch.ones(1, 1, dtype=torch.bool),
        "action_type": torch.full((1, 1), int(ActionType.TASK_GRASP)),
        "evaluation_status": torch.tensor([[int(CandidateStatus.POSITIVE)]]),
        "action_improves_state": torch.tensor([[True]]),
        "action_parameters": {
            "task_grasp_pose_world": pose,
            "grasp_width_m": torch.full((1, 1), 0.05),
        },
    }
    # verifier 字段是硬契约：缺失必须报错而非静默改变监督语义。
    with pytest.raises(KeyError):
        build_grasp_proposal_labels(batch, ModelConfig(task_grasp_candidates=1))


def test_task_grasp_label_routes_are_mutually_exclusive() -> None:
    pose = torch.zeros(1, 4, 7)
    pose[..., 6] = 1.0
    batch = {
        "candidate_mask": torch.ones(1, 4, dtype=torch.bool),
        "action_type": torch.full((1, 4), int(ActionType.TASK_GRASP)),
        "evaluation_status": torch.tensor([[
            int(CandidateStatus.POSITIVE),
            int(CandidateStatus.NEGATIVE),
            int(CandidateStatus.NEGATIVE),
            int(CandidateStatus.NEGATIVE),
        ]]),
        "action_improves_state": torch.tensor([[True, False, False, False]]),
        "action_parameters": {
            "task_grasp_pose_world": pose,
            "grasp_width_m": torch.full((1, 4), 0.05),
            "grasp_confidence": torch.ones(1, 4),
            "verifier_overall_valid": torch.ones(1, 4, dtype=torch.bool),
            # Deliberately mark every pose physically executable: explicit subtype
            # routing must still prevent wrong/collision/approach from becoming positive.
            "verifier_overall_target": torch.ones(1, 4),
            "verifier_task_compatibility_valid": torch.tensor(
                [[True, True, False, False]]
            ),
            "verifier_task_compatibility_target": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]]
            ),
            "verifier_collision_valid": torch.tensor([[False, False, True, False]]),
            "verifier_collision_target": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            "verifier_approach_valid": torch.tensor([[False, False, False, True]]),
            "verifier_approach_target": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        },
    }
    labels = build_grasp_proposal_labels(
        batch, ModelConfig(task_grasp_candidates=4)
    )
    assert labels["target_valid"].sum().item() == 1
    assert labels["wrong_region_valid"].sum().item() == 1
    assert labels["collision_diverted_to_verifier"].item() == 1
    assert labels["approach_diverted_to_verifier"].item() == 1


def test_push_utility_uses_ground_truth_direction_and_keeps_failed_transition() -> None:
    output = {
        "object_logits": torch.zeros(1, 1),
        "contact_logits": torch.zeros(1, 1),
        "direction_logits": torch.zeros(1, 1, 4),
        "direction_residual": torch.zeros(1, 1, 4, 2),
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
    assert labels["direction_evaluated"].any()
    assert not labels["direction_positive"].any()
    assert labels["direction_valid"].any()
    assert labels["direction_bin"].shape == labels["direction_valid"].shape
    assert labels["contact_valid"].any()
    assert not (labels["contact_target"] > 0).any()

    positive_batch = {**batch, "action_improves_state": torch.ones(1, 1, dtype=torch.bool)}
    _, positive_labels = build_push_supervision(output, positive_batch, ModelConfig())
    assert positive_labels["direction_positive"].any()
    assert positive_labels["direction_residual_valid"].any()
    assert (positive_labels["contact_target"] > 0).any()


def test_family_gradient_audit_reports_weighted_shared_norms() -> None:
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    first = parameter.square().sum()
    second = (3.0 * parameter).sum()
    result = family_gradient_norms(
        {"first": first, "second": second}, (parameter,), first + second
    )
    assert result["first"] == pytest.approx(float((2.0 * parameter).norm()))
    assert result["second"] == pytest.approx(float(torch.full_like(parameter, 3.0).norm()))
    assert result["total"] == pytest.approx(float((2.0 * parameter + 3.0).norm()))
