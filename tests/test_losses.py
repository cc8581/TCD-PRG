import pytest
import torch

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.diagnostics import family_gradient_norms
from tcd_prg.losses.masked import multi_positive_listwise_loss, safe_smooth_l1
from tcd_prg.losses.objective import TCDPRGObjective
from tcd_prg.losses.total import MultiTaskLoss
from tcd_prg.trainers import finalize_push_validation_metrics


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


def test_push_validation_utility_coverage_uses_eligible_denominator() -> None:
    metrics = finalize_push_validation_metrics({
        "push_positive_actions_total_count": 10,
        "push_positive_actions_direction_covered_count": 8,
        "push_positive_actions_utility_eligible_count": 4,
        "push_positive_actions_utility_covered_count": 3,
    })
    assert metrics["push_direction_positive_coverage"] == pytest.approx(0.8)
    assert metrics["push_utility_valid_coverage"] == pytest.approx(0.75)


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


def test_exactly_seven_minimal_architecture_objectives() -> None:
    assert len(TCDPRGObjective.MODULE_OBJECTIVES) == 7
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
    assert total == 7
    assert len([name for name in terms if name.startswith("loss_") and name != "loss_total"]) == 7






def test_activity_helpers_report_fraction_of_supervised_rows() -> None:
    valid = torch.tensor([[True, False], [False, False]])
    assert TCDPRGObjective._row_active(valid).float().mean() == 0.5


def test_listwise_activity_requires_positive_and_negative_competitors() -> None:
    positive = torch.tensor([[True, False], [True, True], [False, False]])
    valid = torch.ones_like(positive)
    assert torch.equal(
        TCDPRGObjective._listwise_active_rows(positive, valid),
        torch.tensor([True, False, False]),
    )


def test_push_direction_activity_aligns_batch_and_candidate_axes() -> None:
    positive = torch.zeros((8, 25, 8), dtype=torch.bool)
    valid = torch.zeros_like(positive)
    positive[:, 0, 0] = True
    valid[:, 0, :2] = True

    rank_active = TCDPRGObjective._row_active(
        TCDPRGObjective._listwise_active_rows(positive, valid)
    )
    evaluated_active = TCDPRGObjective._row_active(valid)

    assert rank_active.shape == evaluated_active.shape == (8,)
    assert torch.equal(rank_active | evaluated_active, torch.ones(8, dtype=torch.bool))

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
