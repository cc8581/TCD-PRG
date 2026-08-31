from __future__ import annotations

import pytest
import torch

from tcd_prg.constants import PUSH_DISTANCE_M
from tcd_prg.evaluators.push_effectiveness import push_effectiveness_metrics
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models.push_condition import PushCondition
from tcd_prg.planners.push_decoder import proposal_recall_counts


def test_proposal_recall_uses_fixed_gt_denominator_and_ignores_unknown() -> None:
    rows = [
        {
            "object": torch.tensor([1]),
            "contact_world": torch.tensor([[0.005, 0.0, 0.0]]),
            "direction_world": torch.tensor([[1.0, 0.0, 0.0]]),
        }
    ]
    batch = {
        "xyz": torch.zeros(1, 2, 3),
        "candidate_mask": torch.tensor([[True, True, True]]),
        "action_type": torch.tensor([[0, 0, 0]]),
        "evaluation_status": torch.tensor([[1, 1, -1]]),
        "action_improves_state": torch.tensor([[True, True, True]]),
        "acted_object": torch.tensor([[1, 0, 1]]),
        "action_parameters": {
            "push_contact_world": torch.tensor([[[0.0, 0.0, 0.0]] * 3]),
            "push_direction_world": torch.tensor([[[1.0, 0.0, 0.0]] * 3]),
        },
    }
    hits, total = proposal_recall_counts(
        rows, batch, contact_threshold_m=0.01, direction_threshold_deg=5.0
    )
    assert hits == 1
    assert total == 2


def test_effectiveness_loss_uses_improves_state_and_ignores_unknown() -> None:
    effective = torch.zeros(3, requires_grad=True)
    losses = PushEffectivenessLoss()(
        effective,
        torch.tensor([1, 0, -1]),
        torch.tensor([False, True, True]),
    )
    losses["push_effectiveness"].backward()
    assert effective.grad[2] == 0
    assert effective.grad[0] > 0
    assert effective.grad[1] < 0
    assert losses["push_effectiveness_evaluated_count"] == 2


def test_effectiveness_metrics_report_binary_and_state_ranking_quality() -> None:
    metrics = push_effectiveness_metrics(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([True, False, False, True]),
        torch.tensor([10, 10, 20, 20]),
    )
    assert 0.0 <= metrics["push_evaluator_auprc"] <= 1.0
    assert metrics["push_evaluator_auroc"] == 0.5
    assert metrics["push_evaluator_hit_at_1"] == 0.5
    assert metrics["push_evaluator_recall_at_5"] == 1.0
