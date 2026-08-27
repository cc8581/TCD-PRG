from __future__ import annotations

import json

import pytest
import torch

from tcd_prg.constants import PUSH_DISTANCE_M, CandidateStatus
from tcd_prg.datasets.push_outcome_bank import (
    PushAction,
    PushOutcomeBank,
    PushOutcomeRecord,
    PushSceneState,
)
from tcd_prg.losses.push_critic import PushOutcomeCriticLoss
from tcd_prg.models.push.head import PushHead
from tcd_prg.models.push_condition import PushCondition
from tcd_prg.planners.push_decoder import proposal_recall_counts


def _scene(split: str = "train", state_hash: str = "state-v1") -> PushSceneState:
    return PushSceneState(
        split=split,
        scene_id=1,
        state_id=2,
        state_hash=state_hash,
        object_geometry_ids=("object-a",),
        object_poses_xyzw=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),),
        target_object=0,
        task_category_id=3,
        task_region_id=4,
        simulator_version="pybullet-test",
        physics_parameters={"time_step": 1.0 / 240.0},
        robot_configuration="FR5+AG-160-95",
        label_generation_version="test-v1",
        random_seed=7,
    )


def _action() -> PushAction:
    return PushAction(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), PUSH_DISTANCE_M)


def test_outcome_bank_preserves_unknown_and_robust_trial_confidence(tmp_path) -> None:
    bank = PushOutcomeBank(tmp_path / "bank.jsonl")
    record = PushOutcomeRecord(
        _scene(), _action(), int(CandidateStatus.POSITIVE), True,
        (0.1, 0.2), 0.01, 0.0, False, False, False,
        robust_success_count=18, robust_trial_count=20,
    )
    assert bank.append([record]) == 1
    payload = json.loads((tmp_path / "bank.jsonl").read_text(encoding="utf-8"))
    assert payload["local_robust_success_rate"] == pytest.approx(0.9)
    assert payload["robust_trial_count"] == 20
    with pytest.raises(ValueError, match="Duplicate"):
        bank.append([record])


def test_outcome_bank_rejects_unknown_as_negative_and_nontrain_mining(tmp_path) -> None:
    invalid_unknown = PushOutcomeRecord(
        _scene(), _action(), int(CandidateStatus.UNKNOWN_UNTESTED), False,
        None, None, None, None, None, None,
    )
    with pytest.raises(ValueError, match="UNKNOWN"):
        invalid_unknown.validate()
    validation_record = PushOutcomeRecord(
        _scene("val"), _action(), int(CandidateStatus.NEGATIVE), False,
        None, None, None, False, False, False,
    )
    with pytest.raises(ValueError, match="train scenes only"):
        PushOutcomeBank(tmp_path / "bank.jsonl").append(
            [validation_record], mining=True
        )


def test_proposal_recall_uses_fixed_gt_denominator_and_ignores_unknown() -> None:
    rows = [{
        "object": torch.tensor([1]),
        "contact_world": torch.tensor([[0.005, 0.0, 0.0]]),
        "direction_world": torch.tensor([[1.0, 0.0, 0.0]]),
    }]
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


def test_critic_loss_ignores_unknown_and_keeps_two_targets_separate() -> None:
    effective = torch.zeros(3, requires_grad=True)
    robustness = torch.zeros(3, requires_grad=True)
    losses = PushOutcomeCriticLoss()(
        effective,
        robustness,
        torch.tensor([1, 0, -1]),
        torch.tensor([9, 1, 100]),
        torch.tensor([10, 4, 100]),
    )
    losses["push_critic"].backward()
    assert effective.grad[2] == 0
    assert robustness.grad[2] == 0
    assert losses["push_critic_executed_count"] == 2
    assert losses["push_critic_robust_count"] == 2


def test_critic_heads_detach_proposal_features() -> None:
    head = PushHead(
        dim=16, direction_bins=4, direction_dim=8,
        direction_layers=1, direction_heads=2,
        direction_contact_topk=1, direction_object_topk=1,
        num_categories=4, num_task_regions=4,
    )
    sensor = {
        "xyz": torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        "rgb": torch.zeros(1, 2, 3),
        "point_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    condition = PushCondition(
        torch.tensor([[[1.0, 1.0]]]),
        torch.tensor([[True]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([0]),
    )
    output = head(sensor, condition)
    output["push_effective_logit"].sum().backward()
    assert head.push_effective.weight.grad is not None
    assert head.direction_score.weight.grad is None
    assert head.point_encoder[0].weight.grad is None
