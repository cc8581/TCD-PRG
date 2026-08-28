from __future__ import annotations

import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.evaluators.push_integrated import deterministic_target_prompt_batch
from tcd_prg.models.push.evaluator import PushEffectivenessEvaluator, canonical_push_direction
from tcd_prg.models.push_condition import PushCondition
from tcd_prg.trainers.push_evaluator import push_effectiveness_eligibility


def test_canonical_direction_ignores_proposal_parameterization():
    direction = torch.tensor([[0.99, 0.10, 0.0], [0.99, 0.10, 0.0]])
    canonical, bins, residual = canonical_push_direction(direction, 16)
    assert torch.allclose(canonical[0], canonical[1])
    assert int(bins[0]) == int(bins[1])
    assert torch.allclose(residual[0], residual[1])


def test_evaluator_forward_uses_final_direction_not_raw_bin_residual():
    evaluator = PushEffectivenessEvaluator(feature_dim=4, direction_dim=2)
    push = {
        "proposal_direction_feature": torch.arange(64, dtype=torch.float32).reshape(1, 2, 16, 2),
        "proposal_object_feature": torch.zeros(1, 1, 4),
        "proposal_point_feature": torch.zeros(1, 2, 4),
        "proposal_task_feature": torch.zeros(1, 4),
        "target_center_world": torch.zeros(1, 3),
        "region_center_world": torch.zeros(1, 3),
    }
    base = {
        "point_index": torch.tensor([0]),
        "object": torch.tensor([0]),
        "contact_world": torch.zeros(1, 3),
        "direction_world": torch.tensor([[1.0, 0.0, 0.0]]),
        "push_distance": torch.tensor([0.15]),
    }
    a = dict(base, direction_bin=torch.tensor([0]), direction_residual=torch.tensor([[0.9, 0.9]]))
    b = dict(base, direction_bin=torch.tensor([7]), direction_residual=torch.tensor([[-0.4, 0.2]]))
    assert torch.allclose(evaluator(push, a, batch_index=0), evaluator(push, b, batch_index=0))


def _batch(contact_x: float = 0.0):
    return {
        "xyz": torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        "point_mask": torch.tensor([[True, True]]),
        "candidate_mask": torch.tensor([[True]]),
        "action_type": torch.tensor([[int(ActionType.PUSH)]]),
        "evaluation_status": torch.tensor([[int(CandidateStatus.POSITIVE)]]),
        "action_improves_state": torch.tensor([[True]]),
        "acted_object": torch.tensor([[0]]),
        "action_parameters": {
            "push_contact_world": torch.tensor([[[contact_x, 0.0, 0.0]]]),
            "push_direction_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
        },
    }


def _condition(target_visible: bool = True):
    return PushCondition(
        object_probability=torch.tensor([[[1.0, 0.0]]]),
        object_valid=torch.tensor([[True]]),
        target_probability=torch.tensor([[1.0 if target_visible else 0.0, 0.0]]),
        region_probability=torch.zeros(1, 2),
        target_valid=torch.tensor([target_visible]),
        task_category_id=torch.tensor([0]),
        task_region_id=torch.tensor([0]),
    )


def test_far_logged_contact_is_not_evaluator_supervision():
    valid, anchor = push_effectiveness_eligibility(
        _batch(0.08), _condition(), max_contact_distance_m=0.024
    )
    assert not bool(valid[0, 0])
    assert int(anchor[0, 0]) == -1


def test_invisible_target_is_not_evaluator_supervision():
    valid, _ = push_effectiveness_eligibility(
        _batch(), _condition(False), max_contact_distance_m=0.024
    )
    assert not bool(valid[0, 0])


def test_integrated_prompt_matches_validation_centroid_rule():
    batch = {
        "xyz": torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "point_mask": torch.tensor([[True, True, True]]),
        "target_mask": torch.tensor([[True, True, False]]),
        "task_category_id": torch.tensor([3]),
        "task_region_id": torch.tensor([4]),
    }
    prompted = deterministic_target_prompt_batch(batch)["task_inputs"]
    assert torch.allclose(prompted["target_prompt_xyz"][0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert bool(prompted["target_prompt_valid"][0, 0])
