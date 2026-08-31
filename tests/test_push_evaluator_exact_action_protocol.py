from __future__ import annotations

import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.evaluators.push_integrated import deterministic_target_prompt_batch
from tcd_prg.models.push_condition import PushCondition
from tcd_prg.trainers.push_evaluator import push_effectiveness_eligibility


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


def test_far_logged_contact_remains_evaluator_supervision():
    valid = push_effectiveness_eligibility(_batch(0.08), _condition())
    assert bool(valid[0, 0])


def test_invisible_target_is_not_evaluator_supervision():
    valid = push_effectiveness_eligibility(_batch(), _condition(False))
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
