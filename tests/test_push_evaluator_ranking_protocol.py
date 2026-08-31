from __future__ import annotations

import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.evaluators.push_effectiveness import (
    proposal_positive_match_masks,
    push_candidate_ranking_counts,
)


def _batch() -> dict[str, object]:
    return {
        "candidate_mask": torch.tensor([[True, True]]),
        "action_type": torch.tensor([[int(ActionType.PUSH), int(ActionType.PUSH)]]),
        "evaluation_status": torch.tensor(
            [[int(CandidateStatus.POSITIVE), int(CandidateStatus.NEGATIVE)]]
        ),
        "action_improves_state": torch.tensor([[True, False]]),
        "acted_object": torch.tensor([[0, 0]]),
        "action_parameters": {
            "push_contact_world": torch.tensor([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]]),
            "push_direction_world": torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        },
    }


def test_candidate_ranking_is_conditioned_on_proposal_success() -> None:
    rows = [
        {
            "object": torch.tensor([0, 0]),
            "contact_world": torch.tensor([[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "direction_world": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "proposal_score": torch.tensor([0.9, 0.8]),
            "effective_probability": torch.tensor([0.1, 0.9]),
        }
    ]
    masks = proposal_positive_match_masks(
        rows, _batch(), contact_threshold_m=0.01, direction_threshold_deg=5.0
    )
    assert masks[0].tolist() == [False, True]
    counts = push_candidate_ranking_counts(rows, masks)
    assert counts["push_evaluator_positive_candidate_set_count"] == 1
    assert counts["push_evaluator_hit_at_1_count"] == 1
    assert counts["push_evaluator_hit_at_5_count"] == 1


def test_proposal_miss_is_not_charged_to_evaluator() -> None:
    rows = [
        {
            "object": torch.tensor([0]),
            "contact_world": torch.tensor([[0.30, 0.0, 0.0]]),
            "direction_world": torch.tensor([[1.0, 0.0, 0.0]]),
            "proposal_score": torch.tensor([0.9]),
            "effective_probability": torch.tensor([0.9]),
        }
    ]
    masks = proposal_positive_match_masks(
        rows, _batch(), contact_threshold_m=0.01, direction_threshold_deg=5.0
    )
    counts = push_candidate_ranking_counts(rows, masks)
    assert counts["push_evaluator_positive_candidate_set_count"] == 0
