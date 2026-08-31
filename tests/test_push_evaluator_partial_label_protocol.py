from __future__ import annotations

import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.evaluators.push_effectiveness import (
    proposal_known_outcome_masks,
    push_candidate_ranking_counts,
    push_effectiveness_metrics,
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


def test_unknown_top1_is_indeterminate_not_negative() -> None:
    rows = [
        {
            "object": torch.tensor([0, 0, 0]),
            "contact_world": torch.tensor([[0.30, 0.0, 0.0], [0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]),
            "direction_world": torch.tensor([[1.0, 0.0, 0.0]] * 3),
            "proposal_score": torch.tensor([0.9, 0.8, 0.7]),
            "effective_probability": torch.tensor([0.99, 0.90, 0.10]),
        }
    ]
    positive, _, known = proposal_known_outcome_masks(
        rows, _batch(), contact_threshold_m=0.01, direction_threshold_deg=5.0
    )
    counts = push_candidate_ranking_counts(rows, positive, known)
    assert counts["push_evaluator_positive_candidate_set_count"] == 1
    assert counts["push_evaluator_top1_evaluable_count"] == 0
    assert counts["push_evaluator_top5_evaluable_count"] == 1
    assert counts["push_evaluator_hit_at_5_count"] == 1


def test_rank_auroc_handles_ties_without_pairwise_matrix() -> None:
    metrics = push_effectiveness_metrics(
        torch.tensor([0.9, 0.8, 0.8, 0.1]),
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
    )
    assert metrics["push_evaluator_auroc"].item() == 0.875
