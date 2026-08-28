from __future__ import annotations

import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.evaluators.push_effectiveness import (
    proposal_positive_match_masks,
    push_candidate_ranking_counts,
)
from tcd_prg.models.staged_checkpoint import push_checkpoint_fingerprint


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
    assert counts["push_evaluator_candidate_set_count"] == 1
    assert counts["push_evaluator_hit_at_1_count"] == 1
    assert counts["push_evaluator_recall_at_5_count"] == 1


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
    assert counts["push_evaluator_candidate_set_count"] == 0


def test_stage_c_fingerprint_tracks_ema_and_exact_tensor_state(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    payload = {
        "model": {"push.weight": torch.tensor([1.0])},
        "ema": {"push.weight": torch.tensor([2.0])},
    }
    torch.save(payload, first)
    torch.save(payload, second)
    digest, source = push_checkpoint_fingerprint(first)
    assert source == "ema"
    assert push_checkpoint_fingerprint(second)[0] == digest
    payload["ema"]["push.weight"] += 1.0
    torch.save(payload, second)
    assert push_checkpoint_fingerprint(second)[0] != digest
