from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tcd_prg.config import ModelConfig, TCDPRGConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets.policy_candidates import (
    RAW_FIELDS,
    cache_manifest,
    load_candidate_batch,
    match_generated_candidates,
    save_candidate_entry,
    validate_generated_policy_coverage,
)


def _generated() -> dict[str, torch.Tensor]:
    kind = torch.full((1, 3), int(ActionType.PUSH), dtype=torch.long)
    return {
        "type": kind,
        "object": torch.zeros_like(kind),
        "contact_world": torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "direction_world": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
        "pose_world": torch.full((1, 3, 7), float("nan")),
        "destination_world": torch.full((1, 3, 3), float("nan")),
        "width_m": torch.full((1, 3), float("nan")),
        "proposal_score": torch.ones(1, 3),
        "point_index": torch.arange(3).reshape(1, 3),
        "direction_bin": torch.tensor([[0, 4, 0]]),
        "direction_score": torch.full((1, 3), 0.5),
        "evidence": torch.zeros(1, 3, 7),
        "valid": torch.ones(1, 3, dtype=torch.bool),
    }


def _teacher() -> dict[str, object]:
    return {
        "action_type": torch.full((1, 2), int(ActionType.PUSH), dtype=torch.long),
        "acted_object": torch.zeros(1, 2, dtype=torch.long),
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "evaluation_status": torch.tensor([[
            int(CandidateStatus.POSITIVE), int(CandidateStatus.NEGATIVE)
        ]]),
        "policy_success_mask": torch.tensor([[True, False]]),
        "action_parameters": {
            "push_contact_world": torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
            "push_direction_world": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            "removal_grasp_pose_world": torch.full((1, 2, 7), float("nan")),
            "task_grasp_pose_world": torch.full((1, 2, 7), float("nan")),
            "grasp_width_m": torch.full((1, 2), float("nan")),
        },
    }


def test_generated_policy_matching_keeps_unmatched_candidates_unknown() -> None:
    labels = match_generated_candidates(_generated(), _teacher(), 0, ModelConfig())
    assert labels["label_status"].tolist() == [
        int(CandidateStatus.POSITIVE),
        int(CandidateStatus.NEGATIVE),
        int(CandidateStatus.UNKNOWN_UNTESTED),
    ]
    assert labels["policy_success"].tolist() == [True, False, False]
    assert labels["matched_teacher_index"].tolist() == [0, 1, -1]
    assert not labels["match_conflict"].any()


def test_generated_policy_matching_keeps_positive_negative_conflict_unknown() -> None:
    teacher = _teacher()
    teacher["action_parameters"]["push_contact_world"][0, 1] = torch.tensor([0.0, 0.0, 0.0])
    teacher["action_parameters"]["push_direction_world"][0, 1] = torch.tensor([1.0, 0.0, 0.0])
    labels = match_generated_candidates(_generated(), teacher, 0, ModelConfig())
    assert labels["label_status"][0] == int(CandidateStatus.UNKNOWN_UNTESTED)
    assert labels["match_conflict"][0]
    assert labels["matched_teacher_index"][0] == -1


def test_local_positive_outside_successful_sequence_is_not_a_policy_negative() -> None:
    teacher = _teacher()
    teacher["policy_success_mask"] = torch.zeros(1, 2, dtype=torch.bool)
    labels = match_generated_candidates(_generated(), teacher, 0, ModelConfig())
    assert labels["label_status"].tolist() == [
        int(CandidateStatus.UNKNOWN_UNTESTED),
        int(CandidateStatus.NEGATIVE),
        int(CandidateStatus.UNKNOWN_UNTESTED),
    ]


def test_generated_policy_cache_is_checkpoint_and_config_versioned(tmp_path) -> None:
    checkpoint = tmp_path / "upstream.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = TCDPRGConfig(model=ModelConfig())
    manifest = cache_manifest(checkpoint, config)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sample = SimpleNamespace(
        observation=SimpleNamespace(
            scene_id=1, state_id=2, task_index=3,
            xyz=np.zeros((2, 3), np.float32), rgb=np.zeros((2, 3), np.float32),
            instance_id=np.zeros(2, np.int64), target_mask=np.ones(2, bool),
            object_pose=np.zeros((1, 7), np.float32),
            object_present=np.ones(1, bool), object_active=np.ones(1, bool),
        ),
        candidates=SimpleNamespace(
            candidate_action_ids=np.asarray([7, 8], dtype=np.int64),
            action_type=np.asarray([0, 0], np.int8),
            acted_object=np.asarray([0, 0], np.int16),
            evaluation_status=np.asarray([1, 0], np.int8),
            success_mask=np.asarray([True, False]),
            potential_delta=np.zeros((2, 5), np.float32),
            action_parameters={},
        ),
        sequences=(),
    )
    generated = _generated()
    labels = match_generated_candidates(generated, _teacher(), 0, config.model)
    assert set(RAW_FIELDS).issubset(generated)
    save_candidate_entry(tmp_path, sample, generated, labels)
    batch = load_candidate_batch(
        [sample], tmp_path, config, manifest["checkpoint_sha256"]
    )
    assert batch["label_status"].tolist() == [[1, 0, -1]]
    assert batch["valid"].all()
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_candidate_batch([sample], tmp_path, config, "0" * 64)


def test_generated_policy_coverage_rejects_uninformative_pure_stage() -> None:
    manifest = {"splits": {"train": {
        "entries": 100,
        "groups_with_known_positive": 4,
        "effective_policy_rows": 1,
    }}}
    with pytest.raises(ValueError, match="positive coverage"):
        validate_generated_policy_coverage(
            manifest, "train", minimum_positive=0.05, minimum_effective=0.01
        )
