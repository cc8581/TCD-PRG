from __future__ import annotations

import pytest

from tcd_prg.config import LossConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.scripts.train import validate_checkpoint_gate


def checkpoint(stage: str) -> dict:
    return {"schema_version": 12, "training_stage": stage, "model": {}}


def test_stage_a_starts_without_checkpoint() -> None:
    validate_checkpoint_gate("perception", resume_payload=None)


def test_stage_b_starts_without_perception_checkpoint() -> None:
    validate_checkpoint_gate("grasp", resume_payload=None)


def test_resume_requires_same_stage() -> None:
    validate_checkpoint_gate(
        "grasp", resume_payload=checkpoint("grasp")
    )
    with pytest.raises(RuntimeError, match="Resume requires"):
        validate_checkpoint_gate(
            "grasp", resume_payload=checkpoint("perception")
        )


def test_stage_name_cannot_disguise_wrong_losses() -> None:
    with pytest.raises(ValueError, match="requires exactly loss families"):
        TCDPRGConfig(
            training=TrainingConfig(stage="grasp"),
            losses=LossConfig(instance=1.0, region=1.0, task_grasp=0.0),
        ).validate()
