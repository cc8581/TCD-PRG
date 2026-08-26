from __future__ import annotations

import pytest

from tcd_prg.config import LossConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.scripts.train import validate_checkpoint_gate


def checkpoint(stage: str) -> dict:
    return {"schema_version": 12, "training_stage": stage, "model": {}}


def test_stage_a_starts_without_checkpoint() -> None:
    validate_checkpoint_gate("perception", resume_payload=None, initialize_payload=None)


@pytest.mark.parametrize(
    ("stage", "source"), (("grasp", "perception"), ("push", "grasp"))
)
def test_later_stage_requires_exact_predecessor(stage: str, source: str) -> None:
    validate_checkpoint_gate(
        stage, resume_payload=None, initialize_payload=checkpoint(source)
    )
    with pytest.raises(RuntimeError, match="requires"):
        validate_checkpoint_gate(stage, resume_payload=None, initialize_payload=None)
    with pytest.raises(RuntimeError, match="requires"):
        validate_checkpoint_gate(
            stage, resume_payload=None, initialize_payload=checkpoint(stage)
        )


def test_resume_requires_same_stage() -> None:
    validate_checkpoint_gate(
        "grasp", resume_payload=checkpoint("grasp"), initialize_payload=None
    )
    with pytest.raises(RuntimeError, match="Resume requires"):
        validate_checkpoint_gate(
            "grasp", resume_payload=checkpoint("perception"), initialize_payload=None
        )


def test_stage_a_rejects_initialize() -> None:
    with pytest.raises(RuntimeError, match="must not use"):
        validate_checkpoint_gate(
            "perception", resume_payload=None, initialize_payload=checkpoint("perception")
        )


def test_stage_name_cannot_disguise_wrong_losses() -> None:
    with pytest.raises(ValueError, match="requires exactly loss families"):
        TCDPRGConfig(
            training=TrainingConfig(stage="grasp"),
            losses=LossConfig(instance=1.0, region=1.0, task_grasp=0.0),
        ).validate()
