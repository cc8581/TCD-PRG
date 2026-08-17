from __future__ import annotations

import json
import torch

from tcd_prg.config import LoggingConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.trainers import Trainer


def _make_trainer(tmp_path, *, max_steps: int, interval: int):
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=max_steps,
            gradient_accumulation_steps=1,
            validation_interval=interval,
        ),
        logging=LoggingConfig(backend="none", log_interval=100),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = {"x": torch.ones(1, 1)}

    def loss_step(module, current):
        loss = module(current["x"]).square().mean()
        return loss, {"loss_total": loss}

    return Trainer(model, optimizer, config, loss_step), batch


def test_resume_boundary_retries_unfinished_validation_before_training(tmp_path):
    source, _ = _make_trainer(tmp_path, max_steps=3, interval=2)
    source.state.optimizer_steps = 2
    source.save_checkpoint(tmp_path / "last.pt")

    resumed, batch = _make_trainer(tmp_path, max_steps=3, interval=2)
    resumed.load_checkpoint(tmp_path / "last.pt")
    calls = []

    def validate(_module):
        calls.append(resumed.state.optimizer_steps)
        return (1.0, 1)

    state = resumed.train([batch], validate=validate)
    assert calls == [2]
    assert state.optimizer_steps == 3
    assert state.last_completed_validation_step == 2
    assert state.pending_validation_step == 0


def test_resume_boundary_uses_completed_validation_log_without_duplicate(tmp_path):
    source, _ = _make_trainer(tmp_path, max_steps=3, interval=2)
    source.state.optimizer_steps = 2
    source.save_checkpoint(tmp_path / "last.pt")
    (tmp_path / "validation_metrics.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "optimizer_step": 2,
                "validation_score": 0.5,
                "best_validation": 0.5,
                "improved": True,
                "validation_items": 1,
                "validation_without_improvement": 0,
                "early_stopping_patience": 20,
                "metrics": {},
                "performance": {"count": 0, "scene_count": 0, "metrics": {}},
                "training_stage": "geometry",
            }
        ) + "\n",
        encoding="utf-8",
    )

    resumed, batch = _make_trainer(tmp_path, max_steps=3, interval=2)
    resumed.load_checkpoint(tmp_path / "last.pt")

    def validate(_module):
        raise AssertionError("completed step-2 validation must not run twice")

    state = resumed.train([batch], validate=validate)
    assert state.optimizer_steps == 3
    assert state.last_completed_validation_step == 2
    assert state.pending_validation_step == 0
    assert state.best_validation == 0.5


def test_resume_non_boundary_continues_training_until_next_validation(tmp_path):
    source, _ = _make_trainer(tmp_path, max_steps=4, interval=2)
    source.state.optimizer_steps = 3
    source.save_checkpoint(tmp_path / "last.pt")

    resumed, batch = _make_trainer(tmp_path, max_steps=4, interval=2)
    resumed.load_checkpoint(tmp_path / "last.pt")
    calls = []

    def validate(_module):
        calls.append(resumed.state.optimizer_steps)
        return (1.0, 1)

    state = resumed.train([batch], validate=validate)
    assert calls == [4]
    assert state.optimizer_steps == 4
    assert state.last_completed_validation_step == 4
    assert state.pending_validation_step == 0


def test_pending_flag_is_persisted_before_validation_failure(tmp_path):
    trainer, batch = _make_trainer(tmp_path, max_steps=2, interval=2)

    def validate(_module):
        raise RuntimeError("synthetic validation failure")

    try:
        trainer.train([batch], validate=validate)
    except RuntimeError as error:
        assert "synthetic validation failure" in str(error)
    else:
        raise AssertionError("validation failure was expected")

    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["trainer_state"]["optimizer_steps"] == 2
    assert payload["trainer_state"]["pending_validation_step"] == 2
    assert payload["trainer_state"]["last_completed_validation_step"] == 0
