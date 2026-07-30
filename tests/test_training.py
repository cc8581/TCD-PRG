from __future__ import annotations

import torch

from tcd_prg.config import ModelConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.models import TCDPRGModel
from tcd_prg.trainers import Trainer


class _SkipOnceScaler:
    """CPU test double matching the GradScaler methods used by Trainer."""

    def __init__(self) -> None:
        self.scale_value = 2.0
        self.calls = 0

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        self.calls += 1
        if self.calls > 1:
            optimizer.step()

    def update(self) -> None:
        if self.calls == 1:
            self.scale_value = 1.0


def test_checkpoint_save_resume_consistency(tmp_path, tiny_batch) -> None:
    config = TCDPRGConfig(
        model=ModelConfig(feature_dim=32, task_dim=16, activation_checkpointing=False),
        training=TrainingConfig(device="cpu", amp=False, max_optimizer_steps=1, gradient_accumulation_steps=1),
        output_dir=str(tmp_path),
    )
    model = TCDPRGModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def loss_step(module, batch):
        value = module(batch)["region"]["visibility_logit"].square().mean()
        return value, {"loss": value}

    trainer = Trainer(model, optimizer, config, loss_step)
    trainer.train([tiny_batch])
    path = tmp_path / "resume.pt"
    trainer.save_checkpoint(path)
    reference = {key: value.clone() for key, value in model.state_dict().items()}
    replacement = TCDPRGModel(config.model)
    resumed = Trainer(replacement, torch.optim.AdamW(replacement.parameters(), lr=1e-4), config, loss_step)
    resumed.load_checkpoint(path)
    assert resumed.state.optimizer_steps == 1
    assert resumed.state.amp_skipped_steps == 0
    assert all(torch.equal(reference[key], replacement.state_dict()[key]) for key in reference)


def test_amp_overflow_does_not_advance_optimizer_step(tmp_path) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu", amp=False, max_optimizer_steps=1,
            gradient_accumulation_steps=1,
        ),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = {"x": torch.randn(2, 3), "y": torch.randn(2, 1)}

    def loss_step(module, batch):
        prediction = module(batch["x"])
        loss = torch.nn.functional.mse_loss(prediction, batch["y"])
        return loss, {"loss_total": loss}

    trainer = Trainer(model, optimizer, config, loss_step)
    trainer.scaler = _SkipOnceScaler()
    state = trainer.train([batch], groups_per_effective_epoch=1)
    assert state.optimizer_steps == 1
    assert state.amp_skipped_steps == 1
    assert state.samples_seen == 2
    assert optimizer.state


def test_ten_batch_overfit_smoke(tiny_batch) -> None:
    config = ModelConfig(feature_dim=32, task_dim=16, activation_checkpointing=False)
    model = TCDPRGModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(tiny_batch)["region"]["visibility_logit"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
