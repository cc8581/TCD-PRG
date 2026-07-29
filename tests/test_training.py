from __future__ import annotations

import torch

from tcd_prg.config import ModelConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.models import TCDPRGModel
from tcd_prg.trainers import Trainer


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
    assert all(torch.equal(reference[key], replacement.state_dict()[key]) for key in reference)


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

