from __future__ import annotations

import pytest
import torch

from tcd_prg.config import (
    AblationConfig, BackboneConfig, GraphConfig, LossConfig, ModelConfig,
    RegionHeadConfig, RouterConfig, TCDPRGConfig, TrainingConfig,
)
from tcd_prg.datasets import TaskOrientedClutterAdapter
from tcd_prg.datasets.collate import collate_unified
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.observation.saved import SavedObservationProvider
from tcd_prg.trainers import Trainer
from tcd_prg.datasets.torch_dataset import (
    DistributedEvaluationSampler,
    DistributedWeightedStateSampler,
)


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


def test_checkpoint_rejects_obsolete_schema(tmp_path, tiny_batch) -> None:
    config = TCDPRGConfig(
        model=ModelConfig(feature_dim=32, task_dim=16, activation_checkpointing=False),
        training=TrainingConfig(device="cpu", amp=False),
        output_dir=str(tmp_path),
    )
    model = TCDPRGModel(config.model)
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        config,
        lambda module, batch: (module(batch)["region"]["visibility_logit"].mean(), {}),
    )
    path = tmp_path / "obsolete.pt"
    torch.save({"schema_version": 2}, path)
    with pytest.raises(RuntimeError, match="expects schema 4"):
        trainer.load_checkpoint(path)


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


@pytest.mark.real_data
def test_real_state_group_all_enabled_training_losses_are_finite(dataset_root) -> None:
    scene_root = dataset_root / "task_clutter_scenes_20_categories"
    provider = SavedObservationProvider(scene_root, scene_root / "metadata.json", 256)
    region_root = (
        dataset_root.parent / "Grasp_20_class_object_3D_model"
        / "data" / "manual_function_regions_v1"
    )
    adapter = TaskOrientedClutterAdapter(
        dataset_root, observation_provider=provider, point_count=256,
        functional_region_root=region_root,
    )
    batch = collate_unified([adapter.load_sample(1, 0, 0, 0)])
    config = ModelConfig(feature_dim=32, task_dim=16, activation_checkpointing=False)
    ablation = AblationConfig(use_gripper_scene_verifier=False)
    model = TCDPRGModel(
        config, ablation, GraphConfig(layers=1), RouterConfig(layers=1),
        BackboneConfig(attention_points=64),
    )
    objective = TCDPRGObjective(
        adapter.capabilities, config, ablation, LossConfig(), RegionHeadConfig()
    )
    loss, terms = objective(model, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    assert int(batch["policy_success_mask"].sum()) > 0
    assert int(batch["policy_success_mask"].sum()) < int(batch["action_improves_state"].sum())
    assert not any("approach" in key and key.startswith("push_") for key in terms)
    assert not any("outcome" in key and key.startswith(("push_", "remove_")) for key in terms)
    assert not any("depth" in key and "proposal" in key for key in terms)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters() if parameter.grad is not None
    )


def test_distributed_training_sampler_avoids_rank_overlap() -> None:
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    rank0 = set(DistributedWeightedStateSampler(weights, 0, 2, 4, seed=9))
    rank1 = set(DistributedWeightedStateSampler(weights, 1, 2, 4, seed=9))
    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == set(range(4))


def test_distributed_validation_sampler_has_no_padding_or_overlap() -> None:
    shards = [set(DistributedEvaluationSampler(7, rank, 3)) for rank in range(3)]
    assert set.union(*shards) == set(range(7))
    assert all(shards[left].isdisjoint(shards[right])
               for left in range(3) for right in range(left + 1, 3))
