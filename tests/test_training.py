from __future__ import annotations

import json

import pytest
import torch

from tcd_prg.config import (
    AblationConfig,
    BackboneConfig,
    GraphConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    RegionHeadConfig,
    RouterConfig,
    TCDPRGConfig,
    TrainingConfig,
)
from tcd_prg.datasets import TaskOrientedClutterAdapter
from tcd_prg.datasets.collate import collate_unified
from tcd_prg.datasets.torch_dataset import (
    DistributedEvaluationSampler,
    DistributedWeightedStateSampler,
)
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.observation.saved import SavedObservationProvider
from tcd_prg.trainers import Trainer


pytestmark = pytest.mark.usefixtures("fake_graspnet")


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
        training=TrainingConfig(
            device="cpu", amp=False, max_optimizer_steps=1, gradient_accumulation_steps=1
        ),
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
    resumed = Trainer(
        replacement, torch.optim.AdamW(replacement.parameters(), lr=1e-4), config, loss_step
    )
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
    torch.save({"schema_version": 5}, path)
    with pytest.raises(RuntimeError, match="expects schema 10"):
        trainer.load_checkpoint(path)


def test_amp_overflow_does_not_advance_optimizer_step(tmp_path) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=1,
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
    assert state.samples_seen == 4
    assert optimizer.state


def test_trainer_prints_concise_summary_and_saves_every_step_metric(tmp_path, capsys) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=1,
            gradient_accumulation_steps=2,
        ),
        logging=LoggingConfig(backend="none", log_interval=1),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = [{"x": torch.randn(2, 3), "marker": torch.tensor(value)} for value in (1.0, 3.0)]

    def loss_step(module, batch):
        loss = module(batch["x"]).square().mean()
        return loss, {"loss_total": loss, "diagnostic": batch["marker"]}

    Trainer(model, optimizer, config, loss_step).train(batches)
    terminal = capsys.readouterr().out
    assert "[train-start]" in terminal
    assert "Train [geometry] [0000001/0000001]" in terminal
    assert "eta: 0:00:00" in terminal
    assert "loss:" in terminal
    assert "lr:" in terminal
    assert "grad:" in terminal
    assert "time:" in terminal
    assert "data:" in terminal
    assert "  losses:" not in terminal
    assert "  generated:" not in terminal
    assert "[train-done]" in terminal
    records = [
        json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["schema_version"] == 5
    assert records[0]["gradient_norm_after_clip"] <= config.training.gradient_clip_norm
    assert 0.0 < records[0]["gradient_clip_scale"] <= 1.0
    assert records[0]["training_stage"] == "geometry"
    assert records[0]["micro_batches"] == 2
    assert records[0]["window_samples"] == 4
    assert records[0]["global_states_seen"] == 0
    assert records[0]["window_global_states"] == 0
    assert records[0]["diagnostic"] == pytest.approx(2.0)
    assert records[0]["eta_seconds"] == pytest.approx(0.0)
    assert records[0]["data_seconds"] >= 0.0
    assert records[0]["max_memory_mb"] == pytest.approx(0.0)
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "training_events.jsonl").read_text().splitlines()
    ]
    assert events == ["training_started", "training_completed"]


def test_auxiliary_global_stream_replaces_zero_activity_placeholder(tmp_path) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=1,
            gradient_accumulation_steps=1,
        ),
        logging=LoggingConfig(backend="none", log_interval=1),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    main_batch = {"x": torch.randn(2, 3)}
    global_batch = {"x": torch.randn(2, 3)}

    def loss_step(module, batch):
        loss = module(batch["x"]).square().mean()
        return loss, {
            "loss_total": loss,
            "active_loss_global_grasp": loss.new_zeros(()),
        }

    def auxiliary_loss_step(module, batch):
        loss = module(batch["x"]).square().mean()
        return loss, {
            "loss_global_grasp": loss.detach(),
            "weighted_loss_global_grasp": loss.detach(),
            "active_loss_global_grasp": loss.new_ones(()),
        }

    Trainer(model, optimizer, config, loss_step).train(
        [main_batch],
        auxiliary_loader=[global_batch],
        auxiliary_loss_step=auxiliary_loss_step,
        auxiliary_weight=1.0,
    )
    record = json.loads((tmp_path / "train_metrics.jsonl").read_text().strip())
    assert record["active_loss_global_grasp"] == pytest.approx(1.0)
    assert record["global_states_seen"] == 2
    assert record["window_global_states"] == 2


def test_terminal_window_zero_fills_inactive_loss_contributions() -> None:
    summary = Trainer._summarize_terminal_window(
        [
            {
                "loss_total": 3.0,
                "loss_task_grasp": 2.0,
                "active_loss_task_grasp": 1.0,
            },
            {"loss_total": 1.0, "active_loss_task_grasp": 0.0},
        ]
    )
    assert summary["loss_total"] == pytest.approx(2.0)
    assert summary["loss_task_grasp"] == pytest.approx(1.0)
    assert summary["active_loss_task_grasp"] == pytest.approx(0.5)


def test_terminal_weighted_groups_reconcile_with_total_loss() -> None:
    records = [
        {
            "loss_total": 3.0,
            "weighted_loss_task_grasp": 2.0,
            "weighted_loss_push_object": 1.0,
        },
        {"loss_total": 1.0, "weighted_loss_push_object": 1.0},
    ]
    summary = Trainer._summarize_terminal_window(records)
    grouped = dict(Trainer._grouped_losses(summary))
    assert grouped == {"task_g": pytest.approx(1.0), "push": pytest.approx(1.0)}
    assert sum(grouped.values()) == pytest.approx(summary["loss_total"])


def test_validation_metrics_and_checkpoint_events_are_persisted(tmp_path, capsys) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=1,
            gradient_accumulation_steps=1,
            validation_interval=1,
        ),
        logging=LoggingConfig(backend="none", log_interval=10),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = {"x": torch.randn(2, 3)}

    def loss_step(module, current):
        loss = module(current["x"]).square().mean()
        return loss, {"loss_total": loss}

    trainer = Trainer(model, optimizer, config, loss_step)
    trainer.train(
        [batch],
        validate=lambda module: {
            "score_sum": 2.0,
            "score_count": 4,
            "metric_sums": {"loss_policy_candidate": 1.2},
            "metric_counts": {"loss_policy_candidate": 4},
        },
    )
    validation = json.loads((tmp_path / "validation_metrics.jsonl").read_text().strip())
    assert validation["validation_score"] == pytest.approx(0.5)
    assert validation["schema_version"] == 3
    assert validation["training_stage"] == "policy_teacher"
    assert validation["metrics"]["loss_policy_candidate"] == pytest.approx(0.3)
    assert validation["improved"]
    assert (tmp_path / "best.pt").is_file()
    terminal = capsys.readouterr().out
    assert "Val [policy_teacher] [0000001]" in terminal
    assert "score: 0.500000" in terminal
    assert "policy: 0.3000" in terminal
    assert "items: 4" in terminal
    assert "improved: yes" in terminal
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "training_events.jsonl").read_text().splitlines()
    ]
    assert "checkpoint_saved" in events
    assert "validation_completed" in events


def test_validation_overwrites_last_without_step_archives(tmp_path) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=2,
            gradient_accumulation_steps=1,
            validation_interval=1,
        ),
        logging=LoggingConfig(backend="none", log_interval=10),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = {"x": torch.randn(2, 3)}

    def loss_step(module, current):
        loss = module(current["x"]).square().mean()
        return loss, {"loss_total": loss}

    trainer = Trainer(model, optimizer, config, loss_step)
    finished_steps: list[int] = []
    trainer.train(
        [batch],
        validate=lambda module: (1.0, 1),
        step_finished=finished_steps.append,
    )

    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["trainer_state"]["optimizer_steps"] == 2
    assert finished_steps == [1, 2]
    assert (tmp_path / "best.pt").is_file()
    assert list(tmp_path.glob("step_*.pt")) == []


def test_terminal_summary_averages_interval_and_shows_generated_coverage(tmp_path, capsys) -> None:
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=3,
            gradient_accumulation_steps=1,
            generated_policy_candidate_cache="generated-cache",
            generated_policy_candidate_ratio=1.0,
        ),
        logging=LoggingConfig(backend="none", log_interval=3),
        output_dir=str(tmp_path),
    )
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = [{"x": torch.randn(2, 3), "marker": torch.tensor(value)} for value in (1.0, 2.0, 3.0)]

    def loss_step(module, batch):
        differentiable = module(batch["x"]).square().mean() * 0.0
        loss = differentiable + batch["marker"]
        return loss, {
            "loss_total": loss,
            "loss_policy_candidate": loss,
            "generated_states": loss.new_tensor(4.0),
            "generated_states_with_positive": loss.new_tensor(3.0),
            "generated_effective_policy_rows": loss.new_tensor(2.0),
            "generated_known_candidates": loss.new_tensor(10.0),
            "generated_unknown_candidates": loss.new_tensor(5.0),
            "generated_conflict_candidates": loss.new_tensor(1.0),
        }

    Trainer(model, optimizer, config, loss_step).train(batches)
    terminal = capsys.readouterr().out
    summaries = [line for line in terminal.splitlines() if line.startswith("Train ")]
    assert len(summaries) == 2
    assert "Train [policy_generated] [0000003/0000003]" in summaries[-1]
    assert "loss: 2.5000" in summaries[-1]
    assert "policy: 2.5000" in summaries[-1]
    assert "pos_cov: 75.0%" in summaries[-1]
    assert "eff_rows: 4/8" in summaries[-1]
    assert "known" not in summaries[-1]
    assert "unknown" not in summaries[-1]
    assert "conflict" not in summaries[-1]


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
        dataset_root.parent / "Grasp_20_class_object_3D_model" / "data" / "manual_function_regions"
    )
    adapter = TaskOrientedClutterAdapter(
        dataset_root,
        observation_provider=provider,
        point_count=256,
        functional_region_root=region_root,
    )
    batch = collate_unified([adapter.load_sample(1, 0, 0, 0)])
    config = ModelConfig(feature_dim=32, task_dim=16, activation_checkpointing=False)
    ablation = AblationConfig(use_gripper_scene_verifier=False)
    model = TCDPRGModel(
        config,
        ablation,
        GraphConfig(layers=1),
        RouterConfig(layers=1),
        BackboneConfig(backend="legacy", attention_points=64),
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
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_distributed_training_sampler_avoids_rank_overlap() -> None:
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    rank0 = set(DistributedWeightedStateSampler(weights, 0, 2, 4, seed=9))
    rank1 = set(DistributedWeightedStateSampler(weights, 1, 2, 4, seed=9))
    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == set(range(4))


def test_single_gpu_training_sampler_is_without_replacement() -> None:
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
    sampler = DistributedWeightedStateSampler(weights, 0, 1, len(weights), seed=9)
    first_epoch = list(sampler)
    assert len(first_epoch) == len(weights)
    assert len(set(first_epoch)) == len(weights)

    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert len(set(second_epoch)) == len(weights)
    assert second_epoch != first_epoch


def test_distributed_training_sampler_rejects_non_divisible_sample_count() -> None:
    weights = torch.ones(7, dtype=torch.double)
    with pytest.raises(ValueError, match="divisible by world_size"):
        DistributedWeightedStateSampler(weights, 0, 3, len(weights), seed=9)


def test_distributed_validation_sampler_has_no_padding_or_overlap() -> None:
    shards = [set(DistributedEvaluationSampler(7, rank, 3)) for rank in range(3)]
    assert set.union(*shards) == set(range(7))
    assert all(
        shards[left].isdisjoint(shards[right]) for left in range(3) for right in range(left + 1, 3)
    )
