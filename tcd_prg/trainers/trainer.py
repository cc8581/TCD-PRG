"""AMP/DDP-compatible trainer with state-group counters and exact resume."""

from __future__ import annotations

import json
import time
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import TCDPRGConfig

from .ema import ModelEMA
from .reproducibility import seed_everything


@dataclass(slots=True)
class TrainerState:
    optimizer_steps: int = 0
    samples_seen: int = 0
    states_seen: int = 0
    candidate_groups_seen: int = 0
    effective_epochs: float = 0.0
    best_validation: float = float("inf")
    validation_without_improvement: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TCDPRGConfig,
        loss_step: Callable[[nn.Module, Mapping[str, Any]], tuple[Tensor, Mapping[str, Tensor]]],
        scheduler: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.is_primary = self.rank == 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_step = loss_step
        self.state = TrainerState()
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.amp and self.device.type == "cuda")
        source_model = self.model.module if hasattr(self.model, "module") else self.model
        self.ema = ModelEMA(source_model, config.training.ema_decay) if config.training.ema_decay else None
        seed_everything(config.training.seed, config.training.deterministic)
        if self.is_primary:
            self._save_run_metadata()
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.tensorboard = None
        if self.is_primary and config.logging.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError as error:
                raise RuntimeError(
                    "logging.backend=tensorboard requires the tensorboard package"
                ) from error

    def _set_frozen_modules(self, frozen: bool) -> None:
        source = self.model.module if hasattr(self.model, "module") else self.model
        prefixes = tuple(self.config.training.frozen_modules)
        if not prefixes:
            return
        for name, parameter in source.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                parameter.requires_grad_(not frozen)

    def _save_run_metadata(self) -> None:
        import yaml

        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(asdict(self.config), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "uncommitted"
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps({"git_commit": commit, "torch": torch.__version__}, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _move(value: Any, device: torch.device) -> Any:
        if isinstance(value, Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {k: Trainer._move(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Trainer._move(v, device) for v in value)
        return value

    def train(
        self,
        loader: Iterable[Mapping[str, Any]],
        validate: Callable[[nn.Module], float] | None = None,
        groups_per_effective_epoch: int | None = None,
    ) -> TrainerState:
        accumulation = self.config.training.gradient_accumulation_steps
        unfreeze_step = self.config.training.unfreeze_at_optimizer_step
        self._set_frozen_modules(
            unfreeze_step is None or self.state.optimizer_steps < unfreeze_step
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        micro_step, saw_batch, stop = 0, False, False
        started = time.time()
        epoch = 0
        while self.state.optimizer_steps < self.config.training.max_optimizer_steps and not stop:
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            epoch += 1
            for raw_batch in loader:
                saw_batch = True
                micro_step += 1
                batch = self._move(raw_batch, self.device)
                autocast_enabled = self.config.training.amp and self.device.type == "cuda"
                amp_dtype = torch.bfloat16 if self.config.training.amp_dtype == "bfloat16" else torch.float16
                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled
                ):
                    loss, terms = self.loss_step(self.model, batch)
                    scaled_loss = loss / accumulation
                self.scaler.scale(scaled_loss).backward()
                batch_size = int(batch["xyz"].shape[0]) if "xyz" in batch else 1
                self.state.samples_seen += batch_size
                self.state.states_seen += batch_size
                self.state.candidate_groups_seen += batch_size
                if micro_step % accumulation:
                    continue
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.optimizer_steps += 1
                if (
                    self.config.training.unfreeze_at_optimizer_step is not None
                    and self.state.optimizer_steps >= self.config.training.unfreeze_at_optimizer_step
                ):
                    self._set_frozen_modules(False)
                if self.ema is not None:
                    source = self.model.module if hasattr(self.model, "module") else self.model
                    self.ema.update(source)
                if self.scheduler is not None:
                    self.scheduler.step()
                if groups_per_effective_epoch:
                    self.state.effective_epochs = (
                        self.state.candidate_groups_seen / groups_per_effective_epoch
                    )
                step = self.state.optimizer_steps
                if self.is_primary and (step % self.config.logging.log_interval == 0 or step == 1):
                    record = {
                        "optimizer_step": step,
                        "samples_seen": self.state.samples_seen,
                        "states_seen": self.state.states_seen,
                        "candidate_groups_seen": self.state.candidate_groups_seen,
                        "effective_epochs": self.state.effective_epochs,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": time.time() - started,
                        "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                        **{key: float(value.detach()) for key, value in terms.items()},
                    }
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if self.tensorboard is not None:
                        for key, value in record.items():
                            if isinstance(value, (int, float)):
                                self.tensorboard.add_scalar(key, value, step)
                        for index, rate in enumerate(record["learning_rates"]):
                            self.tensorboard.add_scalar(f"learning_rate/group_{index}", rate, step)
                if self.is_primary and step % self.config.training.checkpoint_interval == 0:
                    self.save_checkpoint(self.output_dir / f"step_{step:08d}.pt")
                if validate and step % self.config.training.validation_interval == 0:
                    score = float(validate(self.ema.model if self.ema else self.model))
                    if torch.distributed.is_initialized():
                        value = torch.tensor(score, device=self.device)
                        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                        score = float(value / torch.distributed.get_world_size())
                    if score < self.state.best_validation:
                        self.state.best_validation = score
                        self.state.validation_without_improvement = 0
                        if self.is_primary:
                            self.save_checkpoint(self.output_dir / "best.pt")
                    else:
                        self.state.validation_without_improvement += 1
                    if (
                        self.state.validation_without_improvement
                        >= self.config.training.early_stopping_patience
                    ):
                        stop = True
                        break
                if step >= self.config.training.max_optimizer_steps:
                    stop = True
                    break
            if not saw_batch:
                raise RuntimeError("Training loader yielded no batches")
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        return self.state

    def save_checkpoint(self, path: str | Path) -> None:
        if not self.is_primary:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        source = self.model.module if hasattr(self.model, "module") else self.model
        payload = {
            "schema_version": 1,
            "model": source.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.model.state_dict() if self.ema else None,
            "trainer_state": asdict(self.state),
            "config": asdict(self.config),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        source = self.model.module if hasattr(self.model, "module") else self.model
        source.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler and payload["scheduler"] is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        if self.ema and payload["ema"] is not None:
            self.ema.model.load_state_dict(payload["ema"])
        self.state = TrainerState(**payload["trainer_state"])
        torch.set_rng_state(payload["rng_cpu"])
        if torch.cuda.is_available() and payload["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
