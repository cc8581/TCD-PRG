"""AMP/DDP-compatible trainer with state-group counters and exact resume."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    amp_skipped_steps: int = 0
    samples_seen: int = 0
    states_seen: int = 0
    candidate_groups_seen: int = 0
    generated_states_seen: int = 0
    generated_positive_states_seen: int = 0
    effective_policy_rows_seen: int = 0
    effective_epochs: float = 0.0
    best_validation: float = float("inf")
    validation_without_improvement: int = 0


class Trainer:
    COUNT_TERMS = {
        "generated_states",
        "generated_states_with_positive",
        "generated_effective_policy_rows",
        "generated_known_candidates",
        "generated_unknown_candidates",
        "generated_conflict_candidates",
    }

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
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed_everything(config.training.seed + self.rank, config.training.deterministic)
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.is_primary = self.rank == 0
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_step = loss_step
        self.state = TrainerState()
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.validation_metrics_path = self.output_dir / "validation_metrics.jsonl"
        self.events_path = self.output_dir / "training_events.jsonl"
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.amp and self.device.type == "cuda")
        source_model = self.model.module if hasattr(self.model, "module") else self.model
        self.ema = ModelEMA(source_model, config.training.ema_decay) if config.training.ema_decay else None
        if self.is_primary:
            self._save_run_metadata()
        self.tensorboard = None
        if self.is_primary and config.logging.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError as error:
                raise RuntimeError(
                    "logging.backend=tensorboard requires the tensorboard package"
                ) from error

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    def _write_event(self, event: str, **values: Any) -> None:
        if self.is_primary:
            self._append_jsonl(self.events_path, {
                "timestamp_utc": self._timestamp(),
                "event": event,
                "optimizer_step": self.state.optimizer_steps,
                **values,
            })

    @classmethod
    def _aggregate_window_terms(
        cls, sums: Mapping[str, Tensor], counts: Mapping[str, int]
    ) -> dict[str, float]:
        return {
            key: float(value.cpu())
            if key in cls.COUNT_TERMS
            else float((value / counts[key]).cpu())
            for key, value in sums.items()
        }

    def _print_train_summary(self, record: Mapping[str, Any]) -> None:
        step = int(record["optimizer_step"])
        maximum = self.config.training.max_optimizer_steps
        status = (
            f"[train {step:07d}/{maximum:07d}] "
            f"epoch={float(record['effective_epochs']):.3f} "
            f"loss={float(record['loss_total']):.5f} "
            f"lr={max(record['learning_rates']):.3e} "
            f"grad={float(record['gradient_norm']):.3f} "
            f"samples/s={float(record['samples_per_second']):.1f} "
            f"amp_skip={int(record['amp_skipped_steps'])}"
        )
        print(status, flush=True)
        names = (
            ("region", "loss_region"),
            ("task_g", "loss_task_grasp"),
            ("global_g", "loss_global_grasp"),
            ("graph_p", "loss_physical_edge"),
            ("graph_t", "loss_task_edge"),
            ("verify", "loss_verify_overall"),
            ("push_o", "loss_push_object"),
            ("push_c", "loss_push_contact"),
            ("push_d", "loss_push_direction"),
            ("push_u", "loss_push_potential"),
            ("policy", "loss_policy_candidate"),
        )
        losses = [
            f"{label}={float(record[key]):.4f}"
            for label, key in names if key in record
        ]
        if losses:
            print("  losses: " + " ".join(losses), flush=True)
        generated = float(record.get("generated_states", 0.0))
        if generated:
            positive = float(record.get("generated_states_with_positive", 0.0))
            effective = float(record.get("generated_effective_policy_rows", 0.0))
            known = float(record.get("generated_known_candidates", 0.0))
            unknown = float(record.get("generated_unknown_candidates", 0.0))
            conflicts = float(record.get("generated_conflict_candidates", 0.0))
            print(
                "  generated: "
                f"positive={positive / generated:.1%} "
                f"effective={effective / generated:.1%} "
                f"known={known:.0f} unknown={unknown:.0f} conflicts={conflicts:.0f}",
                flush=True,
            )

    def _set_frozen_modules(self, frozen: bool) -> None:
        source = self.model.module if hasattr(self.model, "module") else self.model
        prefixes = tuple(self.config.training.frozen_modules)
        if not prefixes:
            return
        for name, parameter in source.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                parameter.requires_grad_(not frozen)

    def _apply_frozen_module_modes(self) -> None:
        """Keep frozen upstream modules deterministic during router-only stages."""

        source = self.model.module if hasattr(self.model, "module") else self.model
        for prefix in self.config.training.frozen_modules:
            module = source.get_submodule(prefix)
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()

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
            json.dumps({
                "git_commit": commit,
                "torch": torch.__version__,
                "train_metrics_schema_version": 2,
                "validation_metrics_schema_version": 1,
                "training_events_schema_version": 1,
            }, indent=2),
            encoding="utf-8",
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

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        if "xyz" in batch and isinstance(batch["xyz"], Tensor):
            return int(batch["xyz"].shape[0])
        for value in batch.values():
            if isinstance(value, Tensor) and value.ndim:
                return int(value.shape[0])
        return 1

    def train(
        self,
        loader: Iterable[Mapping[str, Any]],
        validate: Callable[[nn.Module], Any] | None = None,
        groups_per_effective_epoch: int | None = None,
    ) -> TrainerState:
        accumulation = self.config.training.gradient_accumulation_steps
        unfreeze_step = self.config.training.unfreeze_at_optimizer_step
        self._set_frozen_modules(
            unfreeze_step is None or self.state.optimizer_steps < unfreeze_step
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self._apply_frozen_module_modes()
        micro_step, saw_batch, stop = 0, False, False
        started = time.time()
        last_optimizer_time = started
        window_term_sums: dict[str, Tensor] = {}
        window_term_counts: dict[str, int] = {}
        window_micro_batches = 0
        window_samples = 0
        epoch = 0
        self._write_event(
            "training_started",
            max_optimizer_steps=self.config.training.max_optimizer_steps,
            gradient_accumulation_steps=accumulation,
        )
        if self.is_primary:
            print(
                f"[train-start] output={self.output_dir.resolve()} "
                f"steps={self.state.optimizer_steps}->{self.config.training.max_optimizer_steps} "
                f"accumulation={accumulation} "
                f"terminal_interval={self.config.logging.log_interval}",
                flush=True,
            )
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
                synchronize = micro_step % accumulation == 0
                sync_context = (
                    nullcontext()
                    if synchronize or not hasattr(self.model, "no_sync")
                    else self.model.no_sync()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled
                    ):
                        loss, terms = self.loss_step(self.model, batch)
                        scaled_loss = loss / accumulation
                    self.scaler.scale(scaled_loss).backward()
                if "generated_states" in terms:
                    generated_counts = torch.stack([
                        terms["generated_states"],
                        terms["generated_states_with_positive"],
                        terms["generated_effective_policy_rows"],
                    ]).detach().to(dtype=torch.long)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            generated_counts, op=torch.distributed.ReduceOp.SUM
                        )
                    self.state.generated_states_seen += int(generated_counts[0])
                    self.state.generated_positive_states_seen += int(generated_counts[1])
                    self.state.effective_policy_rows_seen += int(generated_counts[2])
                batch_size = self._batch_size(batch)
                global_batch_size = batch_size * (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized() else 1
                )
                self.state.samples_seen += global_batch_size
                self.state.states_seen += global_batch_size
                self.state.candidate_groups_seen += global_batch_size
                window_micro_batches += 1
                window_samples += global_batch_size
                window_values = dict(terms)
                window_values.setdefault("loss_total", loss.detach())
                for key, value in window_values.items():
                    detached = value.detach().float()
                    window_term_sums[key] = window_term_sums.get(
                        key, torch.zeros_like(detached)
                    ) + detached
                    window_term_counts[key] = window_term_counts.get(key, 0) + 1
                if not synchronize:
                    continue
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                previous_scale = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                current_scale = float(self.scaler.get_scale())
                self.optimizer.zero_grad(set_to_none=True)
                # GradScaler deliberately skips optimizer.step when any
                # unscaled gradient is non-finite.  Such an overflow is a
                # micro-step retry, not a completed optimizer step: scheduler,
                # EMA, counters and checkpoints must remain unchanged.
                if self.scaler.is_enabled() and current_scale < previous_scale:
                    self.state.amp_skipped_steps += 1
                    skipped_terms = self._aggregate_window_terms(
                        window_term_sums, window_term_counts
                    )
                    self._write_event(
                        "amp_step_skipped",
                        previous_amp_scale=previous_scale,
                        current_amp_scale=current_scale,
                        gradient_norm=float(gradient_norm),
                        micro_batches=window_micro_batches,
                        samples=window_samples,
                        metrics=skipped_terms,
                    )
                    if self.is_primary:
                        print(
                            f"[amp-skip step={self.state.optimizer_steps:07d}] "
                            f"scale={previous_scale:.0f}->{current_scale:.0f} "
                            f"grad={float(gradient_norm):.3f}",
                            flush=True,
                        )
                    if self.tensorboard is not None:
                        self.tensorboard.add_scalar(
                            "amp/skipped_steps", self.state.amp_skipped_steps,
                            self.state.optimizer_steps,
                        )
                        self.tensorboard.add_scalar(
                            "amp/scale", current_scale, self.state.optimizer_steps
                        )
                    window_term_sums.clear()
                    window_term_counts.clear()
                    window_micro_batches = 0
                    window_samples = 0
                    last_optimizer_time = time.time()
                    continue
                self.state.optimizer_steps += 1
                if (
                    self.config.training.unfreeze_at_optimizer_step is not None
                    and self.state.optimizer_steps >= self.config.training.unfreeze_at_optimizer_step
                ):
                    self._set_frozen_modules(False)
                    self.model.train()
                    self._apply_frozen_module_modes()
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
                now = time.time()
                step_seconds = max(now - last_optimizer_time, 1e-12)
                aggregated_terms = self._aggregate_window_terms(
                    window_term_sums, window_term_counts
                )
                if self.is_primary:
                    record = {
                        "schema_version": 2,
                        "timestamp_utc": self._timestamp(),
                        "optimizer_step": step,
                        "amp_skipped_steps": self.state.amp_skipped_steps,
                        "amp_scale": current_scale,
                        "samples_seen": self.state.samples_seen,
                        "states_seen": self.state.states_seen,
                        "candidate_groups_seen": self.state.candidate_groups_seen,
                        "generated_states_seen": self.state.generated_states_seen,
                        "generated_positive_states_seen": self.state.generated_positive_states_seen,
                        "effective_policy_rows_seen": self.state.effective_policy_rows_seen,
                        "effective_epochs": self.state.effective_epochs,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": now - started,
                        "optimizer_step_seconds": step_seconds,
                        "samples_per_second": window_samples / step_seconds,
                        "micro_batches": window_micro_batches,
                        "window_samples": window_samples,
                        "metric_scope": "rank0",
                        "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                        **aggregated_terms,
                    }
                    self._append_jsonl(self.metrics_path, record)
                    if self.tensorboard is not None:
                        for key, value in record.items():
                            if isinstance(value, (int, float)):
                                self.tensorboard.add_scalar(key, value, step)
                        for index, rate in enumerate(record["learning_rates"]):
                            self.tensorboard.add_scalar(f"learning_rate/group_{index}", rate, step)
                    if step % self.config.logging.log_interval == 0 or step == 1:
                        self._print_train_summary(record)
                window_term_sums.clear()
                window_term_counts.clear()
                window_micro_batches = 0
                window_samples = 0
                last_optimizer_time = now
                if step % self.config.training.checkpoint_interval == 0:
                    self.save_checkpoint(self.output_dir / f"step_{step:08d}.pt")
                if validate and step % self.config.training.validation_interval == 0:
                    validation = validate(self.ema.model if self.ema else self.model)
                    self.model.train()
                    self._apply_frozen_module_modes()
                    validation_items = 1
                    validation_details: dict[str, float] = {}
                    if isinstance(validation, Mapping):
                        summaries: list[Mapping[str, Any]] = [validation]
                        if torch.distributed.is_initialized():
                            gathered: list[Mapping[str, Any] | None] = [
                                None for _ in range(torch.distributed.get_world_size())
                            ]
                            torch.distributed.all_gather_object(gathered, validation)
                            summaries = [item for item in gathered if item is not None]
                        score_sum = sum(float(item["score_sum"]) for item in summaries)
                        validation_items = sum(
                            int(item["score_count"]) for item in summaries
                        )
                        score = score_sum / max(1, validation_items)
                        metric_sums: dict[str, float] = {}
                        metric_counts: dict[str, int] = {}
                        for item in summaries:
                            for key, value in item.get("metric_sums", {}).items():
                                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                            for key, value in item.get("metric_counts", {}).items():
                                metric_counts[key] = metric_counts.get(key, 0) + int(value)
                        validation_details = {
                            key: value / max(1, metric_counts.get(key, 0))
                            for key, value in metric_sums.items()
                        }
                    elif torch.distributed.is_initialized():
                        if isinstance(validation, tuple):
                            value = torch.tensor(validation, dtype=torch.float64, device=self.device)
                            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                            score = float(value[0] / value[1].clamp_min(1))
                            validation_items = int(value[1])
                        else:
                            value = torch.tensor(float(validation), device=self.device)
                            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                            score = float(value / torch.distributed.get_world_size())
                    elif isinstance(validation, tuple):
                        score = float(validation[0] / max(1, validation[1]))
                        validation_items = int(validation[1])
                    else:
                        score = float(validation)
                    previous_best = self.state.best_validation
                    improved = score < previous_best
                    if improved:
                        self.state.best_validation = score
                        self.state.validation_without_improvement = 0
                        self.save_checkpoint(self.output_dir / "best.pt")
                    else:
                        self.state.validation_without_improvement += 1
                    if self.is_primary:
                        validation_record = {
                            "schema_version": 1,
                            "timestamp_utc": self._timestamp(),
                            "optimizer_step": step,
                            "validation_score": score,
                            "best_validation": self.state.best_validation,
                            "improved": improved,
                            "validation_items": validation_items,
                            "validation_without_improvement": (
                                self.state.validation_without_improvement
                            ),
                            "early_stopping_patience": (
                                self.config.training.early_stopping_patience
                            ),
                            "metrics": validation_details,
                        }
                        self._append_jsonl(
                            self.validation_metrics_path, validation_record
                        )
                        if self.tensorboard is not None:
                            self.tensorboard.add_scalar(
                                "validation/score", score, step
                            )
                            self.tensorboard.add_scalar(
                                "validation/best", self.state.best_validation, step
                            )
                            for key, value in validation_details.items():
                                self.tensorboard.add_scalar(
                                    f"validation/{key}", value, step
                                )
                        print(
                            f"[valid {step:07d}] score={score:.6f} "
                            f"best={self.state.best_validation:.6f} "
                            f"improved={'yes' if improved else 'no'} "
                            f"patience={self.state.validation_without_improvement}/"
                            f"{self.config.training.early_stopping_patience}",
                            flush=True,
                        )
                    self._write_event(
                        "validation_completed",
                        validation_score=score,
                        best_validation=self.state.best_validation,
                        improved=improved,
                        validation_items=validation_items,
                        metrics=validation_details,
                    )
                    if (
                        self.state.validation_without_improvement
                        >= self.config.training.early_stopping_patience
                    ):
                        self._write_event(
                            "early_stopping",
                            best_validation=self.state.best_validation,
                            validation_score=score,
                        )
                        if self.is_primary:
                            print(
                                f"[early-stop] step={step:07d} "
                                f"best={self.state.best_validation:.6f}",
                                flush=True,
                            )
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
        self._write_event(
            "training_completed",
            best_validation=self.state.best_validation,
            stopped_early=stop and self.state.optimizer_steps < self.config.training.max_optimizer_steps,
        )
        if self.is_primary:
            print(
                f"[train-done] step={self.state.optimizer_steps:07d} "
                f"best={self.state.best_validation:.6f} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        return self.state

    def save_checkpoint(self, path: str | Path) -> None:
        local_rng = {
            "cpu": torch.get_rng_state().cpu(),
            "cuda": (
                torch.cuda.get_rng_state(self.device).cpu()
                if self.device.type == "cuda" else None
            ),
        }
        rng_by_rank: list[dict[str, Tensor]] | None = None
        if torch.distributed.is_initialized():
            rng_by_rank = [None] * torch.distributed.get_world_size() if self.is_primary else None  # type: ignore[list-item]
            torch.distributed.gather_object(local_rng, rng_by_rank, dst=0)
        else:
            rng_by_rank = [local_rng]
        if not self.is_primary:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        source = self.model.module if hasattr(self.model, "module") else self.model
        payload = {
            "schema_version": 8,
            "model": source.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.model.state_dict() if self.ema else None,
            "trainer_state": asdict(self.state),
            "config": asdict(self.config),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_by_rank": rng_by_rank,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self._write_event(
            "checkpoint_saved",
            path=str(path.resolve()),
            schema_version=payload["schema_version"],
        )

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != 8:
            raise RuntimeError(
                "Unsupported TCD-PRG checkpoint schema "
                f"{schema_version}; this code expects schema 8. Per-direction PUSH residuals, "
                "Top-M direction candidates, and generated-candidate policy training require a "
                "fresh checkpoint. Complete query-based task/global "
                "grasp sets and the eleven-objective training contract changed. Load the "
                "original GAPG encoder through the pretrained-backbone option, "
                "or start a new TCD-PRG run instead of resuming this checkpoint."
            )
        source = self.model.module if hasattr(self.model, "module") else self.model
        source.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler and payload["scheduler"] is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        if self.ema and payload["ema"] is not None:
            self.ema.model.load_state_dict(payload["ema"])
        self.state = TrainerState(**payload["trainer_state"])
        # ``map_location=self.device`` also moves serialized RNG byte tensors
        # to CUDA.  PyTorch's RNG restoration APIs require CPU ByteTensors even
        # when restoring the per-device CUDA generators.
        per_rank = payload.get("rng_by_rank")
        if per_rank and self.rank < len(per_rank):
            torch.set_rng_state(per_rank[self.rank]["cpu"].cpu())
            cuda_rng = per_rank[self.rank].get("cuda")
            if self.device.type == "cuda" and cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng.cpu(), self.device)
        else:
            torch.set_rng_state(payload["rng_cpu"].cpu())
            if torch.cuda.is_available() and payload["rng_cuda"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng_cuda"]])
