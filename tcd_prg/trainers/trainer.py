"""AMP/DDP-compatible trainer with state-group counters and exact resume."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tcd_prg.config import TCDPRGConfig
from tcd_prg.evaluators.offline import OfflineModelEvaluator
from tcd_prg.losses.task_grasp_binary import stageb_split_metrics
from tcd_prg.datasets.stageb_manifest import compatibility_provenance

from .ema import ModelEMA
from .reproducibility import seed_everything


def aggregate_stageb_validation_payloads(
    summaries: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute nonlinear Stage-B metrics once over all gathered DDP ranks."""
    scores = [np.asarray(item["stageb_scores"]) for item in summaries if "stageb_scores" in item]
    targets = [np.asarray(item["stageb_targets"]) for item in summaries if "stageb_targets" in item]
    if not scores:
        return {}
    if len(scores) != len(targets):
        raise RuntimeError("Incomplete distributed Stage-B validation payload")
    return stageb_split_metrics(np.concatenate(scores), np.concatenate(targets))


@dataclass(slots=True)
class TrainerState:
    # optimizer_steps 只统计成功更新；samples/states/groups 用于跨阶段核对实际数据覆盖。
    optimizer_steps: int = 0
    amp_skipped_steps: int = 0
    samples_seen: int = 0
    states_seen: int = 0
    candidate_groups_seen: int = 0
    global_states_seen: int = 0
    effective_epochs: float = 0.0
    best_validation: float = float("inf")
    validation_without_improvement: int = 0
    # Validation is a transaction tied to an optimizer step. The pending flag
    # is persisted before validation starts so resume can finish validation
    # before consuming the next training batch.
    last_completed_validation_step: int = 0
    pending_validation_step: int = 0


class Trainer:
    COUNT_TERMS = {
        "task_grasp_supervised_candidates",
        "push_object_effective_rows",
        "push_multiobject_states",
        "push_positive_objects",
        "push_negative_objects",
        "push_contact_positive_points",
        "push_contact_negative_points",
        "push_contact_valid_points",
        "push_direction_effective_rows",
        "push_direction_residual_targets",
        "push_potential_valid_candidates",
    }
    LOSS_GROUPS = (
        ("instance", ("weighted_loss_instance",)),
        ("region", ("weighted_loss_region",)),
        ("task_g", ("weighted_loss_task_grasp",)),
        (
            "push",
            (
                "weighted_loss_push_object",
                "weighted_loss_push_contact",
                "weighted_loss_push_direction",
                "weighted_loss_push_potential",
            ),
        ),
    )
    TERMINAL_AVERAGE_TERMS = {
        "gradient_norm",
        "gradient_norm_after_clip",
        "gradient_clip_scale",
        "gradient_clipped",
        "optimizer_step_seconds",
        "data_seconds",
        "samples_per_second",
        "task_grasp_binary_accuracy",
        "task_grasp_binary_f1",
        "task_grasp_positive_fraction",
    }
    GRADIENT_GROUPS = {
        "encoder": ("encoder",),
        "region": ("region_head",),
        "grasp": ("task_grasp",),
        "push": ("push",),
    }

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TCDPRGConfig,
        loss_step: Callable[[nn.Module, Mapping[str, Any]], tuple[Tensor, Mapping[str, Tensor]]],
        scheduler: Any | None = None,
        output_dir: str | Path | None = None,
        stageb_provenance: Mapping[str, Any] | None = None,
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
        self.task_grasp_probability_threshold = float(config.model.task_grasp_probability_threshold)
        self.stageb_provenance = dict(stageb_provenance) if stageb_provenance else None
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # JSONL 是逐步完整记录，终端输出只是低频核心摘要，两者职责不同。
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.validation_metrics_path = self.output_dir / "validation_metrics.jsonl"
        self.events_path = self.output_dir / "training_events.jsonl"
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config.training.amp and self.device.type == "cuda",
            init_scale=config.training.amp_initial_scale,
        )
        source_model = self.model.module if hasattr(self.model, "module") else self.model
        self.ema = (
            ModelEMA(source_model, config.training.ema_decay) if config.training.ema_decay else None
        )
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
            self._append_jsonl(
                self.events_path,
                {
                    "timestamp_utc": self._timestamp(),
                    "event": event,
                    "optimizer_step": self.state.optimizer_steps,
                    **values,
                },
            )

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

    def _reduce_distributed_terms(self, terms: Mapping[str, float]) -> dict[str, float]:
        """Return world-mean metrics and world-sum counters on every rank."""

        # loss/速率在 rank 间取均值，样本和候选数量在 rank 间求和。
        reduced = dict(terms)
        if not torch.distributed.is_initialized() or not reduced:
            return reduced
        keys = sorted(reduced)
        values = torch.tensor(
            [reduced[key] for key in keys], dtype=torch.float64, device=self.device
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        return {
            key: float(values[index])
            if key in self.COUNT_TERMS
            else float(values[index] / world_size)
            for index, key in enumerate(keys)
        }

    def _training_stage(self, record: Mapping[str, Any]) -> str:
        has_grasp = "loss_task_grasp" in record
        has_push = any(
            f"loss_{name}" in record
            for name in ("push_object", "push_contact", "push_direction", "push_potential")
        )
        if has_grasp and has_push:
            return "full"
        if has_push:
            return "push"
        if has_grasp:
            return "grasp"
        return "perception"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return str(timedelta(seconds=max(0, int(seconds))))

    @classmethod
    def _grouped_losses(cls, record: Mapping[str, Any]) -> list[tuple[str, float]]:
        grouped: list[tuple[str, float]] = []
        for label, keys in cls.LOSS_GROUPS:
            values = [float(record[key]) for key in keys if key in record]
            if not values:
                fallback = tuple(key.removeprefix("weighted_") for key in keys)
                values = [float(record[key]) for key in fallback if key in record]
            if values:
                grouped.append((label, sum(values)))
        return grouped

    @staticmethod
    def _group_coverage(record: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
        values = [
            float(record[f"active_{key.removeprefix('weighted_')}"])
            for key in keys
            if f"active_{key.removeprefix('weighted_')}" in record
        ]
        return sum(values) / len(values) if values else None

    @classmethod
    def _summarize_terminal_window(cls, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Average live metrics while preserving exact counters and latest state."""

        summary = dict(records[-1])
        keys = set().union(*(record.keys() for record in records))
        for key in keys:
            values = [
                float(record[key])
                for record in records
                if key in record and isinstance(record[key], (int, float))
            ]
            if not values:
                continue
            if key in cls.COUNT_TERMS:
                summary[key] = sum(values)
            elif key.startswith("loss_") or key.startswith("weighted_loss_"):
                # 缺失监督的 family 对总损失贡献为零；终端也必须按完整窗口平均。
                summary[key] = sum(values) / len(records)
            elif (
                key.startswith("active_loss_")
                or key.startswith("gradient_")
                or key in cls.TERMINAL_AVERAGE_TERMS
            ):
                summary[key] = sum(values) / len(values)
        return summary

    def _print_train_summary(self, record: Mapping[str, Any]) -> None:
        step = int(record["optimizer_step"])
        maximum = self.config.training.max_optimizer_steps
        fields = [
            f"Train [{self._training_stage(record)}] [{step:07d}/{maximum:07d}]",
            f"eta: {self._format_duration(float(record['eta_seconds']))}",
            f"loss: {float(record['loss_total']):.4f}",
        ]
        group_keys = dict(self.LOSS_GROUPS)
        for label, value in self._grouped_losses(record):
            keys = group_keys[label]
            coverage = self._group_coverage(record, keys)
            suffix = f"({coverage:.0%})" if coverage is not None and coverage < 1.0 else ""
            fields.append(f"{label}: {value:.4f}{suffix}")
        if "loss_task_grasp" in record:
            fields.extend(
                (
                    f"sup: {float(record.get('active_loss_task_grasp', 0.0)):.0%}",
                    f"acc: {float(record.get('task_grasp_binary_accuracy', 0.0)):.1%}",
                    f"f1: {float(record.get('task_grasp_binary_f1', 0.0)):.1%}",
                    f"pos: {float(record.get('task_grasp_positive_fraction', 0.0)):.1%}",
                )
            )
        fields.extend(
            (
                f"lr: {max(record['learning_rates']):.3e}",
                f"grad: {float(record['gradient_norm']):.3f}"
                f"->{float(record.get('gradient_norm_after_clip', record['gradient_norm'])):.3f}",
                f"clip: {float(record.get('gradient_clip_scale', 1.0)):.3f}",
                f"time: {float(record['optimizer_step_seconds']):.3f}",
                f"data: {float(record['data_seconds']):.3f}",
            )
        )
        if float(record.get("max_memory_mb", 0.0)) > 0.0:
            fields.append(f"max mem: {float(record['max_memory_mb']):.0f}M")
        print("  ".join(fields), flush=True)

    def _print_validation_summary(self, record: Mapping[str, Any]) -> None:
        fields = [
            f"Val [{self._training_stage(record['metrics'])}] "
            f"[{int(record['optimizer_step']):07d}]",
            f"score: {float(record['validation_score']):.6f}",
            f"best: {float(record['best_validation']):.6f}",
        ]
        fields.extend(
            f"{label}: {value:.4f}" for label, value in self._grouped_losses(record["metrics"])
        )
        metrics = record["metrics"]
        if "standard_region_miou" in metrics:
            fields.append(f"mIoU: {float(metrics['standard_region_miou']):.1%}")
        print("  ".join(fields), flush=True)

    # validation-resume-transaction-v1
    def _validation_record_for_step(self, step: int) -> dict[str, Any] | None:
        # Return the last successfully written validation record for this step.
        if not self.validation_metrics_path.is_file():
            return None
        found: dict[str, Any] | None = None
        with self.validation_metrics_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Malformed validation_metrics.jsonl line {line_number}: "
                        f"{self.validation_metrics_path}"
                    ) from error
                if int(record.get("optimizer_step", -1)) == int(step):
                    found = record
        return found

    def _recover_completed_validation(self, step: int) -> bool:
        # Recover post-validation state from the durable validation JSONL.
        record = self._validation_record_for_step(step)
        if record is None:
            return False
        self.state.best_validation = float(
            record.get("best_validation", self.state.best_validation)
        )
        self.state.validation_without_improvement = int(
            record.get(
                "validation_without_improvement",
                self.state.validation_without_improvement,
            )
        )
        self.state.last_completed_validation_step = max(
            int(self.state.last_completed_validation_step), int(step)
        )
        self.state.pending_validation_step = 0
        self._write_event(
            "validation_resume_recovered",
            validation_step=int(step),
            best_validation=self.state.best_validation,
            validation_without_improvement=self.state.validation_without_improvement,
        )
        self.save_checkpoint(self.output_dir / "last.pt")
        return True

    def _resume_validation_due(self) -> bool:
        # True means validate before consuming the next training batch.
        step = int(self.state.optimizer_steps)
        interval = int(self.config.training.validation_interval)
        if step <= 0 or interval <= 0 or step % interval != 0:
            # Do not back-fill historical validation points with a newer model.
            return False

        pending = int(self.state.pending_validation_step)
        if pending not in (0, step):
            raise RuntimeError(
                "Checkpoint validation transaction is inconsistent: "
                f"optimizer_step={step}, pending_validation_step={pending}"
            )
        if pending == 0 and int(self.state.last_completed_validation_step) >= step:
            return False

        # Old checkpoints do not contain the transaction fields. If a successful
        # record exists, recover it; otherwise this exact boundary is pending.
        if self._recover_completed_validation(step):
            return False

        self.state.pending_validation_step = step
        self._write_event("validation_resume_pending", validation_step=step)
        return True

    def _run_validation_cycle(
        self,
        validate: Callable[[nn.Module], Any],
        step: int,
        *,
        resumed: bool,
    ) -> bool:
        # Run one validation transaction; return True if early stopping fires.
        step = int(step)
        interval = int(self.config.training.validation_interval)
        if step != int(self.state.optimizer_steps):
            raise RuntimeError(
                "Validation step must equal the current optimizer step: "
                f"validation={step}, optimizer={self.state.optimizer_steps}"
            )
        if step <= 0 or interval <= 0 or step % interval != 0:
            raise RuntimeError(
                f"Invalid validation transaction boundary: step={step}, interval={interval}"
            )

        # BEGIN: durable checkpoint explicitly says this validation is pending.
        self.state.pending_validation_step = step
        self.save_checkpoint(self.output_dir / "last.pt")
        self._write_event(
            "validation_started",
            validation_step=step,
            resumed=bool(resumed),
        )

        validation = validate(self.ema.model if self.ema else self.model)
        self.model.train()
        self._apply_frozen_module_modes()

        validation_items = 1
        validation_details: dict[str, float] = {}
        performance_summary: dict[str, Any] = {
            "count": 0,
            "scene_count": 0,
            "metrics": {},
        }
        if isinstance(validation, Mapping):
            summaries: list[Mapping[str, Any]] = [validation]
            if torch.distributed.is_initialized():
                gathered: list[Mapping[str, Any] | None] = [
                    None for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather_object(gathered, validation)
                summaries = [item for item in gathered if item is not None]
            score_sum = sum(float(item["score_sum"]) for item in summaries)
            validation_items = sum(int(item["score_count"]) for item in summaries)
            score = score_sum / max(1, validation_items)
            metric_sums: dict[str, float] = {}
            metric_counts: dict[str, int] = {}
            for item in summaries:
                for key, value in item.get("metric_sums", {}).items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                for key, value in item.get("metric_counts", {}).items():
                    metric_counts[key] = metric_counts.get(key, 0) + int(value)
            validation_details = {
                key: (
                    value if key in self.COUNT_TERMS else value / max(1, metric_counts.get(key, 0))
                )
                for key, value in metric_sums.items()
            }
            validation_details.update(aggregate_stageb_validation_payloads(summaries))
            evaluation_records = [
                record for item in summaries for record in item.get("evaluation_records", [])
            ]
            if evaluation_records:
                evaluator = OfflineModelEvaluator(
                    self.config.model,
                    self.config.evaluation.bootstrap_samples,
                    self.config.evaluation.confidence,
                    self.config.evaluation,
                )
                evaluator.evaluator.records = evaluation_records
                performance_summary = evaluator.summarize()
                validation_details.update(
                    {
                        key: float(payload["mean"])
                        for key, payload in performance_summary["metrics"].items()
                        if payload.get("mean") is not None
                    }
                )
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
        else:
            self.state.validation_without_improvement += 1

        # COMMIT state is established before the durable validation record.
        self.state.last_completed_validation_step = step
        self.state.pending_validation_step = 0

        if improved and "task_grasp_validation_threshold" in validation_details:
            self.task_grasp_probability_threshold = float(
                validation_details["task_grasp_validation_threshold"]
            )
            self.config.model.task_grasp_probability_threshold = (
                self.task_grasp_probability_threshold
            )
        if improved:
            self.save_checkpoint(self.output_dir / "best.pt")

        if self.is_primary:
            if improved and "task_grasp_validation_threshold" in validation_details:
                threshold_path = self.output_dir / "stageb_decision_threshold.json"
                temporary = threshold_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "optimizer_step": step,
                            "threshold": validation_details["task_grasp_validation_threshold"],
                            "f1": validation_details.get("task_grasp_validation_f1"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(threshold_path)
            validation_record = {
                "schema_version": 3,
                "timestamp_utc": self._timestamp(),
                "optimizer_step": step,
                "validation_score": score,
                "best_validation": self.state.best_validation,
                "improved": improved,
                "validation_items": validation_items,
                "validation_without_improvement": self.state.validation_without_improvement,
                "early_stopping_patience": self.config.training.early_stopping_patience,
                "metrics": validation_details,
                "performance": performance_summary,
            }
            validation_record["training_stage"] = self._training_stage(validation_details)
            self._append_jsonl(self.validation_metrics_path, validation_record)
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("validation/score", score, step)
                self.tensorboard.add_scalar("validation/best", self.state.best_validation, step)
                for key, value in validation_details.items():
                    self.tensorboard.add_scalar(f"validation/{key}", value, step)
            self._print_validation_summary(validation_record)

        self._write_event(
            "validation_completed",
            validation_step=step,
            validation_score=score,
            best_validation=self.state.best_validation,
            improved=improved,
            validation_items=validation_items,
            metrics=validation_details,
        )
        self.save_checkpoint(self.output_dir / "last.pt")

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
                    f"[early-stop] step={step:07d} best={self.state.best_validation:.6f}",
                    flush=True,
                )
            return True
        return False

    def _set_frozen_modules(self, frozen: bool) -> None:
        source = self.model.module if hasattr(self.model, "module") else self.model
        prefixes = tuple(self.config.training.frozen_modules)
        if not prefixes:
            return
        for name, parameter in source.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                parameter.requires_grad_(not frozen)

    def _apply_frozen_module_modes(self) -> None:
        """Keep frozen upstream modules deterministic during staged training."""

        source = self.model.module if hasattr(self.model, "module") else self.model
        for prefix in self.config.training.frozen_modules:
            try:
                module = source.get_submodule(prefix)
            except AttributeError:
                # Exact parameter names are valid freeze targets, but do not
                # have an independent train/eval mode.
                continue
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()

    def _save_run_metadata(self) -> None:
        import yaml

        (self.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(asdict(self.config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "uncommitted"
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "git_commit": commit,
                    "torch": torch.__version__,
                    "train_metrics_schema_version": 6,
                    "validation_metrics_schema_version": 3,
                    "training_events_schema_version": 1,
                },
                indent=2,
            ),
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

    @classmethod
    def _gradient_group_norms(cls, model: nn.Module) -> dict[str, float]:
        """Measure disjoint top-level module norms without another backward pass."""

        source = model.module if hasattr(model, "module") else model
        squared: dict[str, Tensor] = {}
        for name, parameter in source.named_parameters():
            if parameter.grad is None:
                continue
            group = "other"
            top = name.split(".", 1)[0]
            for label, prefixes in cls.GRADIENT_GROUPS.items():
                if top in prefixes:
                    group = label
                    break
            value = parameter.grad.detach().float().square().sum()
            squared[group] = squared.get(group, value.new_zeros(())) + value
        return {f"gradient_norm_{group}": float(value.sqrt()) for group, value in squared.items()}

    @staticmethod
    def _set_loader_epoch(loader: Any, epoch: int) -> None:
        for name in ("batch_sampler", "sampler"):
            sampler = getattr(loader, name, None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
                return

    def train(
        self,
        loader: Iterable[Mapping[str, Any]],
        validate: Callable[[nn.Module], Any] | None = None,
        groups_per_effective_epoch: int | None = None,
        step_finished: Callable[[int], None] | None = None,
        auxiliary_loader: Iterable[Mapping[str, Any]] | None = None,
        auxiliary_loss_step: Callable[
            [nn.Module, Mapping[str, Any]], tuple[Tensor, Mapping[str, Tensor]]
        ]
        | None = None,
        auxiliary_weight: float = 1.0,
    ) -> TrainerState:
        if (auxiliary_loader is None) != (auxiliary_loss_step is None):
            raise ValueError("auxiliary_loader and auxiliary_loss_step must be provided together")
        if auxiliary_weight <= 0:
            raise ValueError("auxiliary_weight must be positive")
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
        window_global_states = 0
        window_data_seconds = 0.0
        terminal_records: list[Mapping[str, Any]] = []
        initial_optimizer_step = self.state.optimizer_steps
        batch_finished_time = time.time()
        epoch = 0
        auxiliary_epoch = 0
        auxiliary_iterator = None
        self._write_event(
            "training_started",
            max_optimizer_steps=self.config.training.max_optimizer_steps,
            gradient_accumulation_steps=accumulation,
            global_stream=auxiliary_loader is not None,
            global_stream_weight=float(auxiliary_weight),
        )
        if self.is_primary:
            point_count = (
                "variable"
                if self.config.dataset.scene_points == 0
                else str(self.config.dataset.scene_points)
            )
            print(
                f"[train-start] output={self.output_dir.resolve()} "
                f"steps={self.state.optimizer_steps}->{self.config.training.max_optimizer_steps} "
                f"batch={self.config.training.batch_size} accumulation={accumulation} "
                f"global_stream={'on' if auxiliary_loader is not None else 'off'} "
                f"workers={self.config.training.num_workers} "
                f"validation_workers={self.config.training.validation_num_workers} "
                f"points={point_count} "
                f"grid={self.config.backbone.grid_size_m:g}m "
                f"terminal_interval={self.config.logging.log_interval}",
                flush=True,
            )
        # Resume contract: an exact validation-boundary checkpoint must finish
        # or recover that validation before the next training batch is consumed.
        if validate is not None and self._resume_validation_due():
            stop = self._run_validation_cycle(
                validate,
                int(self.state.optimizer_steps),
                resumed=True,
            )
            last_optimizer_time = time.time()
            batch_finished_time = last_optimizer_time

        while self.state.optimizer_steps < self.config.training.max_optimizer_steps and not stop:
            self._set_loader_epoch(loader, epoch)
            epoch += 1
            for raw_batch in loader:
                window_data_seconds += max(time.time() - batch_finished_time, 0.0)
                saw_batch = True
                micro_step += 1
                batch = self._move(raw_batch, self.device)
                auxiliary_batch = None
                if auxiliary_loader is not None:
                    auxiliary_started = time.time()
                    if auxiliary_iterator is None:
                        self._set_loader_epoch(auxiliary_loader, auxiliary_epoch)
                        auxiliary_epoch += 1
                        auxiliary_iterator = iter(auxiliary_loader)
                    try:
                        raw_auxiliary = next(auxiliary_iterator)
                    except StopIteration:
                        self._set_loader_epoch(auxiliary_loader, auxiliary_epoch)
                        auxiliary_epoch += 1
                        auxiliary_iterator = iter(auxiliary_loader)
                        raw_auxiliary = next(auxiliary_iterator)
                    auxiliary_batch = self._move(raw_auxiliary, self.device)
                    window_data_seconds += max(time.time() - auxiliary_started, 0.0)
                autocast_enabled = self.config.training.amp and self.device.type == "cuda"
                amp_dtype = (
                    torch.bfloat16
                    if self.config.training.amp_dtype == "bfloat16"
                    else torch.float16
                )
                # DDP 只在累计窗口末端同步梯度；默认 accumulation=1 时每步同步。
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
                        loss, base_terms = self.loss_step(self.model, batch)
                        scaled_loss = loss / accumulation
                    self.scaler.scale(scaled_loss).backward()
                    terms = dict(base_terms)
                    if auxiliary_batch is not None and auxiliary_loss_step is not None:
                        # Separate forward/backward is intentional: it is the safe
                        # first implementation and avoids coupling two DDP graphs.
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=amp_dtype,
                            enabled=autocast_enabled,
                        ):
                            auxiliary_loss, auxiliary_terms = auxiliary_loss_step(
                                self.model, auxiliary_batch
                            )
                            scaled_auxiliary = auxiliary_weight * auxiliary_loss / accumulation
                        self.scaler.scale(scaled_auxiliary).backward()
                        for key, value in auxiliary_terms.items():
                            if key in terms and key != "loss_total":
                                replace_zero_activity = (
                                    key == "active_loss_global_grasp"
                                    and not bool(terms[key].detach().bool().any())
                                )
                                if not replace_zero_activity:
                                    raise KeyError(
                                        f"Auxiliary loss term collides with main stream: {key}"
                                    )
                            if key.startswith("weighted_loss_"):
                                value = auxiliary_weight * value
                            terms[key] = value
                        loss = loss + auxiliary_weight * auxiliary_loss
                        terms["loss_total"] = loss.detach()
                batch_size = self._batch_size(batch)
                global_batch_size = batch_size * (
                    torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                )
                self.state.samples_seen += global_batch_size
                self.state.states_seen += global_batch_size
                self.state.candidate_groups_seen += global_batch_size
                if auxiliary_batch is not None:
                    auxiliary_size = self._batch_size(auxiliary_batch) * (
                        torch.distributed.get_world_size()
                        if torch.distributed.is_initialized()
                        else 1
                    )
                    self.state.global_states_seen += auxiliary_size
                    window_global_states += auxiliary_size
                window_micro_batches += 1
                window_samples += global_batch_size
                window_values = dict(terms)
                window_values.setdefault("loss_total", loss.detach())
                for key, value in window_values.items():
                    detached = value.detach().float()
                    window_term_sums[key] = (
                        window_term_sums.get(key, torch.zeros_like(detached)) + detached
                    )
                    window_term_counts[key] = window_term_counts.get(key, 0) + 1
                if not synchronize:
                    batch_finished_time = time.time()
                    continue
                self.scaler.unscale_(self.optimizer)
                diagnostics: dict[str, float] = {}
                diagnostics_interval = self.config.logging.gradient_diagnostics_interval
                if (
                    diagnostics_interval > 0
                    and (self.state.optimizer_steps + 1) % diagnostics_interval == 0
                ):
                    diagnostics = self._gradient_group_norms(self.model)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                gradient_norm_value = float(gradient_norm)
                clip_threshold = float(self.config.training.gradient_clip_norm)
                clip_scale = min(1.0, clip_threshold / max(gradient_norm_value, 1e-12))
                gradient_norm_after_clip = min(gradient_norm_value, clip_threshold)
                previous_scale = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                current_scale = float(self.scaler.get_scale())
                self.optimizer.zero_grad(set_to_none=True)
                # AMP 溢出时 GradScaler 会跳过 optimizer.step；此时不能推进 scheduler、
                # EMA、全局步数或 checkpoint，否则记录的“训练步”并未真正更新参数。
                if self.scaler.is_enabled() and current_scale < previous_scale:
                    self.state.amp_skipped_steps += 1
                    skipped_terms = self._reduce_distributed_terms(
                        self._aggregate_window_terms(window_term_sums, window_term_counts)
                    )
                    self._write_event(
                        "amp_step_skipped",
                        previous_amp_scale=previous_scale,
                        current_amp_scale=current_scale,
                        gradient_norm=gradient_norm_value,
                        gradient_norm_after_clip=gradient_norm_after_clip,
                        gradient_clip_scale=clip_scale,
                        micro_batches=window_micro_batches,
                        samples=window_samples,
                        metrics=skipped_terms,
                    )
                    if self.is_primary:
                        print(
                            f"[amp-skip step={self.state.optimizer_steps:07d}] "
                            f"scale={previous_scale:.0f}->{current_scale:.0f} "
                            f"grad={gradient_norm_value:.3f}",
                            flush=True,
                        )
                    if self.tensorboard is not None:
                        self.tensorboard.add_scalar(
                            "amp/skipped_steps",
                            self.state.amp_skipped_steps,
                            self.state.optimizer_steps,
                        )
                        self.tensorboard.add_scalar(
                            "amp/scale", current_scale, self.state.optimizer_steps
                        )
                    window_term_sums.clear()
                    window_term_counts.clear()
                    window_micro_batches = 0
                    window_samples = 0
                    window_global_states = 0
                    window_data_seconds = 0.0
                    last_optimizer_time = time.time()
                    batch_finished_time = last_optimizer_time
                    continue
                self.state.optimizer_steps += 1
                if (
                    self.config.training.unfreeze_at_optimizer_step is not None
                    and self.state.optimizer_steps
                    >= self.config.training.unfreeze_at_optimizer_step
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
                completed_this_run = step - initial_optimizer_step
                remaining_steps = self.config.training.max_optimizer_steps - step
                eta_seconds = (now - started) / max(1, completed_this_run) * remaining_steps
                aggregated_terms = self._reduce_distributed_terms(
                    self._aggregate_window_terms(window_term_sums, window_term_counts)
                )
                if self.is_primary:
                    record = {
                        "schema_version": 5,
                        "timestamp_utc": self._timestamp(),
                        "optimizer_step": step,
                        "amp_skipped_steps": self.state.amp_skipped_steps,
                        "amp_scale": current_scale,
                        "samples_seen": self.state.samples_seen,
                        "states_seen": self.state.states_seen,
                        "candidate_groups_seen": self.state.candidate_groups_seen,
                        "global_states_seen": self.state.global_states_seen,
                        "effective_epochs": self.state.effective_epochs,
                        "gradient_norm": gradient_norm_value,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                        "gradient_clip_scale": clip_scale,
                        "gradient_clipped": float(clip_scale < 1.0),
                        "elapsed_seconds": now - started,
                        "eta_seconds": eta_seconds,
                        "optimizer_step_seconds": step_seconds,
                        "data_seconds": window_data_seconds,
                        "samples_per_second": window_samples / step_seconds,
                        "max_memory_mb": (
                            torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                            if self.device.type == "cuda"
                            else 0.0
                        ),
                        "micro_batches": window_micro_batches,
                        "window_samples": window_samples,
                        "window_global_states": window_global_states,
                        "metric_scope": (
                            "global" if torch.distributed.is_initialized() else "local"
                        ),
                        "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                        **diagnostics,
                        **aggregated_terms,
                    }
                    record["training_stage"] = self._training_stage(record)
                    self._append_jsonl(self.metrics_path, record)
                    if self.tensorboard is not None:
                        for key, value in record.items():
                            if isinstance(value, (int, float)):
                                self.tensorboard.add_scalar(key, value, step)
                        for index, rate in enumerate(record["learning_rates"]):
                            self.tensorboard.add_scalar(f"learning_rate/group_{index}", rate, step)
                    terminal_records.append(record)
                    if (
                        step % self.config.logging.log_interval == 0
                        or step == 1
                        or step == self.config.training.max_optimizer_steps
                    ):
                        self._print_train_summary(self._summarize_terminal_window(terminal_records))
                        terminal_records.clear()
                window_term_sums.clear()
                window_term_counts.clear()
                window_micro_batches = 0
                window_samples = 0
                window_global_states = 0
                window_data_seconds = 0.0
                if step_finished is not None:
                    step_finished(step)
                if (
                    validate is not None
                    and self.config.training.validation_interval > 0
                    and step % self.config.training.validation_interval == 0
                ):
                    stop = self._run_validation_cycle(
                        validate,
                        step,
                        resumed=False,
                    )
                    if stop:
                        break
                last_optimizer_time = time.time()
                batch_finished_time = last_optimizer_time
                if step >= self.config.training.max_optimizer_steps:
                    stop = True
                    break
            if not saw_batch:
                raise RuntimeError("Training loader yielded no batches")
        if self.is_primary and terminal_records:
            self._print_train_summary(self._summarize_terminal_window(terminal_records))
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        self._write_event(
            "training_completed",
            best_validation=self.state.best_validation,
            stopped_early=(
                stop and self.state.optimizer_steps < self.config.training.max_optimizer_steps
            ),
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
                torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None
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
            "schema_version": 12,
            "training_stage": self.config.training.stage,
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
            "task_grasp_probability_threshold": self.task_grasp_probability_threshold,
            "stageb_provenance": self.stageb_provenance,
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
        if schema_version != 12:
            raise RuntimeError(
                "Unsupported TCD-PRG checkpoint schema "
                f"{schema_version}; staged training expects schema 12."
            )
        checkpoint_stage = payload.get("training_stage")
        if checkpoint_stage != self.config.training.stage:
            raise RuntimeError(
                "Resume checkpoint stage mismatch: "
                f"checkpoint={checkpoint_stage!r}, requested={self.config.training.stage!r}"
            )
        if (
            checkpoint_stage == "grasp"
            and compatibility_provenance(payload.get("stageb_provenance", {}))
            != compatibility_provenance(self.stageb_provenance)
        ):
            raise RuntimeError("Stage-B resume checkpoint provenance does not match the dataset")
        source = self.model.module if hasattr(self.model, "module") else self.model
        source.load_state_dict(payload["model"])
        self.task_grasp_probability_threshold = float(
            payload.get(
                "task_grasp_probability_threshold",
                self.config.model.task_grasp_probability_threshold,
            )
        )
        self.config.model.task_grasp_probability_threshold = self.task_grasp_probability_threshold
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
