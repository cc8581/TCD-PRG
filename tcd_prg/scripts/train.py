"""Unified full-model and ablation training entry point."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import (
    ActionStateGroupDataset,
    DistributedEvaluationSampler,
    DistributedTaskStateBatchSampler,
)
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.observation.cached import CachedObservationProvider
from tcd_prg.pretrained import load_pretrained_backbone, prepare_pretrained_checkpoint
from tcd_prg.runtime import (
    UnifiedBatchCollator,
    create_adapter,
)
from tcd_prg.trainers import Trainer


def validate_read_through_observation_cache(adapter) -> dict[str, object]:
    """Validate the bounded cache and its renderer without scanning the dataset."""

    provider = adapter.observation_provider
    if not isinstance(provider, CachedObservationProvider):
        raise TypeError("Formal training requires CachedObservationProvider")
    usage = shutil.disk_usage(provider.cache_dir)
    if usage.free <= provider.min_free_bytes:
        raise OSError(
            f"Observation cache {provider.cache_dir} has {usage.free} free bytes, "
            f"below the configured reserve {provider.min_free_bytes}"
        )
    strict = provider.fallback is None
    return {
        "mode": "strict-cache-only" if strict else "read-through-lru",
        "directory": str(provider.cache_dir.resolve()),
        "max_gb": None if strict else round(provider.max_bytes / (1 << 30), 3),
        "free_gb": round(usage.free / (1 << 30), 3),
        "missing": "error" if strict else "render-on-demand",
    }


def load_or_create_validation_scene_subset(
    scene_ids: list[int] | tuple[int, ...],
    count: int | None,
    seed: int,
    manifest_path: str | Path,
) -> tuple[int, ...]:
    """Select complete validation scenes once and reuse the audited manifest."""

    available = tuple(sorted({int(scene_id) for scene_id in scene_ids}))
    if not available:
        raise ValueError("The validation split contains no scenes")
    if count is None:
        return available
    if count <= 0 or count > len(available):
        raise ValueError(
            f"training.validation_scene_count={count} must be in [1,{len(available)}]"
        )
    path = Path(manifest_path)
    fingerprint = hashlib.sha256(
        json.dumps(available, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError("validation scene subset seed does not match the manifest")
        if int(payload.get("selected_scene_count", -1)) != int(count):
            raise ValueError("validation scene subset count does not match the manifest")
        if payload.get("source_scene_fingerprint") != fingerprint:
            raise ValueError("validation split scenes have changed since the manifest was written")
        selected = tuple(int(value) for value in payload.get("scene_ids", []))
        if len(selected) != count or len(set(selected)) != count:
            raise ValueError("validation scene subset manifest has invalid scene IDs")
        if not set(selected).issubset(available):
            raise ValueError("validation scene subset contains a scene outside the split")
        return selected

    selected = tuple(
        sorted(
            int(value)
            for value in np.random.default_rng(seed).choice(
                np.asarray(available, dtype=np.int64), size=count, replace=False
            )
        )
    )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "seed": int(seed),
            "source_scene_count": len(available),
            "source_scene_fingerprint": fingerprint,
            "selected_scene_count": int(count),
            "scene_ids": list(selected),
        }
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    return selected


def restrict_action_dataset_to_stage(
    dataset,
    allowed_strata: tuple[str, ...],
    *,
    deduplicate_state_task: bool = False,
) -> None:
    """Restrict action groups without changing their published labels."""
    units = dataset.units
    if allowed_strata:
        allowed = frozenset(str(value) for value in allowed_strata)
        units = tuple(unit for unit in units if unit.stratum in allowed)
        if not units:
            raise RuntimeError(
                f"Stage filter {sorted(allowed)} removed every action-state group"
            )
    if deduplicate_state_task:
        unique = {}
        for unit in units:
            unique.setdefault(
                (unit.scene_id, unit.state_id, unit.task_index), unit
            )
        units = tuple(unique.values())
    dataset.units = units
    dataset.selected_group_count = len(units)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--initialize", help="Load model/EMA weights without optimizer state")
    # Windows 原生启动器显式传递进程拓扑；torchrun 的环境变量仅作为 Linux 兼容路径。
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--ddp-init-method")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if config.losses.task_grasp > 0:
        if not config.dataset.acronym_object_grasp_database:
            raise ValueError(
                "Task-grasp training requires dataset.acronym_object_grasp_database; "
                "refusing to silently fall back to the old sparse supervision."
            )
        database_root = Path(config.dataset.acronym_object_grasp_database)
        manifest = database_root / "manifest.json"
        if not database_root.is_dir() or not manifest.is_file():
            raise FileNotFoundError(
                "Task-grasp training requires a valid ACRONYM object-grasp database "
                f"with manifest.json: {database_root}"
            )
    if config.observation.provider != "cached":
        raise ValueError(
            "Formal training requires observation.provider=cached for bounded read-through caching"
        )
    if args.resume and args.initialize:
        raise ValueError("--resume and --initialize are mutually exclusive")
    world_size = (
        args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    )
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    local_rank = (
        args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", "0"))
    )
    use_cuda = torch.cuda.is_available() and config.training.device.startswith("cuda")
    # Windows CUDA DDP 默认 gloo；Linux CUDA 优先 NCCL。每个 rank 绑定唯一显卡。
    if world_size > 1:
        backend = config.training.ddp_backend
        if backend == "auto":
            backend = "gloo" if os.name == "nt" else ("nccl" if use_cuda else "gloo")
        if use_cuda and local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices exist"
            )
        init_method = args.ddp_init_method
        if init_method:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
        else:
            torch.distributed.init_process_group(backend=backend)
        atexit.register(_destroy_process_group)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            config.training.device = f"cuda:{local_rank}"
    # 离线点云生成完成后可关闭 on-miss renderer。严格模式下缺失缓存直接报错，
    # 且训练启动时不会对离线缓存执行 LRU eviction。
    adapter = create_adapter(config, allow_render=config.observation.allow_render_on_miss)
    provider = adapter.observation_provider
    if (
        rank == 0
        and isinstance(provider, CachedObservationProvider)
        and provider.fallback is not None
    ):
        provider.evict()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    observation_cache = validate_read_through_observation_cache(adapter)
    perception_only_stage = (
        config.losses.task_grasp == 0
        and config.losses.push_object == 0
        and config.losses.push_contact == 0
        and config.losses.push_direction == 0
        and config.losses.push_potential == 0
    )
    train_dataset = ActionStateGroupDataset(
        adapter,
        split="train",
        max_groups=config.training.max_train_groups,
        allowed_strata=config.training.allowed_action_strata,
        deduplicate_state_task=perception_only_stage,
        global_grasp_mode="never",
    )
    validation_scene_ids = (
        load_or_create_validation_scene_subset(
            adapter.scene_splits["val"],
            config.training.validation_scene_count,
            config.training.validation_scene_seed,
            os.path.join(config.output_dir, "validation_scene_subset.json"),
        )
        if config.training.validation_interval > 0
        else ()
    )
    validation_dataset = (
        ActionStateGroupDataset(
            adapter,
            split="val",
            scene_ids=frozenset(validation_scene_ids),
            max_groups=config.training.max_validation_groups,
            allowed_strata=config.training.allowed_action_strata,
            deduplicate_state_task=perception_only_stage,
            stratified_max_groups=config.training.max_validation_groups is not None,
            stratum_quota=config.training.validation_stratum_quota,
            subset_manifest_path=(
                os.path.join(config.output_dir, "validation_subset.json")
                if config.training.max_validation_groups is not None
                else None
            ),
            global_grasp_mode="never",
            global_grasp_width_bounds=(
                config.model.min_grasp_width_m,
                config.model.max_grasp_width_m,
            ),
        )
        if config.training.validation_interval > 0
        else None
    )
    if not len(train_dataset):
        raise RuntimeError("The completed-file snapshot contains no training groups")
    if rank == 0:
        print(
            f"[observation-cache] {json.dumps(observation_cache, ensure_ascii=False)}", flush=True
        )
        print(
            f"[train-data] fraction={config.training.data_fraction:g} "
            f"split_ratios={list(config.training.split_ratios)} "
            f"scenes={len(adapter.scene_splits['train'])} groups={len(train_dataset)} "
            f"source_groups={train_dataset.source_group_count}",
            flush=True,
        )
        print(
            f"[train-strata] {json.dumps(train_dataset.stratum_counts, ensure_ascii=False)}",
            flush=True,
        )
        if validation_dataset is not None:
            print(
                f"[val-data] scenes={len(validation_scene_ids)}/"
                f"{len(adapter.scene_splits['val'])} "
                f"groups={len(validation_dataset)}",
                flush=True,
            )
    train_collator = UnifiedBatchCollator(config, training=True)
    validation_collator = UnifiedBatchCollator(config, training=False)
    train_batch_sampler = DistributedTaskStateBatchSampler(
        train_dataset.units,
        batch_size=config.training.batch_size,
        coverage_strata=config.training.action_batch_coverage_strata,
        rank=rank,
        world_size=world_size,
        seed=config.training.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=train_collator,
    )
    validation_sampler = (
        DistributedEvaluationSampler(len(validation_dataset), rank, world_size)
        if world_size > 1
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            # Validation starts while both persistent training loaders remain
            # alive.  A separate setting prevents a third Windows worker pool
            # and its prefetched queue copies from exhausting commit memory.
            num_workers=config.training.validation_num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.validation_num_workers > 0,
            collate_fn=validation_collator,
        )
        if validation_dataset is not None and len(validation_dataset)
        else None
    )
    model = TCDPRGModel(
        config.model, config.ablation, config.backbone, config.graspnet,
    )
    pretrained_report = None
    resume_pretrained_names: list[str] = []
    resume_validation_protocol_changed = False
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = resume_payload.get("config", {})
        resume_extra = resume_config.get("extra", {}) if isinstance(resume_config, dict) else {}
        resume_pretrained_names = list(resume_extra.get("pretrained_matched_parameter_names", []))
        # Older pre-minimal-architecture checkpoints may have been written before the
        # pre-trained parameter-name list was copied into ``config.extra``.
        # The optimizer still contains the original split, so falling back to
        # every encoder parameter changes the parameter-group sizes and makes
        # optimizer state restoration fail.  The run artifact contains the
        # exact audited list used to construct that optimizer.
        if not resume_pretrained_names:
            pretrained_report_path = Path(config.output_dir) / "pretrained_backbone.json"
            if pretrained_report_path.is_file():
                persisted_report = json.loads(
                    pretrained_report_path.read_text(encoding="utf-8")
                )
                resume_pretrained_names = list(
                    persisted_report.get("matched_parameter_names", [])
                )
        resume_training = (
            resume_config.get("training", {}) if isinstance(resume_config, dict) else {}
        )
        if isinstance(resume_training, dict):
            old_validation_signature = (
                resume_training.get("validation_scene_count"),
                int(resume_training.get("validation_scene_seed", 2026)),
                resume_training.get("max_validation_groups"),
            )
            current_validation_signature = (
                config.training.validation_scene_count,
                int(config.training.validation_scene_seed),
                config.training.max_validation_groups,
            )
            resume_validation_protocol_changed = (
                old_validation_signature != current_validation_signature
            )
            config.training.frozen_modules = tuple(
                resume_training.get("frozen_modules", config.training.frozen_modules)
            )
            config.training.unfreeze_at_optimizer_step = resume_training.get(
                "unfreeze_at_optimizer_step",
                config.training.unfreeze_at_optimizer_step,
            )
    if args.initialize:
        initialized = torch.load(args.initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(initialized.get("ema") or initialized.get("model") or initialized)
    elif not args.resume:
        distributed = torch.distributed.is_initialized()
        checkpoint_path = None
        # Only rank zero may resolve a missing managed checkpoint. Other ranks
        # wait until the atomic download and checksum validation are complete.
        if rank == 0 or not distributed:
            checkpoint_path = prepare_pretrained_checkpoint(config.backbone, allow_download=True)
        if distributed:
            torch.distributed.barrier()
        if rank != 0:
            checkpoint_path = prepare_pretrained_checkpoint(config.backbone, allow_download=False)
        if checkpoint_path is not None:
            pretrained_report = load_pretrained_backbone(model, checkpoint_path, config.backbone)
            config.extra["pretrained_matched_parameter_names"] = list(
                pretrained_report["matched_parameter_names"]
            )
            if rank == 0:
                os.makedirs(config.output_dir, exist_ok=True)
                with open(
                    os.path.join(config.output_dir, "pretrained_backbone.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(pretrained_report, handle, ensure_ascii=False, indent=2)
                print(
                    "[pretrained] matched_parameter_fraction="
                    f"{pretrained_report['matched_parameter_fraction']:.1%}",
                    flush=True,
                )
    if config.backbone.freeze and "encoder" not in config.training.frozen_modules:
        config.training.frozen_modules = (*config.training.frozen_modules, "encoder")
    elif pretrained_report and config.backbone.pretrained_freeze_steps > 0:
        prefixes = tuple(pretrained_report["freeze_prefixes"])
        config.training.frozen_modules = tuple(
            dict.fromkeys((*config.training.frozen_modules, *prefixes))
        )
        config.training.unfreeze_at_optimizer_step = max(
            config.backbone.pretrained_freeze_steps,
            config.training.unfreeze_at_optimizer_step or 0,
        )

    # For fixed stages, freeze unused modules before optimizer/DDP construction.
    # This avoids optimizer state and DDP reducer bookkeeping for branches that
    # the stage never executes. Dynamic unfreeze experiments retain Trainer's
    # delayed freeze path so those parameters stay inside the optimizer.
    if (
        config.training.frozen_modules
        and config.training.unfreeze_at_optimizer_step is None
    ):
        for parameter_name, parameter in model.named_parameters():
            if any(
                parameter_name == prefix
                or parameter_name.startswith(prefix + ".")
                for prefix in config.training.frozen_modules
            ):
                parameter.requires_grad_(False)

    pretrained_parameter_names = (
        list(pretrained_report["matched_parameter_names"])
        if pretrained_report is not None
        else resume_pretrained_names
    )
    if pretrained_parameter_names:
        named_parameters = dict(model.named_parameters())
        missing_names = [
            name for name in pretrained_parameter_names if name not in named_parameters
        ]
        if missing_names:
            raise RuntimeError(
                "Checkpoint pre-trained parameter names do not match this model: "
                + ", ".join(missing_names[:8])
            )
        low_lr_parameters = [
            named_parameters[name] for name in pretrained_parameter_names
            if named_parameters[name].requires_grad
        ]
        low_lr = config.optimizer.backbone_learning_rate
    else:
        low_lr_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
        low_lr = config.optimizer.learning_rate
    low_lr_ids = {id(parameter) for parameter in low_lr_parameters}
    other_parameters = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in low_lr_ids
    ]
    optimizer_groups = []
    if low_lr_parameters:
        optimizer_groups.append(
            {"params": low_lr_parameters, "lr": low_lr, "name": "pretrained_trunk"}
        )
    if other_parameters:
        optimizer_groups.append({
            "params": other_parameters,
            "lr": config.optimizer.learning_rate,
            "name": "new_modules",
        })
    if not optimizer_groups:
        raise RuntimeError("Selected training stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=config.optimizer.weight_decay
    )

    def learning_rate(step: int) -> float:
        # 线性 warmup 后按优化器步执行 cosine decay；AMP skip 不会推进此调度器。
        if step < config.scheduler.warmup_steps:
            return max(1e-8, step / max(1, config.scheduler.warmup_steps))
        progress = (step - config.scheduler.warmup_steps) / max(
            1, config.training.max_optimizer_steps - config.scheduler.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    if world_size > 1:
        model = model.to(torch.device(config.training.device))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=config.training.ddp_find_unused_parameters,
        )
    # 数据能力与消融开关共同决定实际启用的任务损失，结果写入 loss_routing.json。
    objective = TCDPRGObjective(
        adapter.capabilities,
        config.model,
        config.ablation,
        config.losses,
        config.region_head,
        config.dataset.acronym_object_grasp_database,
    )
    if rank == 0:
        enabled = {name: objective.total.enabled(name) for name in objective.total.DEFAULT_WEIGHTS}
        output = os.path.join(config.output_dir, "loss_routing.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_capabilities": asdict(adapter.capabilities),
                    "enabled": enabled,
                    "family_weights": objective.total.weights,
                    "action_sampling": "task_state_first_best_effort_stratum_coverage",
                    "action_batch_coverage_strata": list(
                        config.training.action_batch_coverage_strata
                    ),
                    "disabled": {
                        name: "dataset capability or ablation"
                        for name, value in enabled.items()
                        if not value
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    trainer = Trainer(
        model, optimizer, config, objective, scheduler=scheduler, output_dir=config.output_dir
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
        if resume_validation_protocol_changed:
            previous_best = trainer.state.best_validation
            trainer.state.best_validation = float("inf")
            trainer.state.validation_without_improvement = 0
            trainer._write_event(
                "validation_protocol_reset",
                previous_best_validation=previous_best,
                validation_scene_count=config.training.validation_scene_count,
                validation_scene_seed=config.training.validation_scene_seed,
                max_validation_groups=config.training.max_validation_groups,
            )
            if rank == 0:
                print(
                    "[validation-protocol] subset changed on resume; reset best "
                    f"validation from {previous_best:.6f} and early-stopping counter",
                    flush=True,
                )

    def validate(module: torch.nn.Module) -> dict[str, object]:
        if validation_loader is None:
            return {
                "score_sum": float("inf"),
                "score_count": 1,
                "metric_sums": {},
                "metric_counts": {},
            }
        module.eval()
        total, count = 0.0, 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        evaluator = OfflineModelEvaluator(
            config.model,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.evaluation,
        )
        validation_batches = len(validation_loader)
        validation_groups = len(validation_loader.sampler)
        with torch.no_grad():
            for validation_step, raw in enumerate(validation_loader, start=1):
                batch = trainer._move(raw, trainer.device)
                _, terms, model_output = objective(module, batch, return_output=True)
                evaluator.update(batch, model_output, terms)
                score = sum(
                    weight * float(terms[f"loss_{family}"])
                    for family, weight in config.training.validation_family_weights.items()
                    if f"loss_{family}" in terms
                )
                # 验证分数按真实 state-group 数加权，最后一个不满 batch 不得与完整 batch 等权。
                groups = int(batch["xyz"].shape[0])
                total += score * groups
                count += groups
                for key, value in terms.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * groups
                    metric_counts[key] = metric_counts.get(key, 0) + groups
                if rank == 0 and (
                    validation_step == 1
                    or validation_step % config.logging.validation_log_interval == 0
                    or validation_step == validation_batches
                ):
                    print(
                        f"[validation batch={validation_step:07d}/{validation_batches:07d}] "
                        f"groups={count}/{validation_groups} "
                        f"score={total / max(1, count):.6f}",
                        flush=True,
                    )
        module.train()
        return {
            "score_sum": total,
            "score_count": count,
            "metric_sums": metric_sums,
            "metric_counts": metric_counts,
            "evaluation_records": evaluator.evaluator.records,
        }

    state = trainer.train(
        train_loader,
        validate=validate if validation_loader is not None else None,
        groups_per_effective_epoch=train_batch_sampler.global_samples_per_epoch,
    )
    final_checkpoint = os.path.join(config.output_dir, "last.pt")
    trainer.save_checkpoint(final_checkpoint)
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        print(
            f"Saved final checkpoint: {os.path.abspath(final_checkpoint)} "
            f"(step {state.optimizer_steps:07d})",
            flush=True,
        )
    if world_size > 1:
        _destroy_process_group()
        atexit.unregister(_destroy_process_group)


def _destroy_process_group() -> None:
    """Best-effort DDP cleanup for both normal and exceptional exits."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
