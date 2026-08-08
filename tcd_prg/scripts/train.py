"""Unified full-model and ablation training entry point."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tcd_prg.config import load_config
from tcd_prg.datasets import (
    ActionStateGroupDataset,
    DistributedEvaluationSampler,
    DistributedTaskStateBatchSampler,
    GlobalStateDataset,
)
from tcd_prg.datasets.policy_candidates import (
    checkpoint_sha256,
    validate_cache_manifest,
    validate_generated_policy_coverage,
)
from tcd_prg.diagnostics import GraspDiagnosticAccumulator
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.observation.cached import CachedObservationProvider
from tcd_prg.pretrained import load_pretrained_backbone, prepare_pretrained_checkpoint
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter, create_gripper_provider
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
    if config.observation.provider != "cached":
        raise ValueError(
            "Formal training requires observation.provider=cached for bounded read-through caching"
        )
    if args.resume and args.initialize:
        raise ValueError("--resume and --initialize are mutually exclusive")
    # 只要训练使用 generated candidates，就必须校验生成器 checkpoint 和缓存版本签名。
    if config.training.generated_policy_candidate_ratio > 0:
        if args.initialize:
            actual_sha256 = checkpoint_sha256(args.initialize)
            expected = config.training.generated_policy_checkpoint_sha256
            if expected and expected != actual_sha256:
                raise ValueError("--initialize checkpoint does not match generated cache SHA-256")
            config.training.generated_policy_checkpoint_sha256 = actual_sha256
        elif not config.training.generated_policy_checkpoint_sha256:
            raise ValueError(
                "Generated policy training requires --initialize or an explicit "
                "training.generated_policy_checkpoint_sha256"
            )
        manifest = validate_cache_manifest(
            config.training.generated_policy_candidate_cache,
            config,
            config.training.generated_policy_checkpoint_sha256,
        )
        if config.training.generated_policy_candidate_ratio == 1.0:
            coverage = validate_generated_policy_coverage(
                manifest,
                "train",
                minimum_positive=config.training.generated_policy_min_positive_coverage,
                minimum_effective=config.training.generated_policy_min_effective_coverage,
            )
            print({"generated_policy_preflight": coverage})
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
    train_dataset = ActionStateGroupDataset(
        adapter,
        split="train",
        max_groups=config.training.max_train_groups,
        global_grasp_mode="never",
    )
    global_dataset = (
        GlobalStateDataset(
            train_dataset,
            config.model.min_grasp_width_m,
            config.model.max_grasp_width_m,
        )
        if float(config.losses.global_grasp) > 0 and adapter.capabilities.has_global_grasps
        else None
    )
    validation_dataset = (
        ActionStateGroupDataset(
            adapter,
            split="val",
            max_groups=config.training.max_validation_groups,
            stratified_max_groups=config.training.max_validation_groups is not None,
            stratum_quota=config.training.validation_stratum_quota,
            subset_manifest_path=(
                os.path.join(config.output_dir, "validation_subset.json")
                if config.training.max_validation_groups is not None
                else None
            ),
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
        if global_dataset is not None:
            print(
                f"[global-data] unique_supervised_scene_states={len(global_dataset)}",
                flush=True,
            )
        print(
            f"[train-strata] {json.dumps(train_dataset.stratum_counts, ensure_ascii=False)}",
            flush=True,
        )
        if validation_dataset is not None:
            print(
                f"[val-data] scenes={len(adapter.scene_splits['val'])} "
                f"groups={len(validation_dataset)}",
                flush=True,
            )
    gripper = None
    if config.ablation.use_gripper_scene_verifier:
        if rank == 0:
            # 正式训练只在 DataLoader 启动前生成有限宽度档位；worker 永不调用 PyBullet。
            gripper = create_gripper_provider(config, allow_generate=True)
            gripper_paths = gripper.prewarm_uniform_bins()
            gripper.allow_generate = False
            print(
                f"[gripper-cache] ready bins={len(gripper_paths)} "
                f"quantization={config.grasp_verifier.gripper_width_quantization_m:g}m",
                flush=True,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank != 0:
            gripper = create_gripper_provider(config, allow_generate=False)
    train_collator = UnifiedBatchCollator(config, gripper, training=True)
    validation_collator = UnifiedBatchCollator(config, gripper, training=False)
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
    global_loader = None
    if global_dataset is not None:
        global_sampler = DistributedSampler(
            global_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.seed + 100_003,
            drop_last=False,
        )
        global_loader = DataLoader(
            global_dataset,
            batch_size=config.training.batch_size,
            sampler=global_sampler,
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
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.num_workers > 0,
            collate_fn=validation_collator,
        )
        if validation_dataset is not None and len(validation_dataset)
        else None
    )
    model = TCDPRGModel(config.model, config.ablation, config.graph, config.router, config.backbone)
    pretrained_report = None
    resume_pretrained_names: list[str] = []
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = resume_payload.get("config", {})
        resume_extra = resume_config.get("extra", {}) if isinstance(resume_config, dict) else {}
        resume_pretrained_names = list(resume_extra.get("pretrained_matched_parameter_names", []))
        resume_training = (
            resume_config.get("training", {}) if isinstance(resume_config, dict) else {}
        )
        if isinstance(resume_training, dict):
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

    # Only parameters actually restored from pre-training receive the small LR.
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
        low_lr_parameters = [named_parameters[name] for name in pretrained_parameter_names]
        low_lr = config.optimizer.backbone_learning_rate
    else:
        low_lr_parameters = list(model.encoder.parameters())
        low_lr = config.optimizer.learning_rate
    low_lr_ids = {id(parameter) for parameter in low_lr_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in low_lr_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": low_lr_parameters, "lr": low_lr, "name": "pretrained_trunk"},
            {
                "params": other_parameters,
                "lr": config.optimizer.learning_rate,
                "name": "new_modules",
            },
        ],
        weight_decay=config.optimizer.weight_decay,
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
        config.training.generated_policy_candidate_ratio,
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
                    "global_supervision_stream": "unique_scene_state",
                    "global_stream_weight": config.training.global_stream_weight,
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
            config.graph,
            config.evaluation,
        )
        grasp_diagnostics = GraspDiagnosticAccumulator(config.model, config.evaluation)
        validation_batches = len(validation_loader)
        validation_groups = len(validation_loader.sampler)
        with torch.no_grad():
            for validation_step, raw in enumerate(validation_loader, start=1):
                batch = trainer._move(raw, trainer.device)
                _, terms, model_output = objective(module, batch, return_output=True)
                evaluator.update(batch, model_output, terms)
                grasp_diagnostics.update(batch, model_output)
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
        diagnostic_payload = grasp_diagnostics.payload()
        # Preserve per-validation-sample diagnostics without gathering the
        # potentially large record list across ranks.  DDP ranks write
        # disjoint files; aggregate means below still use a small all-gather.
        if grasp_diagnostics.records:
            sample_path = (
                Path(config.output_dir) / "validation_grasp_diagnostic_samples.jsonl"
                if not torch.distributed.is_initialized()
                else Path(config.output_dir)
                / f"validation_grasp_diagnostic_samples.rank{rank:03d}.jsonl"
            )
            for sample_record in grasp_diagnostics.records:
                trainer._append_jsonl(
                    sample_path,
                    {
                        "schema_version": 1,
                        "timestamp_utc": trainer._timestamp(),
                        "optimizer_step": trainer.state.optimizer_steps,
                        **sample_record,
                        "scope": "training_diagnostic_only_not_for_paper",
                    },
                )
        if torch.distributed.is_initialized():
            gathered_diagnostics: list[dict[str, object] | None] = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(gathered_diagnostics, diagnostic_payload)
            diagnostic_payloads = [item for item in gathered_diagnostics if item is not None]
        else:
            diagnostic_payloads = [diagnostic_payload]
        if rank == 0:
            diagnostic_sums: dict[str, float] = {}
            diagnostic_counts: dict[str, int] = {}
            for payload in diagnostic_payloads:
                for key, value in payload.get("sums", {}).items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
                for key, value in payload.get("counts", {}).items():
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(value)
            diagnostic_means = {
                key: value / max(1, diagnostic_counts.get(key, 0))
                for key, value in diagnostic_sums.items()
            }
            trainer._append_jsonl(
                Path(config.output_dir) / "validation_grasp_diagnostics.jsonl",
                {
                    "schema_version": 1,
                    "timestamp_utc": trainer._timestamp(),
                    "optimizer_step": trainer.state.optimizer_steps,
                    **diagnostic_means,
                    "counts": diagnostic_counts,
                    "scope": "training_diagnostic_only_not_for_paper",
                },
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
        auxiliary_loader=(global_loader if objective.total.enabled("global_grasp") else None),
        auxiliary_loss_step=(
            objective.global_grasp_stream if objective.total.enabled("global_grasp") else None
        ),
        auxiliary_weight=config.training.global_stream_weight,
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
