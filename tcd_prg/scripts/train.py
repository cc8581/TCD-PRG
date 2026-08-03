"""Unified full-model and ablation training entry point."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset, DistributedEvaluationSampler
from tcd_prg.datasets.policy_candidates import (
    checkpoint_sha256,
    validate_cache_manifest,
    validate_generated_policy_coverage,
)
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter, create_gripper_provider
from tcd_prg.trainers import Trainer


def preflight_observation_cache(adapter, *named_datasets: tuple[str, object]) -> int:
    """Verify every snapshotted state/task observation before model allocation."""

    checked: set[tuple[int, int, int]] = set()
    for split, dataset in named_datasets:
        if dataset is None:
            continue
        show_progress = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        for unit in tqdm(
            dataset.units,
            desc=f"preflight {split} observations",
            unit="group",
            disable=not show_progress,
        ):
            key = (unit.scene_id, unit.state_id, unit.task_index)
            if key in checked:
                continue
            if not adapter.observation_available(*key):
                scene_id, state_id, task_index = key
                raise RuntimeError(
                    "Cache-only training preflight failed before model initialization: "
                    f"missing split={split} scene={scene_id} state={state_id} "
                    f"task={task_index} after {len(checked)} cached observations. "
                    "Formal training never renders observations synchronously. "
                    "Run tcd-prg-prefetch for the same config, then relaunch training."
                )
            checked.add(key)
    return len(checked)


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
            "Formal training requires observation.provider=cached; generate observations "
            "offline with tcd-prg-prefetch"
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
        args.world_size
        if args.world_size is not None
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    local_rank = (
        args.local_rank
        if args.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", "0"))
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
    adapter = create_adapter(config, allow_render=False)
    train_dataset = ActionStateGroupDataset(
        adapter, split="train", max_groups=config.training.max_train_groups
    )
    validation_dataset = (
        ActionStateGroupDataset(
            adapter, split="val", max_groups=config.training.max_validation_groups
        )
        if config.training.validation_interval > 0
        else None
    )
    if not len(train_dataset):
        raise RuntimeError("The completed-file snapshot contains no training groups")
    # 创建 DataLoader 前完整预检 cache-only 数据，缺失观测会在训练启动前集中报错。
    cached_observations = preflight_observation_cache(
        adapter,
        ("train", train_dataset),
        ("val", validation_dataset),
    )
    if rank == 0:
        print(
            f"[cache-preflight] observations={cached_observations} status=ready",
            flush=True,
        )
    gripper = (
        create_gripper_provider(config, allow_generate=False)
        if config.ablation.use_gripper_scene_verifier
        else None
    )
    collator = UnifiedBatchCollator(config, gripper)
    train_sampler = (
        train_dataset.distributed_balanced_sampler(rank, world_size, config.training.seed)
        if world_size > 1 else train_dataset.balanced_sampler(config.training.seed)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=collator,
    )
    validation_sampler = (
        DistributedEvaluationSampler(len(validation_dataset), rank, world_size)
        if world_size > 1 else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=collator,
    ) if validation_dataset is not None and len(validation_dataset) else None
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    )
    if args.initialize:
        initialized = torch.load(args.initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(initialized.get("ema") or initialized.get("model") or initialized)
    if config.backbone.freeze and "encoder" not in config.training.frozen_modules:
        config.training.frozen_modules = (*config.training.frozen_modules, "encoder")
    if config.backbone.pretrained_checkpoint:
        pretrained = torch.load(config.backbone.pretrained_checkpoint, map_location="cpu",
                                weights_only=False)
        state = pretrained.get("ema") or pretrained.get("model") or pretrained
        encoder_state = {
            key.removeprefix("encoder."): value for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("Pretrained checkpoint contains no compatible encoder.* weights")
        model.encoder.load_state_dict(encoder_state, strict=True)
    backbone_parameters = list(model.encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]
    # PTv3 骨干和新建动作头采用两个参数组，骨干学习率更小以保护预训练表示。
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.optimizer.backbone_learning_rate},
            {"params": other_parameters, "lr": config.optimizer.learning_rate},
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
        adapter.capabilities, config.model, config.ablation, config.losses,
        config.region_head,
        config.training.generated_policy_candidate_ratio,
    )
    if rank == 0:
        enabled = {name: objective.total.enabled(name) for name in objective.total.DEFAULT_WEIGHTS}
        output = os.path.join(config.output_dir, "loss_routing.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({
                "dataset_capabilities": asdict(adapter.capabilities),
                "enabled": enabled,
                "disabled": {name: "dataset capability or ablation"
                             for name, value in enabled.items() if not value},
            }, handle, ensure_ascii=False, indent=2)
    trainer = Trainer(
        model, optimizer, config, objective, scheduler=scheduler, output_dir=config.output_dir
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    def validate(module: torch.nn.Module) -> dict[str, object]:
        if validation_loader is None:
            return {
                "score_sum": float("inf"), "score_count": 1,
                "metric_sums": {}, "metric_counts": {},
            }
        module.eval()
        total, count = 0.0, 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        with torch.no_grad():
            for raw in validation_loader:
                batch = trainer._move(raw, trainer.device)
                _, terms = objective(module, batch)
                score = sum(
                    weight * float(terms[f"loss_{family}"])
                    for family, weight in config.training.validation_family_weights.items()
                    if f"loss_{family}" in terms
                )
                total += score
                count += 1
                for key, value in terms.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                    metric_counts[key] = metric_counts.get(key, 0) + 1
                if count >= config.training.max_validation_groups:
                    break
        module.train()
        return {
            "score_sum": total,
            "score_count": count,
            "metric_sums": metric_sums,
            "metric_counts": metric_counts,
        }

    state = trainer.train(
        train_loader,
        validate=validate if validation_loader is not None else None,
        groups_per_effective_epoch=len(train_dataset),
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
