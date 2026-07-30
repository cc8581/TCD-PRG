"""Unified full-model and ablation training entry point."""

from __future__ import annotations

import argparse
import math
import os
import json
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Subset

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset, DistributedEvaluationSampler
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter, create_gripper_provider
from tcd_prg.trainers import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and config.training.device.startswith("cuda")
    if world_size > 1:
        backend = config.training.ddp_backend
        if backend == "auto":
            backend = "gloo" if os.name == "nt" else ("nccl" if use_cuda else "gloo")
        if use_cuda and local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices exist"
            )
        init_method = os.environ.get("TCD_DDP_INIT_METHOD")
        if init_method:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
        else:
            torch.distributed.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            config.training.device = f"cuda:{local_rank}"
    adapter = create_adapter(config, allow_render=False)
    train_dataset = ActionStateGroupDataset(
        adapter, split="train", max_groups=config.training.max_train_groups
    )
    validation_dataset = ActionStateGroupDataset(
        adapter, split="val", max_groups=config.training.max_validation_groups
    )
    if not len(train_dataset):
        raise RuntimeError("The completed-file snapshot contains no training groups")
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
    ) if len(validation_dataset) else None
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    )
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
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.optimizer.backbone_learning_rate},
            {"params": other_parameters, "lr": config.optimizer.learning_rate},
        ],
        weight_decay=config.optimizer.weight_decay,
    )

    def learning_rate(step: int) -> float:
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
    objective = TCDPRGObjective(
        adapter.capabilities, config.model, config.ablation, asdict(config.losses),
        config.region_head,
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
    if args.dry_run:
        raw = next(iter(train_loader))
        batch = trainer._move(raw, trainer.device)
        loss, terms = objective(trainer.model, batch)
        loss.backward()
        print({"loss": float(loss.detach()), "terms": len(terms), "batch": len(raw["samples"])})
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    def validate(module: torch.nn.Module) -> tuple[float, int]:
        if validation_loader is None:
            return float("inf"), 1
        module.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for raw in validation_loader:
                batch = trainer._move(raw, trainer.device)
                loss, _ = objective(module, batch)
                total += float(loss)
                count += 1
                if count >= config.training.max_validation_groups:
                    break
        module.train()
        return total, count

    state = trainer.train(
        train_loader,
        validate=validate if validation_loader is not None else None,
        groups_per_effective_epoch=len(train_dataset),
    )
    trainer.save_checkpoint(f"{config.output_dir}/last.pt")
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        print(asdict(state))
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
