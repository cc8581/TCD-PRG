"""Train independent PUSH evaluation exclusively on logged evaluated actions."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tcd_prg.trainers.push_workers import PushDataLoader as DataLoader, managed_push_workers
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators import push_effectiveness_metrics
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import StandalonePushModel
from tcd_prg.models.staged_checkpoint import load_push_evaluator
from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint
from tcd_prg.trainers.push_progress import PushTrainingProgress, append_record, print_validation_summary
from tcd_prg.trainers.push_scheduler import PushLRScheduler
from tcd_prg.trainers.push_sampling import compiled_fps
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter
from tcd_prg.trainers import (
    push_effectiveness_batch_loss,
)
from tcd_prg.trainers.reproducibility import seed_everything


def _device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    return value


def accumulated_batches(model, loader, *, device, config, loss_function, optimizer):
    """Accumulate action sums, then normalize by actual known-action count.

    Empty microbatches do not consume accumulation slots. An epoch tail is
    flushed rather than dropped. All optimizer/checkpoint counters count updates.
    """
    limit = config.training.gradient_accumulation_steps
    micro = actions = positives = 0
    loss_sum = data_seconds = 0.
    optimizer.zero_grad(set_to_none=True)
    finished = time.monotonic()
    for cpu_batch in loader:
        data_seconds += time.monotonic() - finished
        batch = _device(cpu_batch, device)
        loss, details = push_effectiveness_batch_loss(
            model, batch, instance_queries=config.model.instance_queries, loss_function=loss_function,
            scene_sample_points=config.training.push_fps_points)
        logits = details['effective_logit']
        count = logits.numel()
        if count:
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite PUSH training logits/loss')
            (loss * count).backward()
            micro += 1
            actions += count
            positives += int(details['effective_target'].sum())
            loss_sum += float(loss.detach()) * count
        if micro == limit:
            for parameter in model.push_evaluator.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(actions)
            yield loss_sum / actions, actions, positives, data_seconds
            optimizer.zero_grad(set_to_none=True)
            micro = actions = positives = 0
            loss_sum = data_seconds = 0.
        finished = time.monotonic()
    if micro:
        for parameter in model.push_evaluator.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(actions)
        yield loss_sum / actions, actions, positives, data_seconds


@torch.no_grad()
def _evaluate(
    model: StandalonePushModel,
    loader: DataLoader,
    *,
    device: torch.device,
    config,
    loss_function: PushEffectivenessLoss,
    phase: str = "periodic",
) -> dict[str, float]:
    """Validate logged actions only; candidate generation is a separate evaluation."""
    model.eval()
    started = time.monotonic()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    group_ids: list[torch.Tensor] = []
    weighted_loss = 0.0
    evaluated = 0
    group_offset = 0
    with tqdm(total=len(loader), desc=f"Val [push_evaluator] [{phase}]", unit="batch",
              dynamic_ncols=True, mininterval=1.0) as progress:
        for cpu_batch in loader:
            batch = _device(cpu_batch, device)
            loss, details = push_effectiveness_batch_loss(
                model,
                batch,
                instance_queries=config.model.instance_queries,
                loss_function=loss_function,
            )
            # Keep accumulated validation predictions off GPU.
            logits = details["effective_logit"].detach().float().cpu()
            target = details["effective_target"].detach().bool().cpu()
            local_group = details["effective_group_index"].detach().long().cpu()
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite PUSH validation logits/loss")
            count = int(logits.numel())
            if count:
                probabilities.append(torch.sigmoid(logits))
                targets.append(target)
                group_ids.append(local_group + group_offset)
                weighted_loss += float(loss.detach()) * count
                evaluated += count
            sensor = batch.get("model_inputs", batch)
            group_offset += int(sensor["point_mask"].shape[0])
            progress.set_postfix(actions=evaluated, loss=weighted_loss / max(evaluated, 1), refresh=False)
            progress.update(1)

    if not evaluated:
        raise RuntimeError("Validation split contains no evaluated PUSH actions")
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    if not bool(target.any()) or not bool((~target).any()):
        raise RuntimeError("Validation split requires both positive and negative PUSH actions")
    print(f"Val [push_evaluator]  aggregating: {evaluated} actions", flush=True)
    metrics = push_effectiveness_metrics(probability, target, torch.cat(group_ids))
    result = {key: float(value) for key, value in metrics.items()}
    result["push_evaluator_loss"] = weighted_loss / evaluated
    result["push_evaluator_evaluated_count"] = float(evaluated)
    result["push_evaluator_logged_empty_group_count"] = group_offset - result["push_evaluator_logged_group_count"]
    result["push_evaluator_validation_seconds"] = time.monotonic() - started
    return result


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage/push_evaluator.yaml")
    parser.add_argument(
        "--pretrain-checkpoint",
        help="Initialize the PUSH evaluator weights while starting a fresh optimizer at step 0.",
    )
    parser.add_argument("--output", default="outputs/push_evaluator.pt")
    parser.add_argument("--resume", help="Continue weights, optimizer and steps from a *_last.pt checkpoint; reshuffle data.")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Deprecated compatibility option; PUSH restart snapshots are validation transactions.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    args.output = str(Path(args.output).expanduser().resolve())
    # One run directory for checkpoints, logs AND augmentation debug snapshots.
    config.output_dir = str(Path(args.output).parent)
    fps_points = config.training.push_fps_points
    if isinstance(fps_points, bool) or not isinstance(fps_points, int) or fps_points <= 0:
        parser.error("training.push_fps_points must be a positive integer")
    if config.dataset.scene_points > 0 and fps_points > config.dataset.scene_points:
        parser.error("training.push_fps_points must not exceed dataset.scene_points")
    if config.training.perception_checkpoint:
        parser.error("PointNet++ PUSH does not use training.perception_checkpoint; remove it")
    if config.training.amp:
        parser.error("PointNet++ PUSH currently uses FP32; set training.amp=false")
    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")
    if args.resume and (args.pretrain_checkpoint or config.training.pretrain_checkpoint):
        parser.error("--resume and pretrain_checkpoint are mutually exclusive")
    seed_everything(config.training.seed, config.training.deterministic)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    # Fail before loading data when the required compiled operator is unavailable.
    compiled_fps()(torch.zeros(1, 1, 3, device=device), K=1)
    print(f'PUSH training input: up to {fps_points} points via compiled PyTorch3D FPS', flush=True)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split="train",
        max_groups=config.training.max_train_groups,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    validation_scenes = tuple(int(value) for value in adapter.scene_splits["val"])
    requested_scenes = config.training.validation_scene_count
    if requested_scenes is None or requested_scenes >= len(validation_scenes):
        periodic_validation_scenes = validation_scenes
    else:
        periodic_validation_scenes = tuple(
            sorted(
                int(value)
                for value in np.random.default_rng(
                    config.training.validation_scene_seed
                ).permutation(validation_scenes)[: int(requested_scenes)]
            )
        )
    validation_dataset = ActionStateGroupDataset(
        adapter,
        split="val",
        scene_ids=frozenset(periodic_validation_scenes),
        max_groups=config.training.max_validation_groups,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    final_validation_dataset = ActionStateGroupDataset(
        adapter,
        split="val",
        scene_ids=frozenset(validation_scenes),
        max_groups=None,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    if not len(dataset):
        raise RuntimeError("PUSH evaluator training requires a non-empty training split")
    if not len(validation_dataset):
        raise RuntimeError("Formal PUSH evaluator training requires a non-empty val split")
    print(f"[push-evaluator-init] train_groups={len(dataset)} "
          f"validation_groups={len(validation_dataset)} pos_weight=1.0; "
          "no label-count prescan; fine-tuning yanx27 PointNet++", flush=True)
    model = StandalonePushModel(config.model).to(device)
    pretrain_checkpoint = args.pretrain_checkpoint or config.training.pretrain_checkpoint
    if not pretrain_checkpoint and not args.resume:
        provenance = model.push_evaluator.backbone.load_pretrained()
        print(f"[pretrain] loaded complete yanx27 S3DIS network: {provenance}", flush=True)
    if pretrain_checkpoint:
        load_push_evaluator(model, pretrain_checkpoint)
        print(
            "[pretrain] initialized PUSH evaluator from "
            f"{Path(pretrain_checkpoint).resolve()}; "
            "optimizer and step start fresh",
            flush=True,
        )
    loss_function = PushEffectivenessLoss(pos_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.push_evaluator.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = PushLRScheduler(optimizer, config.scheduler.warmup_steps, config.training.max_optimizer_steps)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=UnifiedBatchCollator(config, training=True, include_graspnet=False),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        num_workers=config.training.validation_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    final_validation_loader = DataLoader(
        final_validation_dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        num_workers=config.training.validation_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    signature = asdict(config)
    # Device/worker changes and a longer run do not change the training objective.
    for name in (
        "device", "num_workers", "validation_workers", "validation_batch_size",
        "pin_memory", "max_optimizer_steps",
    ):
        signature["training"].pop(name, None)
    checkpoints = PushTrainingCheckpoint(args.output, model, {
        "training_scene_sampling": {"operator": "pytorch3d.compiled_fps", "points": fps_points},
        "pos_weight": 1.0,
        "periodic_validation_scene_count": len(periodic_validation_scenes),
        "final_validation_scene_count": len(validation_scenes),
        "selection_metric": "push_evaluator_ap",
    }, {"config": signature, "pos_weight": 1.0,
        "periodic_scenes": periodic_validation_scenes, "final_scenes": validation_scenes}, scheduler=scheduler)
    step = checkpoints.restore(args.resume, optimizer) if args.resume else 0
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "resolved_config.yaml").write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
    subset_path = Path(args.output).with_name("validation_scene_subset.json")
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(
        json.dumps(
            {
                "seed": config.training.validation_scene_seed,
                "source_scene_count": len(validation_scenes),
                "selected_scene_count": len(periodic_validation_scenes),
                "scene_ids": periodic_validation_scenes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.resume:
        print(f"[resume] restored optimizer and step={step}", flush=True)
    last_validation_step = checkpoints.last_completed_validation_step
    validation_interval = max(0, int(config.training.validation_interval))
    log_interval = max(1, int(config.logging.log_interval))
    progress = PushTrainingProgress(Path(args.output).parent, config.training.max_optimizer_steps, step)

    def validate(loader, phase):
        progress.pause()
        try:
            metrics = _evaluate(model, loader, device=device, config=config,
                                loss_function=loss_function, phase=phase)
            if phase == "periodic":
                checkpoints.consider_best(metrics, step)
            best_ap = checkpoints.best_metrics["push_evaluator_ap"] if checkpoints.best_metrics else float("nan")
            print_validation_summary(metrics, step, best_ap, phase)
            append_record(Path(args.output).parent / "validation_metrics.jsonl",
                          {"optimizer_step": step, "phase": phase,
                           "weights_step": checkpoints.best_step if phase == "final" else step, **metrics})
            return metrics
        finally:
            progress.resume()

    def run_periodic_validation():
        nonlocal last_validation_step
        # BEGIN and COMMIT are separate durable snapshots. An interruption
        # between them resumes this same validation before any training batch.
        checkpoints.begin_validation(optimizer, step)
        metrics = validate(validation_loader, "periodic")
        checkpoints.complete_validation(optimizer, step)
        last_validation_step = step
        model.train()
        return metrics

    print(f"[push-evaluator-train] starting at step={step}; log_interval={log_interval}, "
          f"batch={config.training.batch_size}, "
          f"validation_batch={config.training.validation_batch_size}, "
          f"workers={config.training.num_workers}; waiting for first batch", flush=True)
    print(f"[push-evaluator-train] scheduler: warmup({config.scheduler.warmup_steps}) + cosine; "
          f"eta excludes validation; H@k+ uses groups containing positives; output: {config.output_dir}", flush=True)
    model.train()
    if args.resume and checkpoints.validation_due(step, validation_interval):
        print(f"[resume] periodic validation at step={step} is pending; validating before training", flush=True)
        run_periodic_validation()
    elif args.resume:
        print(f"[resume] checkpoint step={step} is in the training phase; training continues first", flush=True)
    while step < config.training.max_optimizer_steps:
        made_progress = False
        for mean_loss, count, positives, data_seconds in accumulated_batches(
            model, loader, device=device, config=config, loss_function=loss_function,
            optimizer=optimizer,
        ):
            made_progress = True
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.push_evaluator.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True))
            clip_scale = min(1., config.training.gradient_clip_norm / (grad_norm + 1e-6))
            optimizer.step()
            scheduler.step()
            step += 1
            progress.add(mean_loss, count, positives,
                         gradient_norm=grad_norm, clip_scale=clip_scale, data_seconds=data_seconds,
                         max_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.)
            if (step == 1 or step % log_interval == 0 or step >= config.training.max_optimizer_steps
                    or (validation_interval > 0 and step % validation_interval == 0)):
                progress.log(step, optimizer.param_groups[0]["lr"])
            if validation_interval > 0 and step % validation_interval == 0:
                run_periodic_validation()
            if step >= config.training.max_optimizer_steps:
                break
        if not made_progress:
            raise RuntimeError(
                "A complete PUSH evaluator epoch contained no known evaluated PUSH actions"
            )
    if last_validation_step != step:
        run_periodic_validation()
    checkpoints.save_best()
    model.push_evaluator.load_state_dict(checkpoints.best_state, strict=True)
    final_metrics = validate(final_validation_loader, "final")
    checkpoints.save_best(final_metrics)


def main() -> None:
    with managed_push_workers():
        _main()


if __name__ == "__main__":
    main()
