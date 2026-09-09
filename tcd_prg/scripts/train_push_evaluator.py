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
from tcd_prg.datasets.push_value import PUSH_IMPROVEMENT_SCHEMA_VERSION
from tcd_prg.datasets.push_value import PushImprovementStore
from tcd_prg.losses.push_effectiveness import PushImprovementLoss
from tcd_prg.models import StandalonePushModel
from tcd_prg.push_improvement import PUSH_IMPROVEMENT_DEFINITION
from tcd_prg.models.staged_checkpoint import load_push_evaluator
from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint
from tcd_prg.trainers.push_progress import PushTrainingProgress, append_record, print_validation_summary
from tcd_prg.trainers.push_scheduler import PushLRScheduler
from tcd_prg.trainers.push_sampling import compiled_fps
from tcd_prg.runtime import PushValueBatchCollator, create_adapter
from tcd_prg.observation.cached import ObservationCacheMissError
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


def push_optimizer_groups(model: StandalonePushModel, config) -> list[dict[str, Any]]:
    """Keep the pretrained PointNet++ step size separate from random PUSH heads."""
    backbone = list(model.push_evaluator.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    heads = [
        parameter for parameter in model.push_evaluator.parameters()
        if id(parameter) not in backbone_ids
    ]
    if not backbone or not heads:
        raise RuntimeError("PUSH optimizer requires nonempty backbone and head parameter groups")
    return [
        {"params": heads, "lr": config.optimizer.learning_rate, "name": "push_heads"},
        {"params": backbone, "lr": config.optimizer.backbone_learning_rate, "name": "pointnet2_backbone"},
    ]


def _restrict_to_cached_informative_overfit_groups(dataset, root: str, limit: int) -> None:
    """Select fixed cache-only groups containing both binary classes."""
    store = PushImprovementStore(root)
    payloads: dict[int, dict[str, np.ndarray]] = {}
    selected = []
    for index, unit in enumerate(dataset.units):
        try:
            sample = dataset[index]
        except ObservationCacheMissError:
            continue
        payload = payloads.setdefault(unit.scene_id, store.load_scene(unit.scene_id))
        action_ids = np.asarray(payload["action_id"], np.int64)
        candidates = np.asarray(sample.candidates.candidate_action_ids, np.int64)
        locations = np.searchsorted(action_ids, candidates)
        matched = locations < len(action_ids)
        if len(action_ids):
            matched &= action_ids[np.minimum(locations, len(action_ids) - 1)] == candidates
        valid_locations = locations[matched]
        valid_locations = valid_locations[np.asarray(payload["improvement_valid"])[valid_locations]]
        if len(valid_locations) < 2:
            continue
        labels = np.asarray(payload["improvement_target"])[valid_locations]
        if not (bool((labels == 0).any()) and bool((labels == 1).any())):
            continue
        selected.append(unit)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"Only {len(selected)} cached informative PUSH groups were found; requested {limit}"
        )
    dataset.units = tuple(selected)
    dataset.selected_group_count = len(selected)
    print(
        f"[push-overfit] fixed {len(selected)} cached groups containing both classes",
        flush=True,
    )


def accumulated_batches(model, loader, *, device, config, loss_function, optimizer):
    """Accumulate only microbatches that contain valid binary supervision."""
    limit = config.training.gradient_accumulation_steps
    micro = actions = positives = 0
    loss_sum = data_seconds = 0.0
    component_sums = dict(bce=0.0)
    optimizer.zero_grad(set_to_none=True)
    finished = time.monotonic()
    for cpu_batch in loader:
        data_seconds += time.monotonic() - finished
        batch = _device(cpu_batch, device)
        loss, details = push_effectiveness_batch_loss(
            model,
            batch,
            instance_queries=config.model.instance_queries,
            loss_function=loss_function,
            scene_sample_points=config.training.push_fps_points,
        )
        valid = details["improvement_valid"]
        count = int(valid.sum())
        if count:
            if not torch.isfinite(loss) or not torch.isfinite(details["improvement_logit"]).all():
                raise RuntimeError("Non-finite PUSH improvement training values/loss")
            (loss * count).backward()
            micro += 1
            actions += count
            positives += int((details["improvement_target"][valid] > 0.5).sum())
            loss_sum += float(loss.detach()) * count
            component_sums["bce"] += float(details["push_improvement_bce"]) * count
        if micro == limit:
            for parameter in model.push_evaluator.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(actions)
            yield (
                loss_sum / actions,
                actions,
                positives,
                data_seconds,
                {key: value / actions for key, value in component_sums.items()},
            )
            optimizer.zero_grad(set_to_none=True)
            micro = actions = positives = 0
            loss_sum = data_seconds = 0.0
            component_sums = dict.fromkeys(component_sums, 0.0)
        finished = time.monotonic()
    if micro:
        for parameter in model.push_evaluator.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(actions)
        yield (
            loss_sum / actions,
            actions,
            positives,
            data_seconds,
            {key: value / actions for key, value in component_sums.items()},
        )


@torch.no_grad()
def _evaluate(model, loader, *, device, config, loss_function, phase="periodic",
              use_batch_norm_batch_statistics=False) -> dict[str, float]:
    """Evaluate binary PUSH improvement on actions and state-level selection."""
    model.eval()
    if use_batch_norm_batch_statistics:
        for module in model.push_evaluator.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
    started=time.monotonic(); logits=[]; targets=[]; groups=[]; supervised=0; group_offset=0
    weighted_loss=0.0
    with tqdm(total=len(loader),desc=f"Val [push_evaluator] [{phase}]",unit="batch",
              dynamic_ncols=True,mininterval=1.0) as progress:
        for cpu_batch in loader:
            batch=_device(cpu_batch,device)
            loss,details=push_effectiveness_batch_loss(
                model,batch,instance_queries=config.model.instance_queries,
                loss_function=loss_function,scene_sample_points=config.training.push_fps_points)
            logit=details["improvement_logit"].detach().float().cpu()
            target=details["improvement_target"].detach().float().cpu()
            valid=details["improvement_valid"].detach().bool().cpu()
            local_group=details["effective_group_index"].detach().long().cpu()
            if not torch.isfinite(logit).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite PUSH improvement validation value")
            count=int(valid.sum())
            if count:
                logits.append(logit[valid]);targets.append(target[valid])
                groups.append(local_group[valid]+group_offset)
                weighted_loss+=float(loss)*count;supervised+=count
            sensor=batch.get("model_inputs",batch);group_offset+=int(sensor["point_mask"].shape[0])
            progress.set_postfix(actions=supervised,loss=weighted_loss/max(supervised,1),refresh=False)
            progress.update(1)
    if not supervised: raise RuntimeError("Validation contains no valid PUSH improvement targets")
    logit=torch.cat(logits);target=torch.cat(targets);group=torch.cat(groups);prob=logit.sigmoid()
    positive=target.bool();negative=~positive
    order=torch.argsort(prob,descending=True,stable=True); sorted_prob=prob[order]
    sorted_y=target[order];tp=sorted_y.cumsum(0);fp=(1-sorted_y).cumsum(0)
    precision_curve=tp/(tp+fp);recall_curve=tp/max(int(positive.sum()),1)
    distinct=torch.ones_like(sorted_prob,dtype=torch.bool)
    distinct[:-1]=sorted_prob[:-1]!=sorted_prob[1:]
    boundaries=torch.where(distinct)[0]
    if not bool(positive.any() and negative.any()):
        auroc=auprc=float("nan")
    else:
        previous_recall=torch.cat((recall_curve.new_zeros(1),recall_curve[boundaries[:-1]]))
        auprc=float(((recall_curve[boundaries]-previous_recall)*precision_curve[boundaries]).sum())
        ascending=torch.argsort(prob,stable=True);ascending_prob=prob[ascending]
        _,counts=torch.unique_consecutive(ascending_prob,return_counts=True)
        ends=counts.cumsum(0).float();starts=ends-counts.float()+1
        sorted_average_ranks=torch.repeat_interleave((starts+ends)/2,counts)
        average_ranks=torch.empty_like(prob);average_ranks[ascending]=sorted_average_ranks
        pos_count=int(positive.sum());neg_count=int(negative.sum())
        auroc=float((average_ranks[positive].sum()-pos_count*(pos_count+1)/2)/(pos_count*neg_count))
    f1_curve=2*precision_curve*recall_curve/(precision_curve+recall_curve).clamp_min(1e-12)
    candidate_f1=f1_curve[boundaries];best_index=int(candidate_f1.argmax())
    boundary=int(boundaries[best_index]);best_f1=float(candidate_f1[best_index])
    best_threshold=float(sorted_prob[boundary]);best_precision=float(precision_curve[boundary])
    best_recall=float(recall_curve[boundary]);tn=int(negative.sum())-int(fp[boundary])
    specificity=tn/max(int(negative.sum()),1)
    top1=[];top3=[];random1=[]
    for gid in torch.unique(group):
        ids=torch.where(group==gid)[0]
        if not bool(positive[ids].any()): continue
        ranked=ids[torch.argsort(prob[ids],descending=True,stable=True)]
        top1.append(float(positive[ranked[0]]));top3.append(float(positive[ranked[:3]].any()))
        random1.append(float(positive[ids].float().mean()))
    return {
        "push_evaluator_auroc":auroc,"push_evaluator_auprc":auprc,
        "push_evaluator_threshold":best_threshold,"push_evaluator_f1":best_f1,
        "push_evaluator_precision":best_precision,"push_evaluator_recall":best_recall,
        "push_evaluator_balanced_accuracy":(best_recall+specificity)/2,
        "push_evaluator_top1_improvement_rate":float(np.mean(top1)) if top1 else float("nan"),
        "push_evaluator_top3_contains_improvement_rate":float(np.mean(top3)) if top3 else float("nan"),
        "push_evaluator_random_top1_improvement_rate":float(np.mean(random1)) if random1 else float("nan"),
        "push_evaluator_loss":weighted_loss/supervised,
        "push_evaluator_positive_rate":float(target.mean()),
        "push_evaluator_evaluated_count":float(supervised),
        "push_evaluator_logged_group_count":float(torch.unique(group).numel()),
        "push_evaluator_validation_seconds":time.monotonic()-started,
    }

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
    # Batched relation GEMMs change reduction order. Use full FP32 throughout
    # this independent stage so cuDNN TF32 does not amplify those small changes
    # in the PointNet++ backward pass. A/B entry points are unaffected.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    # Fail before loading data when the required compiled operator is unavailable.
    compiled_fps()(torch.zeros(1, 1, 3, device=device), K=1)
    print(f'PUSH training input: up to {fps_points} points via compiled PyTorch3D FPS', flush=True)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split="train",
        max_groups=(
            None if config.training.push_overfit_validate_train
            else config.training.max_train_groups
        ),
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    if config.training.push_overfit_validate_train:
        if config.training.max_train_groups is None:
            raise RuntimeError("PUSH overfit diagnostic requires training.max_train_groups")
        _restrict_to_cached_informative_overfit_groups(
            dataset,
            config.training.push_improvement_root,
            int(config.training.max_train_groups),
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
    if config.training.push_overfit_validate_train:
        validation_dataset = dataset
        final_validation_dataset = dataset
        periodic_validation_scenes = tuple(sorted({item.scene_id for item in dataset.units}))
        validation_scenes = periodic_validation_scenes
    else:
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
            # A bounded validation set marks a diagnostic run. Formal configs use
            # None and therefore still evaluate the complete validation split.
            max_groups=config.training.max_validation_groups,
            allowed_strata=config.training.allowed_action_strata,
            global_grasp_mode="never",
        )
    if not len(dataset):
        raise RuntimeError("PUSH evaluator training requires a non-empty training split")
    if not len(validation_dataset):
        raise RuntimeError("Formal PUSH evaluator training requires a non-empty val split")
    print(f"[push-evaluator-init] train_groups={len(dataset)} "
          f"validation_groups={len(validation_dataset)}; "
          "single-head binary PUSH improvement with BCE; fine-tuning yanx27 PointNet++", flush=True)
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
    loss_function = PushImprovementLoss(
        pos_weight=config.training.push_improvement_pos_weight
    )
    optimizer = torch.optim.AdamW(
        push_optimizer_groups(model, config), weight_decay=config.optimizer.weight_decay,
    )
    scheduler = PushLRScheduler(optimizer, config.scheduler.warmup_steps, config.training.max_optimizer_steps)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=PushValueBatchCollator(config, training=True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        # The training loader keeps its workers alive.  On Windows, starting a
        # second 2*num_workers pool here duplicates the Python/Torch address
        # space and can exhaust host commit memory before the second validation
        # batch.  Validation is infrequent and deterministic, so load it in the
        # owner process without changing samples, batch size, or metrics.
        num_workers=0,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=PushValueBatchCollator(config, training=False),
    )
    final_validation_loader = DataLoader(
        final_validation_dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=PushValueBatchCollator(config, training=False),
    )
    signature = asdict(config)
    # Device/worker changes and a longer run do not change the training objective.
    for name in (
        "device", "num_workers", "validation_workers", "validation_batch_size",
        "pin_memory", "max_optimizer_steps",
    ):
        signature["training"].pop(name, None)
    checkpoints = PushTrainingCheckpoint(args.output, model, {
        "arithmetic_precision": "fp32_no_tf32",
        "training_scene_sampling": {"operator": "pytorch3d.compiled_fps", "points": fps_points},
        "periodic_validation_scene_count": len(periodic_validation_scenes),
        "final_validation_scene_count": len(validation_scenes),
        "selection_metric": "push_evaluator_auprc",
        "selection_policy": "highest binary improvement AUPRC",
        "push_improvement_schema_version": PUSH_IMPROVEMENT_SCHEMA_VERSION,
        "push_improvement_definition": PUSH_IMPROVEMENT_DEFINITION,
    }, {"config": signature,
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
            best_score = checkpoints.best_metrics["push_evaluator_auprc"] if checkpoints.best_metrics else float("nan")
            print_validation_summary(metrics, step, best_score, phase)
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
             f"eta excludes validation; objective is binary BCE only; "
             f"best checkpoint uses AUPRC; output: {config.output_dir}", flush=True)
    model.train()
    if args.resume and checkpoints.validation_due(step, validation_interval):
        print(f"[resume] periodic validation at step={step} is pending; validating before training", flush=True)
        run_periodic_validation()
    elif args.resume:
        print(f"[resume] checkpoint step={step} is in the training phase; training continues first", flush=True)
    while step < config.training.max_optimizer_steps:
        made_progress = False
        for mean_loss, count, positives, data_seconds, components in accumulated_batches(
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
            progress.add(mean_loss, count, positives, components=components,
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
