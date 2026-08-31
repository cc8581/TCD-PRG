"""Train independent PUSH evaluation exclusively on logged evaluated actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators import push_effectiveness_metrics
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import StandalonePushModel, push_condition_from_gt
from tcd_prg.models.staged_checkpoint import PUSH_EVALUATOR_PROTOCOL_VERSION, load_push_evaluator
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter
from tcd_prg.trainers import (
    freeze_perception_geometry,
    push_effectiveness_batch_loss,
    push_effectiveness_eligibility,
)
from tcd_prg.trainers.reproducibility import seed_everything


def _device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    return value


def _counts(loader: DataLoader, config) -> tuple[int, int]:
    """Count the same deterministic, representable population used by the loss."""
    positive = negative = 0
    for batch in loader:
        condition = push_condition_from_gt(batch, config.model.instance_queries)
        eligible = push_effectiveness_eligibility(batch, condition)
        target = batch["action_improves_state"][eligible].bool()
        positive += int(target.sum())
        negative += int((~target).sum())
    return positive, negative


@torch.no_grad()
def _evaluate(
    model: StandalonePushModel,
    loader: DataLoader,
    *,
    device: torch.device,
    config,
    loss_function: PushEffectivenessLoss,
) -> dict[str, float]:
    """Validate logged actions only; candidate generation is a separate evaluation."""
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    group_ids: list[torch.Tensor] = []
    weighted_loss = 0.0
    evaluated = 0
    group_offset = 0
    for cpu_batch in loader:
        batch = _device(cpu_batch, device)
        loss, details = push_effectiveness_batch_loss(
            model,
            batch,
            instance_queries=config.model.instance_queries,
            loss_function=loss_function,
        )
        logits = details["effective_logit"].detach()
        target = details["effective_target"].detach().bool()
        local_group = details["effective_group_index"].detach().long()
        count = int(logits.numel())
        if count:
            probabilities.append(torch.sigmoid(logits))
            targets.append(target)
            group_ids.append(local_group + group_offset)
            weighted_loss += float(loss.detach()) * count
            evaluated += count
        sensor = batch.get("model_inputs", batch)
        group_offset += int(sensor["point_mask"].shape[0])

    if not evaluated:
        raise RuntimeError("Validation split contains no evaluated PUSH actions")
    probability = torch.cat(probabilities)
    target = torch.cat(targets)
    if not bool(target.any()) or not bool((~target).any()):
        raise RuntimeError("Validation split requires both positive and negative PUSH actions")
    metrics = push_effectiveness_metrics(probability, target, torch.cat(group_ids))
    logged_names = {
        "push_evaluator_hit_at_1": "push_evaluator_logged_hit_at_1",
        "push_evaluator_recall_at_5": "push_evaluator_logged_recall_at_5",
        "push_evaluator_precision_at_1": "push_evaluator_logged_precision_at_1",
    }
    result = {
        logged_names.get(key, key): float(value.detach().cpu()) for key, value in metrics.items()
    }
    result["push_evaluator_loss"] = weighted_loss / evaluated
    result["push_evaluator_evaluated_count"] = float(evaluated)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage/push_evaluator.yaml")
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument(
        "--pretrain-checkpoint",
        help="Initialize the PUSH evaluator weights while starting a fresh optimizer at step 0.",
    )
    parser.add_argument("--output", default="outputs/push_evaluator.pt")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(config.training.seed, config.training.deterministic)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
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
    count_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    positives, negatives = _counts(count_loader, config)
    if positives == 0:
        raise RuntimeError("No evaluated positive PUSH action exists in the training split")
    if negatives == 0:
        raise RuntimeError("No evaluated negative PUSH action exists in the training split")
    if not len(validation_dataset):
        raise RuntimeError("Formal PUSH evaluator training requires a non-empty val split")
    model = StandalonePushModel(config.model, config.backbone).to(device)
    model.load_perception_geometry(args.perception_checkpoint)
    freeze_perception_geometry(model)
    pretrain_checkpoint = args.pretrain_checkpoint or config.training.pretrain_checkpoint
    if pretrain_checkpoint:
        load_push_evaluator(model, pretrain_checkpoint)
        print(
            "[pretrain] initialized PUSH evaluator from "
            f"{Path(pretrain_checkpoint).resolve()}; "
            "optimizer and step start fresh",
            flush=True,
        )
    loss_function = PushEffectivenessLoss(negatives / positives)
    optimizer = torch.optim.AdamW(
        model.push_evaluator.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
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
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.validation_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    final_validation_loader = DataLoader(
        final_validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.validation_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    step = 0
    best_auprc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    last_validation_step = -1
    validation_interval = max(0, int(config.training.validation_interval))
    while step < config.training.max_optimizer_steps:
        made_progress = False
        for cpu_batch in loader:
            batch = _device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = push_effectiveness_batch_loss(
                model,
                batch,
                instance_queries=config.model.instance_queries,
                loss_function=loss_function,
            )
            if not int(details["effective_logit"].numel()):
                continue
            made_progress = True
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite PUSH evaluator loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.push_evaluator.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                count = details["effective_logit"].numel()
                print(f"[push-evaluator-train step={step}] loss={float(loss.detach()):.6f} actions={count}", flush=True)
            if validation_interval > 0 and step % validation_interval == 0:
                metrics = _evaluate(
                    model,
                    validation_loader,
                    device=device,
                    config=config,
                    loss_function=loss_function,
                )
                last_validation_step = step
                score = metrics["push_evaluator_auprc"]
                print(f"[push-evaluator-val step={step}] {metrics}", flush=True)
                if math.isfinite(score) and score > best_auprc:
                    best_auprc = score
                    best_metrics = dict(metrics)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.push_evaluator.state_dict().items()
                    }
                model.push_evaluator.train()
            if step >= config.training.max_optimizer_steps:
                break
        if not made_progress:
            raise RuntimeError(
                "A complete PUSH evaluator epoch contained no known evaluated PUSH actions"
            )
    if last_validation_step != step:
        metrics = _evaluate(
            model,
            validation_loader,
            device=device,
            config=config,
            loss_function=loss_function,
        )
        print(f"[push-evaluator-val step={step}] {metrics}", flush=True)
        score = metrics["push_evaluator_auprc"]
        if math.isfinite(score) and score > best_auprc:
            best_auprc = score
            best_metrics = dict(metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.push_evaluator.state_dict().items()
            }
    if best_state is None or best_metrics is None:
        raise RuntimeError("PUSH evaluator validation did not produce a finite AUPRC")
    model.push_evaluator.load_state_dict(best_state, strict=True)
    final_metrics = _evaluate(
        model,
        final_validation_loader,
        device=device,
        config=config,
        loss_function=loss_function,
    )
    print(f"[push-evaluator-final-val] {final_metrics}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "training_stage": "push_evaluator",
            "push_evaluator_protocol_version": PUSH_EVALUATOR_PROTOCOL_VERSION,
            "positive_count": positives,
            "negative_count": negatives,
            "pos_weight": negatives / positives,
            "optimizer_steps": step,
            "validation_metrics": best_metrics,
            "final_validation_metrics": final_metrics,
            "periodic_validation_scene_count": len(periodic_validation_scenes),
            "final_validation_scene_count": len(validation_scenes),
            "selection_metric": "push_evaluator_auprc",
            "perception_geometry_fingerprint": model.perception_geometry_fingerprint,
        },
        output,
    )


if __name__ == "__main__":
    main()
