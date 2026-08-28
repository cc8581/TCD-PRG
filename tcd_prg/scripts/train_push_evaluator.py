"""Train PUSH effectiveness independently from a frozen Stage-C checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators import push_effectiveness_metrics
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import StandalonePushModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter
from tcd_prg.trainers import freeze_push_proposal, push_effectiveness_batch_loss


def _device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    return value


def _counts(dataset: ActionStateGroupDataset) -> tuple[int, int]:
    positive = negative = 0
    for sample in dataset:
        group = sample.candidates
        known_push = (
            group.valid_mask
            & (group.action_type == int(ActionType.PUSH))
            & (group.evaluation_status != int(CandidateStatus.UNKNOWN_UNTESTED))
        )
        positive += int(group.action_improves_state[known_push].sum())
        negative += int(known_push.sum()) - int(group.action_improves_state[known_push].sum())
    return positive, negative


@torch.no_grad()
def _evaluate(
    model: StandalonePushModel,
    loader: DataLoader,
    *,
    device: torch.device,
    instance_queries: int,
    loss_function: PushEffectivenessLoss,
) -> dict[str, float]:
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
            instance_queries=instance_queries,
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
    result = {key: float(value.detach().cpu()) for key, value in metrics.items()}
    result["push_evaluator_loss"] = weighted_loss / evaluated
    result["push_evaluator_evaluated_count"] = float(evaluated)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/push_evaluator.pt")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split="train",
        max_groups=config.training.max_train_groups,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    validation_dataset = ActionStateGroupDataset(
        adapter,
        split="val",
        max_groups=config.training.max_validation_groups,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    positives, negatives = _counts(dataset)
    if positives == 0:
        raise RuntimeError("No evaluated positive PUSH action exists in the training split")
    if negatives == 0:
        raise RuntimeError("No evaluated negative PUSH action exists in the training split")
    if not len(validation_dataset):
        raise RuntimeError("Formal PUSH evaluator training requires a non-empty val split")
    model = StandalonePushModel(config.model).to(device)
    payload = torch.load(args.proposal_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    proposal_state = {key: value for key, value in state.items() if key.startswith("push.")}
    missing, unexpected = model.load_state_dict(proposal_state, strict=False)
    if unexpected or any(not key.startswith("push_evaluator.") for key in missing):
        raise RuntimeError(
            f"Incompatible Stage-C checkpoint: missing={missing}, extra={unexpected}"
        )
    freeze_push_proposal(model)
    loss_function = PushEffectivenessLoss(negatives / positives)
    optimizer = torch.optim.AdamW(
        model.push_evaluator.parameters(), lr=config.optimizer.learning_rate
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=UnifiedBatchCollator(config, training=True, include_graspnet=False),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.validation_num_workers,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    step = 0
    best_auprc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    last_validation_step = -1
    validation_interval = max(0, int(config.training.validation_interval))
    while step < config.training.max_optimizer_steps:
        for cpu_batch in loader:
            batch = _device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = push_effectiveness_batch_loss(
                model,
                batch,
                instance_queries=config.model.instance_queries,
                loss_function=loss_function,
            )
            loss.backward()
            optimizer.step()
            step += 1
            if validation_interval > 0 and step % validation_interval == 0:
                metrics = _evaluate(
                    model,
                    validation_loader,
                    device=device,
                    instance_queries=config.model.instance_queries,
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
    if last_validation_step != step:
        metrics = _evaluate(
            model,
            validation_loader,
            device=device,
            instance_queries=config.model.instance_queries,
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "training_stage": "push_evaluator",
            "positive_count": positives,
            "negative_count": negatives,
            "pos_weight": negatives / positives,
            "optimizer_steps": step,
            "validation_metrics": best_metrics,
            "selection_metric": "push_evaluator_auprc",
        },
        output,
    )


if __name__ == "__main__":
    main()
