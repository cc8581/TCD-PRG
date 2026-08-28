"""Train PUSH effectiveness independently from a frozen Stage-C checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators import push_effectiveness_metrics
from tcd_prg.evaluators.push_effectiveness import (
    proposal_known_outcome_masks,
    push_candidate_ranking_counts,
)
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import StandalonePushModel, push_condition_from_gt
from tcd_prg.models.staged_checkpoint import (
    PUSH_EVALUATOR_PROTOCOL_VERSION,
    load_push_stage,
    push_checkpoint_fingerprint,
)
from tcd_prg.planners.push_decoder import decode_push_candidates
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter
from tcd_prg.trainers import (
    freeze_push_proposal,
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
        eligible, _ = push_effectiveness_eligibility(
            batch,
            condition,
            max_contact_distance_m=config.model.push_contact_match_max_distance_m,
        )
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
    """Validate logged classification and deployment-equivalent ranking."""
    model.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    group_ids: list[torch.Tensor] = []
    weighted_loss = 0.0
    evaluated = 0
    group_offset = 0
    positive_sets = 0
    top1_evaluable = 0
    top5_evaluable = 0
    candidate_hit1 = 0
    candidate_hit5 = 0
    for cpu_batch in loader:
        batch = _device(cpu_batch, device)
        loss, details = push_effectiveness_batch_loss(
            model,
            batch,
            instance_queries=config.model.instance_queries,
            loss_function=loss_function,
            max_contact_distance_m=config.model.push_contact_match_max_distance_m,
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

        proposal_batch = dict(batch)
        proposal_batch["push_condition"] = push_condition_from_gt(
            batch, config.model.instance_queries
        )
        proposal = model(proposal_batch, forward_mode="push")
        _, final_rows = decode_push_candidates(
            proposal["sensor"],
            proposal["push_condition"],
            proposal["push"],
            config.model,
            use_push_potential=config.ablation.use_push_potential,
        )
        for row_index, row in enumerate(final_rows):
            if len(row["point_index"]):
                candidate_logits = model.push_evaluator(
                    proposal["push"], row, batch_index=row_index
                )
                row["effective_logit"] = candidate_logits
                row["effective_probability"] = torch.sigmoid(candidate_logits)
        positive_masks, _, known_masks = proposal_known_outcome_masks(
            final_rows,
            batch,
            contact_threshold_m=config.evaluation.push_match_contact_m,
            direction_threshold_deg=config.evaluation.push_match_direction_deg,
        )
        ranking = push_candidate_ranking_counts(final_rows, positive_masks, known_masks)
        positive_sets += int(ranking["push_evaluator_positive_candidate_set_count"])
        top1_evaluable += int(ranking["push_evaluator_top1_evaluable_count"])
        top5_evaluable += int(ranking["push_evaluator_top5_evaluable_count"])
        candidate_hit1 += int(ranking["push_evaluator_hit_at_1_count"])
        candidate_hit5 += int(ranking["push_evaluator_recall_at_5_count"])
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
    result["push_evaluator_oracle_positive_candidate_set_count"] = float(positive_sets)
    result["push_evaluator_oracle_top1_evaluable_rate"] = (
        top1_evaluable / positive_sets if positive_sets else float("nan")
    )
    result["push_evaluator_oracle_top5_evaluable_rate"] = (
        top5_evaluable / positive_sets if positive_sets else float("nan")
    )
    result["push_evaluator_oracle_hit_at_1"] = (
        candidate_hit1 / top1_evaluable if top1_evaluable else float("nan")
    )
    result["push_evaluator_oracle_recall_at_5"] = (
        candidate_hit5 / top5_evaluable if top5_evaluable else float("nan")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--proposal-checkpoint", required=True)
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
    validation_dataset = ActionStateGroupDataset(
        adapter,
        split="val",
        max_groups=config.training.max_validation_groups,
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
    model = StandalonePushModel(config.model).to(device)
    load_push_stage(model, args.proposal_checkpoint, config)
    proposal_fingerprint, proposal_source = push_checkpoint_fingerprint(args.proposal_checkpoint)
    freeze_push_proposal(model)
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
        num_workers=config.training.validation_num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.validation_num_workers > 0,
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
                max_contact_distance_m=config.model.push_contact_match_max_distance_m,
            )
            if not int(details["effective_logit"].numel()):
                continue
            made_progress = True
            loss.backward()
            optimizer.step()
            step += 1
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
            "selection_metric": "push_evaluator_auprc",
            "proposal_state_fingerprint": proposal_fingerprint,
            "proposal_state_source": proposal_source,
        },
        output,
    )


if __name__ == "__main__":
    main()
