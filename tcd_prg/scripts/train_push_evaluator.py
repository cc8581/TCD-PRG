"""Train PUSH effectiveness independently from a frozen Stage-C checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
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
        known_push = group.valid_mask & (group.action_type == 0) & (group.evaluation_status >= 0)
        positive += int(group.action_improves_state[known_push].sum())
        negative += int(known_push.sum()) - int(group.action_improves_state[known_push].sum())
    return positive, negative


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
    positives, negatives = _counts(dataset)
    if positives == 0:
        raise RuntimeError("No evaluated positive PUSH action exists in the training split")
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
    step = 0
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
            if step >= config.training.max_optimizer_steps:
                break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.push_evaluator.state_dict(),
            "training_stage": "push_evaluator",
            "positive_count": positives,
            "negative_count": negatives,
            "pos_weight": negatives / positives,
        },
        output,
    )


if __name__ == "__main__":
    main()
