"""Evaluate one PUSH-evaluator checkpoint on the complete validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.losses.push_effectiveness import PushImprovementLoss
from tcd_prg.models import StandalonePushModel
from tcd_prg.models.staged_checkpoint import load_push_evaluator
from tcd_prg.runtime import create_adapter
from tcd_prg.scripts.train_push_evaluator import (
    _evaluate,
    _require_cached_observations,
    PushValueBatchCollator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    adapter = create_adapter(config, allow_render=False)
    validation_scenes = tuple(int(value) for value in adapter.scene_splits["val"])
    dataset = ActionStateGroupDataset(
        adapter,
        split="val",
        scene_ids=frozenset(validation_scenes),
        max_groups=None,
        allowed_strata=config.training.allowed_action_strata,
        global_grasp_mode="never",
    )
    if not dataset:
        raise RuntimeError("Complete validation split is empty")
    _require_cached_observations(adapter, dataset, dataset)
    loader = DataLoader(
        dataset,
        batch_size=config.training.validation_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.training.pin_memory,
        persistent_workers=False,
        collate_fn=PushValueBatchCollator(config, training=False),
    )
    model = StandalonePushModel(
        config.model, config.backbone, config.training.push_backbone
    ).to(device)
    load_push_evaluator(model, args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights_step = int(payload.get("optimizer_steps", -1))
    print(
        f"[full-validation] scenes={len(validation_scenes)} groups={len(dataset)} "
        f"weights_step={weights_step} checkpoint={Path(args.checkpoint).resolve()}",
        flush=True,
    )
    metrics = _evaluate(
        model,
        loader,
        device=device,
        config=config,
        loss_function=PushImprovementLoss(pos_weight=None),
        phase="full-final-weights",
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights_step": weights_step,
        "validation_scene_count": len(validation_scenes),
        "validation_group_count": len(dataset),
        **metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
