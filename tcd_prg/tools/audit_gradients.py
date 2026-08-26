"""Audit per-loss gradients on the shared scene encoder from a real checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.diagnostics import family_gradient_norms
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter


def _move(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    config = load_config(args.config, args.overrides)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter, split="train", max_groups=args.batches * config.training.batch_size
    )
    loader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0,
        collate_fn=UnifiedBatchCollator(config, training=False),
    )
    model = TCDPRGModel(
        config.model, config.ablation, config.backbone, config.graspnet
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("ema") or checkpoint.get("model") or checkpoint)
    model.train()
    objective = TCDPRGObjective(
        adapter.capabilities, config.model, config.ablation, config.losses,
        config.region_head,
    ).to(device)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        total, _, family_losses = objective(
            model, batch, return_family_losses=True
        )
        norms = family_gradient_norms(
            family_losses, tuple(model.encoder.parameters()), total
        )
        for name, value in norms.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
        model.zero_grad(set_to_none=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "batches": len(loader),
        "shared_parameter_scope": "encoder",
        "weighted_family_gradient_norm_mean": {
            name: sums[name] / counts[name] for name in sorted(sums)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
