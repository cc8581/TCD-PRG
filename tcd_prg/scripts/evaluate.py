"""Offline evaluation entry point for standard metrics supported by native labels."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.models import TCDPRGModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument(
        "--log-interval-groups",
        type=int,
        default=320,
        help="Print progress after approximately this many evaluated groups.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_interval_groups <= 0:
        raise ValueError("--log-interval-groups must be positive")
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split=args.split,
        max_groups=config.evaluation.max_groups,
        global_grasp_width_bounds=(
            config.model.min_grasp_width_m,
            config.model.max_grasp_width_m,
        ),
    )
    if not len(dataset):
        raise RuntimeError(f"No completed action groups exist for split={args.split}")
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=False),
    )
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model = TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "task_grasp_probability_threshold" in checkpoint:
        config.model.task_grasp_probability_threshold = float(
            checkpoint["task_grasp_probability_threshold"]
        )
    model.load_state_dict(checkpoint["ema"] or checkpoint["model"])
    model.eval()
    evaluator = OfflineModelEvaluator(
        config.model,
        config.evaluation.bootstrap_samples,
        config.evaluation.confidence,
        config.evaluation,
    )
    evaluated_groups = 0
    next_log = args.log_interval_groups
    started = time.time()
    print(
        f"[evaluation-start] split={args.split} groups={len(dataset)} "
        f"batch={config.training.batch_size} workers={config.training.num_workers}",
        flush=True,
    )
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw.items()
            }
            if "action_parameters" in batch:
                batch["action_parameters"] = {
                    key: value.to(device) for key, value in raw["action_parameters"].items()
                }
            if "verifier_inputs" in batch:
                batch["verifier_inputs"] = {
                    key: value.to(device) for key, value in raw["verifier_inputs"].items()
                }
            evaluator.update(batch, model(batch, forward_mode="perception"))
            batch_groups = int(batch["xyz"].shape[0])
            evaluated_groups += batch_groups
            if evaluated_groups >= next_log or evaluated_groups == len(dataset):
                elapsed = max(time.time() - started, 1e-9)
                rate = evaluated_groups / elapsed
                remaining = max(len(dataset) - evaluated_groups, 0)
                print(
                    f"[evaluation] groups={evaluated_groups}/{len(dataset)} "
                    f"rate={rate:.2f}/s eta={remaining / max(rate, 1e-9):.0f}s",
                    flush=True,
                )
                while next_log <= evaluated_groups:
                    next_log += args.log_interval_groups
    evaluator.export(args.output_dir, asdict(config))
    print(
        f"[evaluation-done] groups={evaluated_groups} "
        f"elapsed={time.time() - started:.1f}s output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
