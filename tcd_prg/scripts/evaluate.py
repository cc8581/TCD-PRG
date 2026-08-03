"""Offline evaluation entry point with JSON/CSV and grouped reports."""

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
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter, create_gripper_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter, split=args.split, max_groups=config.evaluation.max_groups
    )
    if not len(dataset):
        raise RuntimeError(f"No completed action groups exist for split={args.split}")
    gripper = create_gripper_provider(config, allow_generate=False) \
        if config.ablation.use_gripper_scene_verifier else None
    loader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers, pin_memory=config.training.pin_memory,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=UnifiedBatchCollator(config, gripper, training=False),
    )
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"] or checkpoint["model"])
    model.eval()
    evaluator = OfflineModelEvaluator(
        config.model, config.evaluation.bootstrap_samples, config.evaluation.confidence,
        config.graph, config.evaluation,
    )
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in raw.items()}
            if "action_parameters" in batch:
                batch["action_parameters"] = {
                    key: value.to(device) for key, value in raw["action_parameters"].items()
                }
            if "verifier_inputs" in batch:
                batch["verifier_inputs"] = {
                    key: value.to(device) for key, value in raw["verifier_inputs"].items()
                }
            start = time.perf_counter()
            output = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output["_inference_time_s_per_sample"] = (
                time.perf_counter() - start
            ) / batch["xyz"].shape[0]
            evaluator.update(batch, output)
    evaluator.finalize_closed_loop_replay(config.evaluation.horizons)
    evaluator.export(args.output_dir, asdict(config))


if __name__ == "__main__":
    main()
