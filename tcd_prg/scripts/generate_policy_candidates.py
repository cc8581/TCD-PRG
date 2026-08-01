"""Generate deployment-path policy candidates and tri-state supervision caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset, collate_unified
from tcd_prg.datasets.policy_candidates import (
    cache_manifest,
    match_generated_candidates,
    save_candidate_entry,
)
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import create_action_certifier, create_adapter, create_gripper_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--skip-exact-certification", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter, split=args.split, max_groups=args.max_groups
    )
    if not len(dataset):
        raise RuntimeError(f"No completed action groups exist for split={args.split}")
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("ema") or checkpoint.get("model") or checkpoint)
    model.eval()
    gripper = (
        create_gripper_provider(config, allow_generate=False)
        if config.ablation.use_gripper_scene_verifier else None
    )
    exact = config.planner.exact_certification and not args.skip_exact_certification
    certifier = create_action_certifier(config) if exact else None
    policy = TCDPRGPolicy(model, config, gripper, certifier)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = cache_manifest(args.checkpoint, config.model, exact_certification=exact)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "format", "checkpoint_sha256", "generator_signature",
            "exact_certification", "certifier_version",
        ):
            if existing.get(key) != manifest[key]:
                raise ValueError(
                    f"Existing generated policy cache has incompatible {key}"
                )
        manifest["splits"] = existing.get("splits", {})
    positive = negative = unknown = written = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            policy.preparation_actions = min(
                config.planner.max_preparation_actions,
                int(sample.state_labels.sequence_depth),
            )
            policy.previous_action = None
            encoded = policy.encode_observation(sample.observation)
            group = policy.generate_candidates(encoded)
            candidates = group["candidates"]
            candidates["evidence"] = candidates["evidence"].clone()
            verifier = encoded.output.get("verifier")
            if verifier is not None:
                candidates["evidence"][..., 2] = torch.sigmoid(verifier["overall_logit"])
            teacher = collate_unified([sample])
            labels = match_generated_candidates(candidates, teacher, 0, config.model)
            save_candidate_entry(output, sample, candidates, labels)
            positive += int(labels["policy_success"].sum())
            negative += int((labels["label_status"] == 0).sum())
            unknown += int((labels["label_status"] < 0).sum())
            written += 1
    manifest["splits"][args.split] = {
        "entries": written,
        "known_positive_candidates": positive,
        "known_negative_candidates": negative,
        "unknown_candidates": unknown,
    }
    manifest["entries"] = len(tuple(output.glob("*.npz")))
    temporary = output / "manifest.work.json"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()
