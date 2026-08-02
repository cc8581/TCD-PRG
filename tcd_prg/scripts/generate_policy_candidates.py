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
from tcd_prg.runtime import create_adapter, create_gripper_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-groups", type=int)
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
    # Generated caches preserve the generator's full open-world proposal set.
    # Robot approach/path certification belongs to the final executor, while
    # action outcomes come from teacher transitions or dedicated rollouts.
    policy = TCDPRGPolicy(model, config, gripper, certifier=None)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = cache_manifest(args.checkpoint, config, certification_scope="none")
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "format", "checkpoint_sha256", "generator_signature",
            "certification_scope", "label_version",
        ):
            if existing.get(key) != manifest[key]:
                raise ValueError(
                    f"Existing generated policy cache has incompatible {key}"
                )
        manifest["splits"] = existing.get("splits", {})
    positive = negative = unknown = conflict = written = 0
    groups_positive = groups_negative = effective_rows = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            policy.preparation_actions = min(
                config.planner.max_preparation_actions,
                int(sample.state_labels.sequence_depth),
            )
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
            valid = candidates["valid"][0].cpu()
            positive_mask = valid & (labels["label_status"] == 1)
            negative_mask = valid & (labels["label_status"] == 0)
            unknown_mask = valid & (labels["label_status"] < 0)
            conflict_mask = valid & labels["match_conflict"]
            has_positive = bool(positive_mask.any())
            has_negative = bool(negative_mask.any())
            positive += int(positive_mask.sum())
            negative += int(negative_mask.sum())
            unknown += int(unknown_mask.sum())
            conflict += int(conflict_mask.sum())
            groups_positive += int(has_positive)
            groups_negative += int(has_negative)
            effective_rows += int(has_positive and has_negative)
            written += 1
    manifest["splits"][args.split] = {
        "entries": written,
        "known_positive_candidates": positive,
        "known_negative_candidates": negative,
        "unknown_candidates": unknown,
        "conflict_unknown_candidates": conflict,
        "unmatched_unknown_candidates": unknown - conflict,
        "groups_with_known_positive": groups_positive,
        "groups_with_known_negative": groups_negative,
        "effective_policy_rows": effective_rows,
        "generated_positive_coverage": groups_positive / max(1, written),
        "effective_policy_row_coverage": effective_rows / max(1, written),
    }
    manifest["entries"] = len(tuple(output.glob("*.npz")))
    temporary = output / "manifest.work.json"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()
