"""Run one formally certified TCD-PRG decision on a real cached state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcd_prg.config import load_config
from tcd_prg.baselines import create_baseline
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import (
    create_action_certifier,
    create_adapter,
    create_gripper_provider,
)


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-certification", action="store_true")
    parser.add_argument("--allow-gripper-generate", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    observation = adapter.load_observation(args.scene_id, args.state_id, args.task_index)
    certifier = None if args.no_certification else create_action_certifier(config)
    if config.baseline.type == "original_gapg_wrapper":
        policy = create_baseline(config)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for learned TCD-PRG candidates")
        device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        model = TCDPRGModel(
            config.model, config.ablation, config.graph, config.router, config.backbone
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
        gripper = (create_gripper_provider(config, args.allow_gripper_generate)
                   if config.ablation.use_gripper_scene_verifier else None)
        base_policy = TCDPRGPolicy(model, config, gripper, certifier)
        policy = create_baseline(config, base_policy)
    encoded = policy.encode_observation(observation)
    candidates = policy.generate_candidates(encoded)
    action = policy.select_action(candidates)
    final_certification = None
    if config.baseline.type == "original_gapg_wrapper" and certifier is not None and action is not None:
        certifier.set_observation(observation)
        final_certification = certifier.certify(action)
        if not final_certification[0]:
            action = None
    tensor_candidates = candidates.get("candidates") if isinstance(candidates, dict) else None
    valid_count = (int(tensor_candidates["valid"].sum())
                   if isinstance(tensor_candidates, dict) and "valid" in tensor_candidates
                   else len(candidates.get("candidates", [])) if isinstance(candidates, dict) else 0)
    payload = {
        "scene_id": args.scene_id, "state_id": args.state_id,
        "task_index": args.task_index, "selected_action": serializable(action),
        "valid_candidate_count": valid_count,
        "certification_reasons": (candidates.get("certification_reasons", [])
                                  if isinstance(candidates, dict) else []),
        "final_certification": final_certification,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "baseline": config.baseline.type,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
