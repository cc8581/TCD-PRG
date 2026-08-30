"""Run one formally certified TCD-PRG decision on a real cached state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcd_prg.baselines import create_baseline
from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.models import (
    TCDPRGModel,
    load_push_evaluator,
    load_staged_tcd_prg,
    resolve_staged_checkpoint_root,
)
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import (
    create_action_certifier,
    create_adapter,
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


def requires_robot_certification(action: dict[str, Any]) -> bool:
    """Only grasp actions use the current grasp/robot certifier."""
    return int(action["action_type"]) != int(ActionType.PUSH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint-root",
        help="Run directory containing checkpoints.json and the four stage subdirectories.",
    )
    parser.add_argument("--stage-a-checkpoint")
    parser.add_argument("--stage-b-checkpoint")
    parser.add_argument("--stage-c-checkpoint")
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-certification", action="store_true")
    parser.add_argument("--push-evaluator-checkpoint")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.checkpoint_root:
        explicit = (
            args.stage_a_checkpoint,
            args.stage_b_checkpoint,
            args.stage_c_checkpoint,
            args.push_evaluator_checkpoint,
        )
        if any(explicit):
            parser.error("--checkpoint-root cannot be combined with individual checkpoints")
        checkpoints = resolve_staged_checkpoint_root(args.checkpoint_root)
        args.stage_a_checkpoint = str(checkpoints["perception"])
        args.stage_b_checkpoint = str(checkpoints["grasp"])
        args.stage_c_checkpoint = str(checkpoints["push"])
        args.push_evaluator_checkpoint = str(checkpoints["push_evaluator"])
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    observation = adapter.load_observation(args.scene_id, args.state_id, args.task_index)
    certifier = None if args.no_certification else create_action_certifier(config)
    if config.baseline.type == "original_gapg_wrapper":
        policy = create_baseline(config)
    else:
        if not all((args.stage_a_checkpoint, args.stage_b_checkpoint, args.stage_c_checkpoint)):
            raise ValueError("learned inference requires all three staged checkpoints")
        if not args.push_evaluator_checkpoint:
            raise ValueError("learned PUSH inference requires --push-evaluator-checkpoint")
        device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        model = TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet).to(
            device
        )
        config.model.task_grasp_probability_threshold = load_staged_tcd_prg(
            model,
            args.stage_a_checkpoint,
            args.stage_b_checkpoint,
            args.stage_c_checkpoint,
            config,
        )
        load_push_evaluator(
            model,
            args.push_evaluator_checkpoint,
            proposal_checkpoint=args.stage_c_checkpoint,
        )
        model.to(device)
        # Candidate scoring remains robot-agnostic. The deterministic
        # controller exact-certifies grasp actions and falls through on rejection.
        base_policy = TCDPRGPolicy(model, config)
        policy = create_baseline(config, base_policy)
    encoded = policy.encode_observation(observation)
    candidates = policy.generate_candidates(encoded)
    # Certification is an execution-boundary operation, not part of Policy.
    # Rejected grasps are masked and fixed-priority selection is repeated without
    # rerunning the scene backbone.
    action = policy.select_action(candidates)
    final_certification = None
    if certifier is not None:
        certifier.set_observation(observation)
        while action is not None:
            if not requires_robot_certification(action):
                action["certification_skipped"] = True
                action["certification_reason"] = "push_effectiveness_evaluator_only"
                break
            accepted, reason = certifier.certify(action)
            if accepted:
                action["certified"] = True
                action["certification_reason"] = reason
                final_certification = (True, reason)
                break
            tensor_group = candidates.get("candidates")
            index = int(action["candidate_index"])
            tensor_group["valid"][0, index] = False
            candidates["certification_reasons"].append({"candidate_index": index, "reason": reason})
            action = policy.select_action(candidates)
    tensor_candidates = candidates.get("candidates") if isinstance(candidates, dict) else None
    valid_count = (
        int(tensor_candidates["valid"].sum())
        if isinstance(tensor_candidates, dict) and "valid" in tensor_candidates
        else len(candidates.get("candidates", []))
        if isinstance(candidates, dict)
        else 0
    )
    survival_keys = (
        "task_grasp_query_count",
        "task_grasp_after_nms_count",
        "unique_task_grasp_count",
    )
    task_grasp_survival = {
        key: int(tensor_candidates[key][0])
        for key in survival_keys
        if isinstance(tensor_candidates, dict) and key in tensor_candidates
    }
    payload = {
        "scene_id": args.scene_id,
        "state_id": args.state_id,
        "task_index": args.task_index,
        "selected_action": serializable(action),
        "valid_candidate_count": valid_count,
        "task_grasp_survival": task_grasp_survival,
        "certification_reasons": (
            candidates.get("certification_reasons", []) if isinstance(candidates, dict) else []
        ),
        "final_certification": final_certification,
        "checkpoints": (
            {
                "perception": str(Path(args.stage_a_checkpoint).resolve()),
                "grasp": str(Path(args.stage_b_checkpoint).resolve()),
                "push": str(Path(args.stage_c_checkpoint).resolve()),
                "push_evaluator": str(Path(args.push_evaluator_checkpoint).resolve()),
            }
            if args.stage_a_checkpoint
            else None
        ),
        "baseline": config.baseline.type,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
