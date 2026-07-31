"""Evaluate the task-free global grasp branch independently of TCD-PRG routing."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.evaluators import GlobalGraspEvaluator
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import create_action_certifier, create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    config.model.candidate_topk = args.topk
    adapter = create_adapter(config, allow_render=False)
    observation = adapter.load_observation(args.scene_id, args.state_id, args.task_index)
    truth = adapter.load_global_grasps(
        args.scene_id, args.state_id, observation, training=False
    )
    if truth is None:
        raise RuntimeError("This dataset has no task-free global grasp ground truth")

    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
    policy = TCDPRGPolicy(model.eval(), config)
    with torch.no_grad():
        encoded = policy.encode_observation(observation)
        predictions = policy.predict_global_grasps(encoded)[: args.topk]

    if args.certify:
        certifier = create_action_certifier(config)
        certifier.set_observation(observation)
        certified = []
        for prediction in predictions:
            executable, _ = certifier.certify({
                "action_type": int(ActionType.PICK_REMOVE),
                "acted_object": prediction.object_index,
                "grasp_pose_world": prediction.grasp_pose_world,
                "grasp_width_m": prediction.width_m,
            })
            certified.append(replace(prediction, certified=bool(executable)))
        predictions = certified

    evaluator = GlobalGraspEvaluator()
    metrics = evaluator.evaluate(predictions, truth, certified=False)
    if args.certify:
        metrics.update(evaluator.evaluate(predictions, truth, certified=True))
    serializable_metrics = {
        key: (float(value) if math.isfinite(float(value)) else None)
        for key, value in metrics.items()
    }
    payload = {
        "scene_id": args.scene_id,
        "state_id": args.state_id,
        "task_index": args.task_index,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "truth_count_after_nms": int(truth.valid_mask.sum()),
        "prediction_count_before_common_nms": len(predictions),
        "metrics": serializable_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
