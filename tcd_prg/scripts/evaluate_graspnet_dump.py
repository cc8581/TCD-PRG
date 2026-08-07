"""Run the official GraspNet evaluator on a GraspNet-format prediction dump.

This command intentionally delegates the complete official evaluation path to
``graspnetAPI.GraspNetEval`` rather than reimplementing it with TCD labels. The
official ``eval_grasp`` path applies 3 cm / 30 degree grasp NMS, object
association, Top-10 per object / scene Top-50 selection, collision and
force-closure scoring, and AP aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graspnet-root", required=True)
    parser.add_argument("--dump-folder", required=True)
    parser.add_argument("--camera", choices=("kinect", "realsense"), required=True)
    parser.add_argument("--split", choices=("all", "seen", "similar", "novel"), default="all")
    parser.add_argument("--proc", type=int, default=4)
    parser.add_argument("--output", default="outputs/graspnet_metrics.json")
    args = parser.parse_args()

    try:
        from graspnetAPI import GraspNetEval
    except ImportError as error:
        raise RuntimeError(
            "Strict GraspNet comparison requires the official graspnetAPI package/source. "
            "Install graspnet/graspnetAPI in the evaluation environment."
        ) from error

    evaluator = GraspNetEval(root=args.graspnet_root, camera=args.camera, split="test")
    function = {
        "all": evaluator.eval_all,
        "seen": evaluator.eval_seen,
        "similar": evaluator.eval_similar,
        "novel": evaluator.eval_novel,
    }[args.split]
    result, ap = function(args.dump_folder, proc=args.proc)
    accuracy = np.asarray(result, dtype=np.float64)
    if accuracy.ndim < 2 or accuracy.shape[-1] != 6:
        raise RuntimeError(
            f"Unexpected GraspNetEval accuracy shape {accuracy.shape}; expected friction axis of 6"
        )
    payload = {
        "protocol": "official GraspNetEval",
        "camera": args.camera,
        "split": args.split,
        "standard_graspnet_AP": float(np.mean(accuracy)),
        # Official friction axis is [0.2,0.4,0.6,0.8,1.0,1.2].
        "standard_graspnet_AP_mu_0.4": float(np.mean(accuracy[..., 1])),
        "standard_graspnet_AP_mu_0.8": float(np.mean(accuracy[..., 3])),
    }
    if args.split == "all":
        # eval_all returns [all, seen, similar, novel]; expose all four rather
        # than coercing the list to float.
        split_ap = np.asarray(ap, dtype=np.float64).reshape(-1)
        if len(split_ap) == 4:
            payload.update({
                "standard_graspnet_AP_seen": float(split_ap[1]),
                "standard_graspnet_AP_similar": float(split_ap[2]),
                "standard_graspnet_AP_novel": float(split_ap[3]),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
