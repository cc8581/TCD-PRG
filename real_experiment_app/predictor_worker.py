from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from .config import AppConfig
from .predictor import TCDPRGPredictor
from .types import FusedScene


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def respond(value):
    print(json.dumps(serializable(value), ensure_ascii=False), flush=True)


def load_scene(path: str) -> FusedScene:
    with np.load(path, allow_pickle=False) as data:
        mapping = {
            int(k): int(v)
            for k, v in zip(
                data["category_keys"], data["category_values"], strict=True
            )
        }
        return FusedScene(
            data["xyz_m"].copy(),
            data["rgb"].copy(),
            data["instance_id"].copy(),
            data["source_view"].copy(),
            mapping,
            tuple(
                matrix.copy()
                for matrix in (
                    data["camera_to_world"]
                    if "camera_to_world" in data.files
                    else np.empty((0, 4, 4), np.float32)
                )
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    predictor = TCDPRGPredictor(AppConfig.load(args.config))
    respond({"ready": True})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "perceive":
                scene = predictor.perceive(load_scene(request["scene"]))
                result = {
                    "instance_id": scene.instance_id,
                    "category_by_instance": scene.category_by_instance,
                }
            elif command == "predict":
                scene = load_scene(request["scene"])
                prediction = predictor.predict(
                    scene,
                    int(request["target"]),
                    int(request["category"]),
                    int(request["region"]),
                    int(request["required"]),
                )
                result = {
                    "action": prediction.action,
                    "inference_seconds": prediction.inference_seconds,
                }
            elif command == "action_executed":
                predictor.policy.update_after_action(request["action"], None)
                result = True
            elif command == "reset":
                predictor.reset()
                result = True
            elif command == "close":
                respond({"ok": True, "result": True})
                return
            else:
                raise ValueError(f"Unknown command {command}")
            respond({"ok": True, "result": result})
        except Exception as error:
            respond({
                "ok": False,
                "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            })


if __name__ == "__main__":
    main()
