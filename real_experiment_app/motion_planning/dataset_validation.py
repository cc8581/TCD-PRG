"""End-to-end validation: dataset cloud -> TCD-PRG grasp -> MoveIt trajectory."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.runtime import create_adapter

from ..config import AppConfig
from ..grasp_safety import grasp_local_z_is_downward_or_level
from ..offline_ui import _camera_transform, target_query_near_point
from ..predictor import TCDPRGPredictor
from ..types import FusedScene
from .client import WSLMoveItPlanner


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def load_dataset_scene(app: AppConfig, scene_id: int, state_id: int, task_index: int):
    section = app.raw["tcd_prg"]
    paths = yaml.safe_load(app.resolve(section["paths_config"]).read_text(encoding="utf-8"))
    overrides = [
        f"dataset.root={json.dumps(paths['dataset_root'], ensure_ascii=False)}",
        f"dataset.acronym_root={json.dumps(paths['acronym_root'], ensure_ascii=False)}",
        f"dataset.functional_region_root={json.dumps(paths['functional_region_root'], ensure_ascii=False)}",
        f"observation.pybullet_python={json.dumps(paths['pybullet_python'], ensure_ascii=False)}",
        f"cache.directory={json.dumps(paths['observation_cache_dir'], ensure_ascii=False)}",
    ]
    config = load_config(str(app.resolve(section["config"])), overrides)
    observation = create_adapter(config, allow_render=False).load_observation(
        scene_id, state_id, task_index
    )
    observation.validate()
    valid = observation.point_valid
    if valid is None:
        valid = np.ones(len(observation.xyz), bool)
    keep = np.asarray(valid, bool)
    if observation.source_view is None:
        raise RuntimeError("dataset observation has no source_view")
    scene = FusedScene(
        np.asarray(observation.xyz[keep], np.float32),
        np.asarray(observation.rgb[keep], np.float32),
        np.full(int(keep.sum()), -1, np.int64),
        np.asarray(observation.source_view[keep], np.int16),
        {},
        tuple(_camera_transform(camera) for camera in observation.camera_parameters),
    )
    target_visible = np.asarray(observation.target_mask, bool)[keep]
    target_cloud = scene.xyz_m[target_visible]
    if not len(target_cloud):
        raise RuntimeError("dataset target has no visible points")
    target_click = target_cloud[np.linalg.norm(target_cloud - target_cloud.mean(0), axis=1).argmin()]
    category = int(observation.object_category_id[observation.target_object])
    return scene, target_click, category, int(observation.task_region_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-config", default="real_experiment_app/configs/real_experiment.yaml")
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--execute", action="store_true", help="execute only the first successful plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    app = AppConfig.load(args.app_config)
    scene, target_click, category, region = load_dataset_scene(
        app, args.scene_id, args.state_id, args.task_index
    )
    predictor = TCDPRGPredictor(app)
    predictor.perceive(scene)
    target_query = target_query_near_point(scene, target_click)
    prediction = predictor.analyze(scene, target_query, category, region)
    candidates = [
        dict(candidate)
        for candidate in prediction.candidates
        if int(candidate["action_type"]) == int(ActionType.TASK_GRASP)
        and int(candidate["acted_object"]) == int(prediction.target_query)
        and grasp_local_z_is_downward_or_level(candidate)
    ]
    candidates.sort(key=lambda item: float(item.get("proposal_score", float("-inf"))), reverse=True)
    if not candidates:
        raise RuntimeError("TCD-PRG produced no direction-valid target grasp candidate")

    planner = WSLMoveItPlanner(timeout_s=90.0)
    attempts = []
    selected = None
    for candidate in candidates[: args.max_candidates]:
        result = planner.plan_grasp(
            grasp_pose_model_xyzw=candidate["grasp_pose_world"],
            model_tcp_to_robot_tcp=app.tcp_transform,
            scene_xyz_m=scene.xyz_m,
            scene_instance_id=scene.instance_id,
            target_instance=target_query,
            pregrasp_distance_m=float(app.raw["robot"]["pregrasp_distance_m"]),
            obstacle_voxel_size_m=0.02,
            target_voxel_size_m=0.01,
            collision_padding_m=0.0,
            table_z_m=0.0,
            execute=False,
        )
        attempts.append({
            "candidate_index": int(candidate["candidate_index"]),
            "candidate": _jsonable(candidate),
            "planning": _jsonable(result.__dict__ if hasattr(result, "__dict__") else {
                field: getattr(result, field) for field in result.__dataclass_fields__
            }),
        })
        if result.success:
            selected = candidate
            if args.execute:
                executed = planner.execute_plan(result.plan_id)
                attempts[-1]["execution"] = _jsonable({
                    field: getattr(executed, field) for field in executed.__dataclass_fields__
                })
                if not executed.success:
                    raise RuntimeError(f"fake-controller execution failed: {executed.message}")
            break

    report = {
        "success": selected is not None,
        "source": {
            "scene_id": args.scene_id,
            "state_id": args.state_id,
            "task_index": args.task_index,
            "points": len(scene.xyz_m),
        },
        "target": {
            "query": int(prediction.target_query),
            "instance": int(target_query),
            "category": category,
            "region": region,
            "operator_click_world_m": target_click.tolist(),
        },
        "predicted_candidate_count": len(candidates),
        "selected_candidate": _jsonable(selected),
        "attempts": attempts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)
    output.write_text(payload, encoding="utf-8")
    print(payload)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
