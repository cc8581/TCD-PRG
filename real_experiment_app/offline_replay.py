"""Replay one cached dataset observation through the real-experiment predictor."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.runtime import create_adapter

from .config import AppConfig
from .offline_ui import _camera_transform, _target_query
from .predictor import TCDPRGPredictor
from .types import FusedScene


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _candidate_rows(tensors: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index in np.flatnonzero(tensors["valid"][0].detach().cpu().numpy()):
        row = {"candidate_index": int(index)}
        for key, value in tensors.items():
            if key == "valid" or not hasattr(value, "shape") or value.ndim < 2:
                continue
            row[key] = _jsonable(value[0, index])
        rows.append(row)
    return rows


def _visualize(scene, target_query, candidates, selected, output: Path, show: bool) -> None:
    count = min(12000, len(scene.xyz_m))
    ids = np.linspace(0, len(scene.xyz_m) - 1, count, dtype=np.int64)
    xyz, rgb, instance = scene.xyz_m[ids], scene.rgb[ids], scene.instance_id[ids]
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    axes[0].scatter(*xyz.T, c=rgb, s=1)
    axes[0].set_title(f"Preprocessed XYZRGB ({len(scene.xyz_m):,} points)")
    palette = plt.get_cmap("tab20")((instance % 20).clip(min=0))
    palette[instance < 0] = (0.55, 0.55, 0.55, 0.15)
    axes[1].scatter(*xyz.T, c=palette, s=1)
    target = instance == int(target_query)
    axes[1].scatter(*xyz[target].T, c="red", s=3)
    axes[1].set_title(f"Predicted instances; target query={target_query}")
    axes[2].scatter(*xyz.T, c=rgb, s=0.6, alpha=0.25)
    for row in candidates:
        if int(row.get("type", -1)) == int(ActionType.PUSH):
            contact = np.asarray(row["contact_world"], float)
            direction = np.asarray(row["direction_world"], float)
            color = "red" if row["candidate_index"] == selected["candidate_index"] else "orange"
            axes[2].quiver(*contact, *direction, length=float(row["push_distance_m"]), color=color)
        elif np.isfinite(np.asarray(row.get("pose_world", [np.nan])[:3], float)).all():
            point = np.asarray(row["pose_world"][:3], float)
            color = "red" if row["candidate_index"] == selected["candidate_index"] else "cyan"
            axes[2].scatter(*point, c=color, marker="x", s=35)
    axes[2].set_title("Rule-generated PUSH arrows / grasp candidates")
    for axis in axes:
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.view_init(elev=28, azim=-55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-config", default="real_experiment_app/configs/real_experiment.yaml")
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--show", action="store_true", help="Keep the interactive inference visualization open"
    )
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    total_started = time.perf_counter()
    app = AppConfig.load(Path(args.app_config).resolve())
    section = app.raw["tcd_prg"]
    paths = __import__("yaml").safe_load(
        app.resolve(section["paths_config"]).read_text(encoding="utf-8")
    )
    overrides = [
        f"dataset.root={json.dumps(paths['dataset_root'], ensure_ascii=False)}",
        f"dataset.acronym_root={json.dumps(paths['acronym_root'], ensure_ascii=False)}",
        f"dataset.functional_region_root={json.dumps(paths['functional_region_root'], ensure_ascii=False)}",
        f"observation.pybullet_python={json.dumps(paths['pybullet_python'], ensure_ascii=False)}",
        f"cache.directory={json.dumps(paths['observation_cache_dir'], ensure_ascii=False)}",
    ]
    source_config = load_config(str(app.resolve(section["config"])), overrides)
    adapter = create_adapter(source_config, allow_render=False)
    started = time.perf_counter()
    observation = adapter.load_observation(args.scene_id, args.state_id, args.task_index)
    timings["cached_observation_load_seconds"] = time.perf_counter() - started
    observation.validate()
    source = observation.source_view
    if source is None:
        raise RuntimeError("Offline replay requires the formal source_view array")
    point_valid = (
        observation.point_valid
        if observation.point_valid is not None
        else np.ones(len(observation.xyz), bool)
    )
    keep = np.asarray(point_valid, bool)
    scene = FusedScene(
        np.asarray(observation.xyz[keep], np.float32),
        np.asarray(observation.rgb[keep], np.float32),
        np.full(int(keep.sum()), -1, np.int64),
        np.asarray(source[keep], np.int16),
        {},
        tuple(_camera_transform(camera) for camera in observation.camera_parameters),
    )
    target_cloud = observation.xyz[observation.target_mask & keep]
    if not len(target_cloud):
        raise RuntimeError("Selected dataset target has no visible point")
    target_click = target_cloud[
        np.linalg.norm(target_cloud - target_cloud.mean(0), axis=1).argmin()
    ]
    category = int(observation.object_category_id[observation.target_object])
    region = int(observation.task_region_id)

    started = time.perf_counter()
    predictor = TCDPRGPredictor(app)
    timings["model_load_seconds"] = time.perf_counter() - started
    sampled_count = len(predictor.policy.sensor_point_indices(scene.xyz_m))
    started = time.perf_counter()
    predictor.perceive(scene)
    timings["perception_seconds"] = time.perf_counter() - started
    target_query = _target_query(scene, target_click)
    prompt = predictor.policy.target_prompt_from_instance(
        scene.xyz_m, scene.instance_id, target_query
    )
    started = time.perf_counter()
    encoded = predictor.policy.encode_fused_scene(
        scene.xyz_m,
        scene.rgb,
        category,
        region,
        source_view=scene.source_view,
        camera_parameters=predictor._camera_parameters(scene),
        target_prompt_xyz=prompt,
        continue_target=False,
        enforce_target_confidence=True,
    )
    timings["integrated_encoding_and_grasp_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    generated = predictor.policy.generate_candidates(encoded)
    timings["rule_generation_and_push_scoring_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    selected = predictor.policy.select_action(generated)
    timings["action_selection_seconds"] = time.perf_counter() - started
    if selected is None:
        raise RuntimeError("No valid action was produced")
    candidates = _candidate_rows(generated["candidates"])
    timings["total_including_load_seconds"] = time.perf_counter() - total_started
    timings["warm_inference_seconds"] = sum(
        timings[key]
        for key in (
            "perception_seconds",
            "integrated_encoding_and_grasp_seconds",
            "rule_generation_and_push_scoring_seconds",
            "action_selection_seconds",
        )
    )
    result = {
        "source": {
            "scene_id": args.scene_id,
            "state_id": args.state_id,
            "task_index": args.task_index,
        },
        "input": {
            "cached_valid_points": len(scene.xyz_m),
            "deployment_grid_sampled_points": sampled_count,
            "source_views": sorted(np.unique(scene.source_view).astype(int).tolist()),
            "category_id": category,
            "functional_region_id": region,
            "operator_click_world_m": target_click.tolist(),
            "predicted_target_query": target_query,
        },
        "checkpoints": predictor.loaded_checkpoints,
        "preprocessing": {
            "dataset_observation": "formal cached observation produced by the configured adapter",
            "deployment": "deterministic grid sampling in TCDPRGPolicy",
            "stage_c": f"compiled PyTorch3D FPS to {predictor.model.push_fps_points} points",
            "gt_usage": "target mask only simulates the operator click; it is not passed to the model",
        },
        "candidate_count": len(candidates),
        "candidate_actions": candidates,
        "selected_action": _jsonable(selected),
        "timings": timings,
    }
    (output / "inference_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _visualize(
        scene,
        target_query,
        candidates,
        selected,
        output / "inference_visualization.png",
        args.show,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
