"""Dataset replay adapter for the production real-experiment UI."""

# ruff: noqa: E501
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from tcd_prg.config import load_config
from tcd_prg.runtime import create_adapter

from .task_semantics import dataset_root_from_app_config
from .types import FusedScene


def local_scene_ids(app_config) -> tuple[int, ...]:
    """List only scenes with both published geometry and task labels."""
    root = Path(dataset_root_from_app_config(app_config))
    scenes = root / "task_clutter_scenes_20_categories"
    labels = root / "task_positive_multistep_sequences" / "scene_labels"
    if not scenes.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"本地场景数据集目录不存在或不完整: {root}")
    result = []
    for path in labels.glob("scene_*.h5"):
        suffix = path.stem.removeprefix("scene_")
        if suffix.isdigit() and (scenes / f"scene_{int(suffix):04d}" / "scene.npz").is_file():
            result.append(int(suffix))
    return tuple(sorted(set(result)))


def local_scene_shape(app_config, scene_id: int) -> tuple[int, int]:
    """Return state and task counts without loading/rendering an observation."""
    root = Path(dataset_root_from_app_config(app_config))
    path = root / "task_positive_multistep_sequences" / "scene_labels" / f"scene_{scene_id:04d}.h5"
    with h5py.File(path, "r", swmr=True) as handle:
        scene = next(iter(handle.values()))
        return len(scene["states"]["object_pose"]), len(scene["catalog"]["task_object_index"])


def _camera_transform(camera) -> np.ndarray:
    z = np.asarray(camera.target_world, np.float64) - np.asarray(camera.eye_world, np.float64)
    z /= np.linalg.norm(z)
    y = -np.asarray(camera.up_world, np.float64)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.column_stack((x, y, z))
    transform[:3, 3] = np.asarray(camera.eye_world, np.float32)
    return transform


def target_query_near_point(scene: FusedScene, target_point: np.ndarray) -> int:
    distance = np.linalg.norm(scene.xyz_m - target_point[None], axis=1)
    for index in np.argsort(distance):
        query = int(scene.instance_id[index])
        if query >= 0:
            return query
    raise RuntimeError("感知结果没有为目标点击附近分配有效实例")


def _offline_adapter(app_config):
    section = app_config.raw["tcd_prg"]
    paths = __import__("yaml").safe_load(
        app_config.resolve(section["paths_config"]).read_text(encoding="utf-8")
    )
    overrides = [
        f"dataset.root={json.dumps(paths['dataset_root'], ensure_ascii=False)}",
        f"dataset.acronym_root={json.dumps(paths['acronym_root'], ensure_ascii=False)}",
        f"dataset.functional_region_root={json.dumps(paths['functional_region_root'], ensure_ascii=False)}",
        f"observation.pybullet_python={json.dumps(paths['pybullet_python'], ensure_ascii=False)}",
        f"cache.directory={json.dumps(paths['observation_cache_dir'], ensure_ascii=False)}",
    ]
    config = load_config(str(app_config.resolve(section["config"])), overrides)
    return create_adapter(config, allow_render=False)


def cached_states_for_task(app_config, scene_id: int, task_index: int) -> tuple[int, ...]:
    """List exact cached state requests; never select or render one implicitly."""
    adapter = _offline_adapter(app_config)
    state_count, _ = local_scene_shape(app_config, int(scene_id))
    keys = [(int(scene_id), state, int(task_index)) for state in range(state_count)]
    available = adapter.observations_available(keys)
    return tuple(state for _, state, task in keys if available[(int(scene_id), state, task)])


def load_offline_scene(controller, scene_id, state_id, task_index):
    adapter = _offline_adapter(controller.config)
    started = time.perf_counter()
    observation = adapter.load_observation(int(scene_id), int(state_id), int(task_index))
    observation.validate()
    valid = (
        np.asarray(observation.point_valid, bool)
        if observation.point_valid is not None
        else np.ones(len(observation.xyz), bool)
    )
    if observation.source_view is None:
        raise RuntimeError("正式离线回放要求 source_view")
    scene = FusedScene(
        np.asarray(observation.xyz[valid], np.float32),
        np.asarray(observation.rgb[valid], np.float32),
        np.full(int(valid.sum()), -1, np.int64),
        np.asarray(observation.source_view[valid], np.int16),
        {},
        tuple(_camera_transform(camera) for camera in observation.camera_parameters),
    )
    target_cloud = observation.xyz[np.asarray(observation.target_mask, bool) & valid]
    if not len(target_cloud):
        raise RuntimeError("数据集目标在当前观测中不可见")
    click = target_cloud[np.linalg.norm(target_cloud - target_cloud.mean(0), axis=1).argmin()]
    category = int(observation.object_category_id[observation.target_object])
    region = int(observation.task_region_id)
    return (
        scene,
        click,
        category,
        region,
        {
            "input_points": int(valid.sum()),
            "observation_load_seconds": time.perf_counter() - started,
        },
    )
