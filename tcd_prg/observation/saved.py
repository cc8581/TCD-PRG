"""Read state-0 RGB-D and deterministically reconstruct three PRO S point clouds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import ObservationProvider, ObservationRequest, PointObservation


def _camera_basis(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.sqrt(np.sum(forward * forward))
    right = np.cross(forward, up)
    right /= np.sqrt(np.sum(right * right))
    camera_up = np.cross(right, forward)
    camera_up /= np.sqrt(np.sum(camera_up * camera_up))
    return right, camera_up, forward


def reconstruct_pinhole_world(
    depth: np.ndarray,
    rgb: np.ndarray,
    instance: np.ndarray,
    camera: dict,
    view_index: int,
) -> PointObservation:
    """Project metric axial depth to world coordinates; image v grows downward."""

    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > camera["z_near"]) & (depth < camera["z_far"]) & (instance >= 0)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - float(camera["cx"])) * z / float(camera["fx"])
    y = (v.astype(np.float32) - float(camera["cy"])) * z / float(camera["fy"])
    eye = np.asarray(camera["eye"], dtype=np.float32)
    right, camera_up, forward = _camera_basis(
        eye, np.asarray(camera["target"], np.float32), np.asarray(camera["up"], np.float32)
    )
    xyz = eye + x[:, None] * right - y[:, None] * camera_up + z[:, None] * forward
    return PointObservation(
        xyz=xyz.astype(np.float32),
        rgb=(rgb[v, u].astype(np.float32) / 255.0),
        instance_id=instance[v, u].astype(np.int64),
        source_view=np.full(len(v), view_index, dtype=np.int16),
    )


def deterministic_stratified_sample(obs: PointObservation, count: int, seed: int) -> PointObservation:
    """Instance-preserving sampling with deterministic fill from the full union."""

    n = len(obs.xyz)
    if n == 0:
        raise ValueError("Observation contains no object points")
    rng = np.random.default_rng(seed)
    if n <= count:
        index = np.resize(np.arange(n, dtype=np.int64), count)
        rng.shuffle(index)
    else:
        groups = np.unique(obs.instance_id)
        per_group = max(1, count // max(1, len(groups)) // 2)
        chosen: list[np.ndarray] = []
        used = np.zeros(n, dtype=bool)
        for group in groups:
            candidates = np.flatnonzero(obs.instance_id == group)
            take = min(per_group, len(candidates))
            selected = rng.choice(candidates, take, replace=False)
            chosen.append(selected)
            used[selected] = True
        base = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        remaining = count - len(base)
        pool = np.flatnonzero(~used)
        fill = rng.choice(pool, remaining, replace=False) if remaining else np.empty(0, np.int64)
        index = np.concatenate((base, fill))
        rng.shuffle(index)
    return PointObservation(obs.xyz[index], obs.rgb[index], obs.instance_id[index], obs.source_view[index])


class SavedObservationProvider(ObservationProvider):
    """State-0 provider. Intermediate states must use a rendered provider."""

    def __init__(self, scene_root: str | Path, metadata_file: str | Path, point_count: int = 16_384):
        self.scene_root = Path(scene_root)
        self.metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
        self.point_count = point_count
        self.cameras = [c for c in self.metadata["camera_parameters"] if c["sensor_type"] != "oracle"]
        if len(self.cameras) != 3:
            raise ValueError("Formal input requires exactly three non-Oracle PRO S cameras")

    def get(self, request: ObservationRequest) -> PointObservation:
        if request.state_id != 0:
            raise KeyError("SavedObservationProvider only has the initial state")
        path = self.scene_root / f"scene_{request.scene_id:04d}" / "scene.npz"
        with np.load(path, allow_pickle=False) as data:
            camera_types = data["view_camera_type"].astype(str)
            if any(x.lower() == "oracle" for x in camera_types[:3]) or camera_types[3].lower() != "oracle":
                raise ValueError("Unexpected camera order; refusing possible Oracle leakage")
            views = [
                reconstruct_pinhole_world(
                    data["view_depth"][i], data["view_rgb"][i], data["view_instance"][i], camera, i
                )
                for i, camera in enumerate(self.cameras)
            ]
        union = PointObservation(
            np.concatenate([x.xyz for x in views]),
            np.concatenate([x.rgb for x in views]),
            np.concatenate([x.instance_id for x in views]),
            np.concatenate([x.source_view for x in views]),
        )
        return deterministic_stratified_sample(union, request.point_count or self.point_count, request.render_seed)
