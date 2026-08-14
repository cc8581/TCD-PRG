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
    # Sensor validity is geometry-only; instance==-1 is a valid background/table point.
    valid = np.isfinite(depth) & (depth > camera["z_near"]) & (depth < camera["z_far"])
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


def deterministic_stratified_sample(
    obs: PointObservation, count: int, seed: int
) -> PointObservation:
    """Deterministic sensor-only sampling; never stratify by GT instance id."""

    n = len(obs.xyz)
    if n == 0:
        raise ValueError("Observation contains no valid sensor points")
    if count <= 0 or n <= count:
        return obs
    rng = np.random.default_rng(seed)
    index = rng.choice(n, count, replace=False)
    rng.shuffle(index)
    return PointObservation(
        obs.xyz[index], obs.rgb[index], obs.instance_id[index], obs.source_view[index]
    )


def _resize_view_nearest(
    depth: np.ndarray, rgb: np.ndarray, instance: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match cached state-0 views to the formal PyBullet render resolution."""

    source_height, source_width = depth.shape
    if (source_width, source_height) == (width, height):
        return depth, rgb, instance
    y = np.minimum(
        ((np.arange(height) + 0.5) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    x = np.minimum(
        ((np.arange(width) + 0.5) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    index = np.ix_(y, x)
    return depth[index], rgb[index], instance[index]


class SavedObservationProvider(ObservationProvider):
    """State-0 provider. Intermediate states must use a rendered provider."""

    def __init__(
        self,
        scene_root: str | Path,
        metadata_file: str | Path,
        point_count: int = 0,
        width: int | None = None,
        height: int | None = None,
    ):
        self.scene_root = Path(scene_root)
        self.metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
        self.point_count = point_count
        source_width, source_height = self.metadata["image_size"]
        self.width = int(width or source_width)
        self.height = int(height or source_height)
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
            views = []
            for i, camera in enumerate(self.cameras):
                depth = data["view_depth"][i]
                rgb = data["view_rgb"][i]
                instance = data["view_instance"][i]
                source_height, source_width = depth.shape
                depth, rgb, instance = _resize_view_nearest(
                    depth, rgb, instance, self.width, self.height
                )
                scaled_camera = dict(camera)
                scaled_camera.update(
                    fx=float(camera["fx"]) * self.width / source_width,
                    fy=float(camera["fy"]) * self.height / source_height,
                    cx=(float(camera["cx"]) + 0.5) * self.width / source_width - 0.5,
                    cy=(float(camera["cy"]) + 0.5) * self.height / source_height - 0.5,
                )
                views.append(
                    reconstruct_pinhole_world(depth, rgb, instance, scaled_camera, i)
                )
        union = PointObservation(
            np.concatenate([x.xyz for x in views]),
            np.concatenate([x.rgb for x in views]),
            np.concatenate([x.instance_id for x in views]),
            np.concatenate([x.source_view for x in views]),
        )
        count = request.point_count if request.point_count != 0 else self.point_count
        return deterministic_stratified_sample(union, count, request.render_seed)
