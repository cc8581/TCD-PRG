from __future__ import annotations

import numpy as np

from .types import FusedScene, RGBDFrame


def build_segmenter(config):
    """Deprecated compatibility hook.

    Instance perception now runs inside TCD-PRG after fused point-cloud creation.
    """
    del config
    return None


def _points(frame: RGBDFrame, depth_min: float, depth_max: float):
    depth = frame.depth_mm
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64) * 0.001
    intr = frame.intrinsics
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    camera = np.column_stack((x, y, z, np.ones_like(z)))
    base = (frame.camera_to_base @ camera.T).T[:, :3]
    rgb = frame.color_rgb[v, u].astype(np.float32) / 255.0
    return base.astype(np.float32), rgb


def remove_calibrated_table(
    xyz: np.ndarray, rgb: np.ndarray, source: np.ndarray, settings: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the calibrated tabletop and every point below it.

    The plane is expressed in the robot-base frame as ``normal @ xyz + offset = 0``.
    ``normal`` must point from the table towards the usable workspace.  A small
    positive clearance also removes depth noise immediately above the plane.
    """
    plane = settings.get("table_plane_base")
    if not isinstance(plane, dict):
        raise RuntimeError(
            "Table plane is not calibrated. Record fusion.table_plane_base before capture."
        )
    normal = np.asarray(plane.get("normal"), dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("table_plane_base.normal must contain three finite values")
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        raise ValueError("table plane normal must be non-zero")
    normal /= length
    if normal[2] <= 0:
        raise ValueError("table plane normal must point upward in the robot-base frame")
    offset = float(plane.get("offset_m")) / length
    clearance = float(settings.get("table_clearance_m", 0.003))
    if not np.isfinite(offset) or not 0.0 <= clearance <= 0.03:
        raise ValueError("table offset must be finite and clearance must be in [0,0.03] m")
    signed_distance = xyz.astype(np.float64) @ normal + offset
    keep = signed_distance > clearance
    xyz, rgb, source = xyz[keep], rgb[keep], source[keep]
    if not len(xyz):
        raise RuntimeError("Table removal deleted every point; verify table calibration")
    return xyz, rgb, source


def sample_scene_points(
    xyz: np.ndarray, rgb: np.ndarray, source: np.ndarray, target: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample to an exact deployment count without inventing duplicate points."""
    target = int(target)
    if target <= 0:
        return xyz, rgb, source
    if len(xyz) < target:
        raise RuntimeError(
            f"Only {len(xyz)} points remain after table removal/voxelization; "
            f"cannot downsample to the configured training reference {target}. "
            "Reduce fusion.voxel_size_m or target_scene_points."
        )
    if len(xyz) == target:
        return xyz, rgb, source
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(xyz), target, replace=False))
    return xyz[selected], rgb[selected], source[selected]


def fuse_frames(frames: list[RGBDFrame], segments, settings: dict) -> FusedScene:
    """Fuse raw RGB-D into XYZRGB without external instance segmentation."""
    del segments
    xyzs, rgbs, views = [], [], []
    for view, frame in enumerate(frames):
        xyz, rgb = _points(
            frame, settings["depth_min_mm"], settings["depth_max_mm"]
        )
        xyzs.append(xyz)
        rgbs.append(rgb)
        views.append(np.full(len(xyz), view, np.int16))
    if not xyzs or not sum(map(len, xyzs)):
        raise RuntimeError("No valid RGB-D points")
    xyz = np.concatenate(xyzs)
    rgb = np.concatenate(rgbs)
    source = np.concatenate(views)

    low = np.asarray(settings["workspace_min_m"])
    high = np.asarray(settings["workspace_max_m"])
    keep = np.all((xyz >= low) & (xyz <= high), axis=1)
    xyz, rgb, source = xyz[keep], rgb[keep], source[keep]
    if not len(xyz):
        raise RuntimeError("No fused points remain inside the configured workspace")

    xyz, rgb, source = remove_calibrated_table(xyz, rgb, source, settings)

    voxel = float(settings["voxel_size_m"])
    if voxel > 0:
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, selected = np.unique(keys, axis=0, return_index=True)
        selected.sort()
        xyz, rgb, source = xyz[selected], rgb[selected], source[selected]

    xyz, rgb, source = sample_scene_points(
        xyz,
        rgb,
        source,
        int(settings.get("target_scene_points", 62076)),
        int(settings.get("point_sample_seed", 20260813)),
    )

    # -1 means "not assigned yet". The integrated InstanceQueryHead fills it.
    instance = np.full(len(xyz), -1, np.int64)
    return FusedScene(xyz, rgb, instance, source, {})
