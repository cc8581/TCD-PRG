from pathlib import Path

import numpy as np
import pytest
import yaml

from real_experiment_app.config import AppConfig
from real_experiment_app.perception import (
    fuse_frames,
    remove_calibrated_table,
    sample_scene_points,
)
from real_experiment_app.transforms import model_pose_to_robot_pose, xyz_rpy_to_matrix
from real_experiment_app.types import RGBDFrame


def synthetic_frame(camera_id: str, view_index: int = 0) -> RGBDFrame:
    height, width = 80, 120
    yy, xx = np.mgrid[:height, :width]
    color = np.zeros((height, width, 3), np.uint8)
    # The tabletop is at base z=0; objects are above it.
    depth = np.full((height, width), 550.0, np.float32)
    color[:] = (90, 90, 90)
    objects = (
        ((xx - 45) ** 2 + (yy - 44) ** 2 < 13**2, 690.0, (50, 160, 240)),
        ((xx - 78) ** 2 + (yy - 40) ** 2 < 11**2, 745.0, (235, 120, 70)),
        ((xx - 62) ** 2 + (yy - 22) ** 2 < 8**2, 650.0, (90, 220, 120)),
    )
    for mask, depth_mm, rgb in objects:
        depth[mask], color[mask] = depth_mm, rgb
    transform = np.eye(4)
    transform[:3, 3] = (0.5, 0.01 * view_index, -0.55)
    return RGBDFrame(
        camera_id,
        color,
        depth,
        {"fx": 115.0, "fy": 115.0, "cx": width / 2, "cy": height / 2},
        transform,
    )


def fusion_settings(**changes):
    values = {
        "depth_min_mm": 150,
        "depth_max_mm": 1500,
        "voxel_size_m": 0.005,
        "workspace_min_m": [0.1, -0.5, -0.05],
        "workspace_max_m": [0.95, 0.5, 0.65],
        "table_plane_base": {"normal": [0, 0, 1], "offset_m": 0.0},
        "table_clearance_m": 0.003,
        "target_scene_points": 0,
        "point_sample_seed": 17,
    }
    values.update(changes)
    return values


def test_tcp_identity_and_translation():
    pose = [0.4, 0.1, 0.3, 0, 0, 0, 1]
    assert np.allclose(model_pose_to_robot_pose(pose, np.eye(4)), [400, 100, 300, 0, 0, 0])
    compensation = xyz_rpy_to_matrix([10, -20, 30, 0, 0, 0], 0.001)
    assert np.allclose(model_pose_to_robot_pose(pose, compensation)[:3], [410, 80, 330])


def test_fusion_removes_table_before_integrated_instance_perception():
    scene = fuse_frames(
        [synthetic_frame("a", 0), synthetic_frame("b", 1)],
        None,
        fusion_settings(),
    )
    assert len(scene.xyz_m) > 0
    assert float(scene.xyz_m[:, 2].min()) > 0.003
    assert np.all(scene.instance_id == -1)
    assert scene.instance_ids == []
    assert set(scene.source_view.tolist()) == {0, 1}
    assert len(scene.camera_to_world) == 2
    assert np.allclose(
        scene.camera_to_world[1], synthetic_frame("b", 1).camera_to_base
    )


def test_capture_fails_closed_without_table_calibration():
    with pytest.raises(RuntimeError, match="not calibrated"):
        fuse_frames([synthetic_frame("a")], None, fusion_settings(table_plane_base=None))


def test_table_plane_removes_surface_and_below_points():
    xyz = np.asarray([[0, 0, -0.01], [0, 0, 0], [0, 0, 0.002], [0, 0, 0.02]], np.float32)
    rgb = np.zeros_like(xyz)
    source = np.arange(4, dtype=np.int16)
    filtered = remove_calibrated_table(xyz, rgb, source, fusion_settings())
    assert np.allclose(filtered[0], [[0, 0, 0.02]])
    assert filtered[2].tolist() == [3]


def test_deployment_sampling_matches_training_reference_exactly():
    xyz = np.arange(300, dtype=np.float32).reshape(100, 3)
    rgb = np.zeros_like(xyz)
    source = np.arange(100, dtype=np.int16)
    first = sample_scene_points(xyz, rgb, source, 37, seed=9)
    second = sample_scene_points(xyz, rgb, source, 37, seed=9)
    assert first[0].shape == (37, 3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    with pytest.raises(RuntimeError, match="cannot downsample"):
        sample_scene_points(xyz[:10], rgb[:10], source[:10], 11)


def test_operator_can_record_and_persist_table_calibration(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("robot:\n  ip: 192.168.58.2\nfusion: {}\n", encoding="utf-8")
    config = AppConfig.load(path)
    config.record_table_plane([0, 0, 2], [0.1, -0.2, 0.42])
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["fusion"]["table_plane_base"] == {
        "normal": [0.0, 0.0, 1.0],
        "offset_m": -0.42,
    }
