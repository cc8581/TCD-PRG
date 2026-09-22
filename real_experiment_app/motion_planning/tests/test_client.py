import os
import time

import numpy as np

from real_experiment_app.motion_planning.client import (
    model_tcp_to_moveit_pose,
    voxel_centers,
    windows_to_wsl,
    WSLMoveItPlanner,
)


def test_tcp_transform_is_composed_on_the_right():
    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.0, 0.05]
    actual = model_tcp_to_moveit_pose(pose, offset)
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.35, 0, 0, 0, 1], atol=1e-8)


def test_voxelization_keeps_target_separate_and_solidifies_instances():
    xyz = np.array([[0.001, 0.001, 0.001], [0.009, 0.009, 0.009], [0.021, 0, 0]])
    instance = np.array([3, 7, 7])
    environment, target = voxel_centers(xyz, instance, 3, 0.02, 0.01)
    assert len(environment) > 0
    assert len(target) > 0
    assert np.any(np.all(np.isclose(target, [0.005, 0.005, 0.005]), axis=1))
    assert np.any(np.all(np.isclose(environment, [0.03, 0.01, 0.01]), axis=1))


def test_windows_path_conversion():
    assert windows_to_wsl(r"D:\Codex运行垃圾\request.npz") == "/mnt/d/Codex运行垃圾/request.npz"


def test_planner_removes_only_stale_ipc_files(tmp_path):
    stale = tmp_path / "request-stale.npz"
    current = tmp_path / "result-current.json"
    stale.write_bytes(b"old")
    current.write_text("{}", encoding="utf-8")
    old = time.time() - 300
    os.utime(stale, (old, old))

    WSLMoveItPlanner(scratch_root=tmp_path, timeout_s=60)

    assert not stale.exists()
    assert current.exists()
