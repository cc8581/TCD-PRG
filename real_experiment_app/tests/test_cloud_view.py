import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication

from real_experiment_app.cloud_view import (
    GRASP_AXIS_LENGTH_M,
    OpenGLCloudView,
    _gripper_reference,
    selected_gripper_points,
    spatial_display_indices,
)
from real_experiment_app.main import MainWindow
from real_experiment_app.workflow import ObstructionRelation


def test_display_sampling_preserves_full_cloud_below_limit():
    xyz = np.arange(90, dtype=float).reshape(30, 3)
    assert np.array_equal(spatial_display_indices(xyz, 30), np.arange(30))


def test_display_sampling_is_deterministic_and_respects_limit():
    rng = np.random.default_rng(17)
    xyz = rng.uniform([-1, -2, 0], [1, 2, 3], (2000, 3))
    first = spatial_display_indices(xyz, 300)
    second = spatial_display_indices(xyz, 300)
    assert len(first) == 300
    assert len(np.unique(first)) == 300
    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < len(xyz)


def test_push_overlay_has_visible_arrow_head_for_candidate_and_selection():
    push = {
        "action_type": 0,
        "push_contact_world": np.array([0.0, 0.0, 0.1]),
        "push_direction_world": np.array([1.0, 0.0, 0.0]),
        "push_distance_m": 0.15,
    }
    view = SimpleNamespace(action=push, _candidate_color=OpenGLCloudView._candidate_color)
    lines = OpenGLCloudView._build_lines(view, [push])
    assert lines.shape == (12, 6)  # shaft + two wings for each of two arrows
    assert np.allclose(lines[1, :3], [0.15, 0.0, 0.1])
    assert np.allclose(lines[3, :3], [0.15, 0.0, 0.1])
    assert lines[2, 1] < 0 < lines[4, 1] or lines[4, 1] < 0 < lines[2, 1]
    assert np.allclose(lines[9, 3:], [1.0, 0.15, 0.12])


def test_grasp_overlay_shows_only_selected_short_xyz_arrows():
    selected = {
        "action_type": 2,
        "grasp_pose_world": [0.4, -0.2, 0.3, 0, 0, 0, 1],
        "grasp_width_m": 0.04,
    }
    other = {
        **selected,
        "grasp_pose_world": [0.8, 0.7, 0.6, 0, 0, 0, 1],
    }
    view = SimpleNamespace(action=selected, _candidate_color=OpenGLCloudView._candidate_color)
    lines = OpenGLCloudView._build_lines(view, [other, selected])
    assert lines.shape == (18, 6)  # three arrows, shaft plus two wings each
    for axis in range(3):
        start, tip = lines[axis * 6, :3], lines[axis * 6 + 1, :3]
        assert np.allclose(start, selected["grasp_pose_world"][:3])
        assert np.isclose(np.linalg.norm(tip - start), GRASP_AXIS_LENGTH_M)
        assert np.allclose(lines[axis * 6 + 3, :3], tip)
        assert np.allclose(lines[axis * 6 + 5, :3], tip)
    assert not np.any(np.isclose(lines[:, 0], 0.8))


def test_selected_gripper_cloud_tracks_pose_and_opening_width():
    reference, part_id, open_width = _gripper_reference()
    action = {
        "action_type": 2,
        "grasp_pose_world": [0.4, -0.2, 0.3, 0, 0, 0, 1],
        "grasp_width_m": open_width,
    }
    open_points = selected_gripper_points(action)
    assert open_points.shape == (len(reference), 6)
    assert np.allclose(open_points[:, :3], reference + [0.4, -0.2, 0.3])
    action["grasp_width_m"] = open_width - 0.04
    narrower = selected_gripper_points(action)
    displacement = narrower[:, 0] - open_points[:, 0]
    assert np.allclose(displacement[part_id == 1], 0)
    assert np.allclose(displacement[part_id == 2], 0.02)
    assert np.allclose(displacement[part_id == 3], -0.02)
    action["grasp_width_m"] = open_width
    action["grasp_pose_world"] = [0.4, -0.2, 0.3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    rotated = selected_gripper_points(action)
    expected = np.column_stack((-reference[:, 1], reference[:, 0], reference[:, 2]))
    assert np.allclose(rotated[:, :3], expected + [0.4, -0.2, 0.3], atol=1e-6)
    assert selected_gripper_points({"action_type": 0}).shape == (0, 6)


def test_view_keeps_scene_and_selected_gripper_in_separate_buffers():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    view = OpenGLCloudView()
    scene = SimpleNamespace(
        xyz_m=np.array([[0., 0., 0.], [.1, 0., 0.]], np.float32),
        rgb=np.ones((2, 3), np.float32),
        instance_id=np.array([1, 2]),
    )
    action = {
        "action_type": 2,
        "acted_object": 1,
        "grasp_pose_world": [0.1, 0.2, 0.3, 0, 0, 0, 1],
        "grasp_width_m": 0.04,
    }
    view.set_data(scene, target=1, action=action, candidates=[action])
    assert len(view._points) == 2
    assert len(view._gripper_points) == 4096
    assert len(view._lines) == 18
    view.set_data(scene, target=1, action=None)
    assert len(view._points) == 2
    assert len(view._gripper_points) == 0
    assert len(view._lines) == 0
    view.close()


def test_obstruction_colors_persist_across_stages_and_differ_from_target():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    scene = SimpleNamespace(
        xyz_m=np.array([[0., 0., 0.], [.1, 0., .05], [.2, 0., .1], [.3, 0., 0.]], np.float32),
        rgb=np.ones((4, 3), np.float32), instance_id=np.array([1, 2, 3, 4]),
    )
    state = SimpleNamespace(_obstruction_highlight=None)
    inferred = MainWindow._stage_obstructions(
        state, scene, 1, {"relations": (ObstructionRelation(1, 2, .5, .05),
                                           ObstructionRelation(1, 3, .5, .1))})
    assert inferred == (2, 3)
    assert MainWindow._stage_obstructions(state, scene, 1, {"stage": "PUSH_RULE_GENERATION"}) == inferred
    view = OpenGLCloudView()
    view.set_data(scene, target=1, obstructions=inferred)
    colors = view._points[:, 3:]
    assert len({tuple(color) for color in colors[:3]}) == 3
    assert np.allclose(colors[3], [.10, .82, 1.])
    assert view._target_point_start == 3
    assert MainWindow._stage_obstructions(state, scene, 4, {}) == ()
    view.close()


def test_paint_sets_point_sizes_with_opengl_float_uniform():
    class Program:
        def bind(self): pass
        def release(self): pass
        def setUniformValue(self, location, value): pass

    class GL:
        def __init__(self): self.sizes = []
        def glClearColor(self, *args): pass
        def glClear(self, *args): pass
        def glDrawArrays(self, *args): pass
        def glUniform1f(self, location, value):
            assert isinstance(location, int)
            self.sizes.append(value)

    vao = SimpleNamespace(bind=lambda: None, release=lambda: None)
    gl = GL()
    view = SimpleNamespace(
        gl=gl, program=Program(), point_vao=vao,
        point_size_location=2, _points=np.zeros((3, 6), np.float32),
        _target_point_start=2, _gripper_points=np.empty((0, 6)),
        _lines=np.empty((0, 6)), _distance=1.2, _pitch=24., _yaw=-35.,
        _center=np.zeros(3), width=lambda: 500, height=lambda: 400,
    )
    OpenGLCloudView.paintGL(view)
    assert gl.sizes == [1.0, 5.0]


def test_gripper_overlay_does_not_load_second_openmp_runtime_after_torch():
    code = (
        "import torch; "
        "from real_experiment_app.cloud_view import selected_gripper_points; "
        "selected_gripper_points({'action_type': 2, "
        "'grasp_pose_world': [.4, 0, .3, 0, 0, 0, 1], "
        "'grasp_width_m': .04}); print('overlay-ok')"
    )
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "overlay-ok" in result.stdout
