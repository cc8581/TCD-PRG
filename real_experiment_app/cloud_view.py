"""GPU point-cloud viewport with independent scene and overlay buffers."""
from __future__ import annotations

import colorsys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMatrix4x4
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .workflow import PUSH_DISTANCE_M

GRIPPER_CLOUD = (
    Path(__file__).resolve().parents[1]
    / "assets/robots/FR5_AG-160-95/ag16095_open_tcp_4096.npz"
)
GRASP_AXIS_LENGTH_M = .02
SCENE_POINT_SIZE = 1.0
TARGET_POINT_SIZE = 5.0


def obstruction_colors(instance_ids) -> dict[int, tuple[float, float, float]]:
    """Assign stable, distinguishable colors to the inferred blockers."""
    return {
        instance: colorsys.hsv_to_rgb((0.08 + instance * 0.61803398875) % 1.0, .88, 1.0)
        for instance in sorted(set(map(int, instance_ids)))
    }


@lru_cache(maxsize=1)
def _gripper_reference() -> tuple[np.ndarray, np.ndarray, float]:
    """Load the AG-160-95 CAD cloud in the same model TCP frame as grasp poses."""
    with np.load(GRIPPER_CLOUD, allow_pickle=False) as geometry:
        return (
            np.asarray(geometry["points_tcp"], np.float32),
            np.asarray(geometry["part_id"], np.int64),
            float(geometry["width_m"]),
        )


def selected_gripper_points(action: dict | None) -> np.ndarray:
    """Place the selected, width-adjusted parallel-jaw CAD cloud in the scene."""
    if not action or int(action.get("action_type", -1)) != 2:
        return np.empty((0, 6), np.float32)
    from .transforms import pose7_to_matrix

    pose = pose7_to_matrix(action["grasp_pose_world"])
    reference, part_id, open_width = _gripper_reference()
    width = float(action["grasp_width_m"])
    if not np.isfinite(width) or not 0 <= width <= open_width + 1e-6:
        raise ValueError("selected grasp width is outside the AG-160-95 display range")
    points = reference.copy()
    # The reference CAD cloud is fully open. Move the two jaw assemblies
    # symmetrically; the palm remains fixed in the model TCP frame.
    half_closure = (open_width - width) * .5
    points[part_id == 2, 0] += half_closure
    points[part_id == 3, 0] -= half_closure
    # A 4096x3 @ 3x3 NumPy matmul initializes MKL's OpenMP DLL in the UI
    # process. Offline scene loading has already initialized PyTorch's copy.
    # These three elementwise transforms avoid loading a second runtime.
    world = np.empty_like(points)
    for axis in range(3):
        world[:, axis] = (
            pose[axis, 3]
            + points[:, 0] * pose[axis, 0]
            + points[:, 1] * pose[axis, 1]
            + points[:, 2] * pose[axis, 2]
        )
    colors = np.empty_like(world)
    colors[:] = (.92, .90, .75)
    colors[part_id != 1] = (1., .68, .22)
    return np.ascontiguousarray(np.column_stack((world, colors)), np.float32)


def append_grasp_axes(lines: list, pose) -> None:
    """Draw three 2 cm local-axis arrows at the selected grasp TCP."""
    from .transforms import pose7_to_matrix

    transform = pose7_to_matrix(pose)
    origin, rotation = transform[:3, 3], transform[:3, :3]
    for axis, color in enumerate(((1., .2, .2), (.2, 1., .4), (.2, .6, 1.))):
        direction = rotation[:, axis]
        tip = origin + direction * GRASP_AXIS_LENGTH_M
        wing_direction = rotation[:, (axis + 1) % 3]
        wing_base = tip - direction * .005
        wings = (wing_base + wing_direction * .0025,
                 wing_base - wing_direction * .0025)
        for start, end in ((origin, tip), (wings[0], tip), (wings[1], tip)):
            lines.extend(([*start, *color], [*end, *color]))


def spatial_display_indices(xyz: np.ndarray, limit: int) -> np.ndarray:
    """Deterministic spatial sampling used only by the display layer."""
    count = len(xyz)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    points = np.asarray(xyz, dtype=np.float64)
    lo = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lo, 1e-9)
    cells = max(2, int(np.ceil((limit * 1.35) ** (1.0 / 3.0))))
    cell = np.floor((points - lo) / span * (cells - 1)).astype(np.int64)
    key = cell[:, 0] + cells * (cell[:, 1] + cells * cell[:, 2])
    _, first = np.unique(key, return_index=True)
    selected = np.sort(first)
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(count, dtype=np.int64), selected, assume_unique=True)
        fill = remaining[np.linspace(0, len(remaining) - 1, limit - len(selected), dtype=np.int64)]
        selected = np.sort(np.concatenate((selected, fill)))
    elif len(selected) > limit:
        selected = selected[np.linspace(0, len(selected) - 1, limit, dtype=np.int64)]
    return selected


def append_push_arrow(lines: list, action: dict, color: tuple[float, float, float]) -> None:
    """Append a shaft and two tip wings in the horizontal PUSH direction."""
    start = np.asarray(action["push_contact_world"], dtype=float)
    direction = np.asarray(action["push_direction_world"], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        return
    direction /= norm
    distance = float(action.get("push_distance_m", PUSH_DISTANCE_M))
    if not np.isfinite(distance) or distance <= 0:
        return
    end = start + direction * distance
    head = min(.02, .2 * distance)
    perpendicular = np.array([-direction[1], direction[0], 0.])
    perpendicular /= max(float(np.linalg.norm(perpendicular)), 1e-8)
    left = end - direction * head + perpendicular * (.5 * head)
    right = end - direction * head - perpendicular * (.5 * head)
    for first, second in ((start, end), (left, end), (right, end)):
        lines.extend(([*first, *color], [*second, *color]))


class OpenGLCloudView(QOpenGLWidget):
    def __init__(self, display_max_points=300_000, parent=None):
        super().__init__(parent)
        # The confirmed UI contract requires complete display up to 300k.
        self.display_max_points = max(300_000, int(display_max_points))
        self.scene = self.target = self.action = None
        self._points = np.empty((0, 6), np.float32)
        self._gripper_points = np.empty((0, 6), np.float32)
        self._lines = np.empty((0, 6), np.float32)
        self._point_cache_key = None
        self._target_point_start = 0
        self._points_dirty = self._gripper_dirty = self._lines_dirty = True
        self._yaw, self._pitch, self._distance = -35.0, 24.0, 1.2
        self._last = QPoint()
        self._center = np.zeros(3, np.float32)

    @staticmethod
    def _candidate_color(candidate):
        state = candidate.get("_visual_state", "candidate")
        if state in ("collision_free", "simulated_effective"):
            return .20, .95, .42
        if state == "simulated_invalid":
            return .48, .48, .52
        return (.90, .55, .16) if int(candidate.get("action_type", -1)) == 0 else (.25, .70, .88)

    def _build_lines(self, candidates):
        lines = []
        for candidate in candidates:
            if int(candidate.get("action_type", -1)) == 0:
                append_push_arrow(lines, candidate, self._candidate_color(candidate))
        action = self.action
        if action and int(action.get("action_type", -1)) == 0:
            append_push_arrow(lines, action, (1., .15, .12))
        elif action and "grasp_pose_world" in action:
            append_grasp_axes(lines, action["grasp_pose_world"])
        return np.ascontiguousarray(lines, np.float32).reshape((-1, 6))

    def set_data(self, scene, target=None, action=None, candidates=(), obstructions=()):
        self.scene, self.target, self.action = scene, target, action
        acted = None if not action else action.get("acted_object")
        obstruction_ids = tuple(sorted(set(map(int, obstructions))))
        point_key = (id(scene), target, acted, obstruction_ids, self.display_max_points)
        if scene is None:
            if len(self._points):
                self._points = np.empty((0, 6), np.float32)
                self._target_point_start = 0
                self._points_dirty = True
            self._point_cache_key = point_key
        elif point_key != self._point_cache_key:
            indices = spatial_display_indices(scene.xyz_m, self.display_max_points)
            xyz = np.asarray(scene.xyz_m[indices], np.float32)
            colors = np.asarray(scene.rgb[indices], np.float32).copy()
            labels = scene.instance_id[indices]
            target_mask = labels == int(target) if target is not None else np.zeros(len(labels), bool)
            if target is not None:
                colors[target_mask] = (.10, .82, 1.)
                colors[~target_mask] *= .35
            highlighted = set(obstruction_ids)
            if acted is not None:
                highlighted.add(int(acted))
            highlighted.discard(int(target) if target is not None else -999)
            for instance, color in obstruction_colors(highlighted).items():
                colors[labels == instance] = color
            # Keep target points last so one draw call can use a larger point size.
            order = np.concatenate((np.flatnonzero(~target_mask), np.flatnonzero(target_mask)))
            self._target_point_start = int((~target_mask).sum())
            self._points = np.ascontiguousarray(np.column_stack((xyz[order], colors[order])), np.float32)
            self._center = np.median(xyz, axis=0)
            self._point_cache_key = point_key
            self._points_dirty = True
        self._lines = self._build_lines(candidates)
        self._gripper_points = selected_gripper_points(action)
        self._gripper_dirty = True
        self._lines_dirty = True
        if self.context() and self.context().isValid():
            self._upload_dirty()
        self.update()

    def initializeGL(self):
        self.program = QOpenGLShaderProgram(self)
        self.program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, "attribute highp vec3 position; attribute lowp vec3 color; uniform highp mat4 mvp; uniform highp float point_size; varying lowp vec3 vcolor; void main(){gl_Position=mvp*vec4(position,1.0);gl_PointSize=point_size;vcolor=color;}"  # noqa: E501
        )
        self.program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, "varying lowp vec3 vcolor; void main(){gl_FragColor=vec4(vcolor,1.0);}"  # noqa: E501
        )
        if not self.program.link():
            raise RuntimeError(self.program.log())
        self.point_size_location = self.program.uniformLocation("point_size")
        self.point_vao, self.point_vbo = self._new_buffer_pair()
        self.gripper_vao, self.gripper_vbo = self._new_buffer_pair()
        self.line_vao, self.line_vbo = self._new_buffer_pair()
        self.gl = self.context().functions()
        self.gl.glEnable(0x0B71)
        if not self.context().isOpenGLES():
            self.gl.glEnable(0x8642)  # GL_PROGRAM_POINT_SIZE
        self._upload_dirty()

    def _new_buffer_pair(self):
        vao = QOpenGLVertexArrayObject(self)
        vao.create()
        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        return vao, vbo

    def _upload(self, vao, vbo, vertices):
        vao.bind()
        vbo.bind()
        vbo.allocate(vertices.tobytes(), vertices.nbytes)
        self.program.bind()
        self.program.enableAttributeArray("position")
        self.program.setAttributeBuffer("position", 0x1406, 0, 3, 24)
        self.program.enableAttributeArray("color")
        self.program.setAttributeBuffer("color", 0x1406, 12, 3, 24)
        vbo.release()
        vao.release()
        self.program.release()

    def _upload_dirty(self):
        self.makeCurrent()
        if self._points_dirty:
            self._upload(self.point_vao, self.point_vbo, self._points)
            self._points_dirty = False
        if self._gripper_dirty:
            self._upload(self.gripper_vao, self.gripper_vbo, self._gripper_points)
            self._gripper_dirty = False
        if self._lines_dirty:
            self._upload(self.line_vao, self.line_vbo, self._lines)
            self._lines_dirty = False

    def paintGL(self):
        self.gl.glClearColor(.055, .09, .14, 1)
        self.gl.glClear(0x00004000 | 0x00000100)
        if not len(self._points):
            return
        projection = QMatrix4x4()
        projection.perspective(42, max(1, self.width()) / max(1, self.height()), .001, 20)
        view = QMatrix4x4()
        view.translate(0, 0, -self._distance)
        view.rotate(self._pitch, 1, 0, 0)
        view.rotate(self._yaw, 0, 0, 1)
        view.translate(float(-self._center[0]), float(-self._center[1]), float(-self._center[2]))
        self.program.bind()
        self.program.setUniformValue("mvp", projection * view)
        self.point_vao.bind()
        self.gl.glUniform1f(self.point_size_location, SCENE_POINT_SIZE)
        self.gl.glDrawArrays(0x0000, 0, self._target_point_start)
        if self._target_point_start < len(self._points):
            self.gl.glUniform1f(self.point_size_location, TARGET_POINT_SIZE)
            self.gl.glDrawArrays(0x0000, self._target_point_start, len(self._points) - self._target_point_start)
        self.point_vao.release()
        if len(self._gripper_points):
            self.gl.glUniform1f(self.point_size_location, SCENE_POINT_SIZE)
            self.gripper_vao.bind()
            self.gl.glDrawArrays(0x0000, 0, len(self._gripper_points))
            self.gripper_vao.release()
        if len(self._lines):
            self.gl.glLineWidth(4.0)
            self.line_vao.bind()
            self.gl.glDrawArrays(0x0001, 0, len(self._lines))
            self.line_vao.release()
        self.program.release()

    def mousePressEvent(self, event): self._last = event.position().toPoint()

    def mouseMoveEvent(self, event):
        delta = event.position().toPoint() - self._last
        self._last = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._yaw += delta.x() * .45
            self._pitch = float(np.clip(self._pitch + delta.y() * .45, -89, 89))
            self.update()

    def wheelEvent(self, event):
        self._distance = float(
            np.clip(self._distance * np.exp(-event.angleDelta().y() / 1200), .08, 8))
        self.update()
