"""Windows-side reusable client for the ROS 2 planner running inside WSL2."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..transforms import matrix_to_pose7, pose7_to_matrix


class MoveItPlanningError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MoveItPlanningResult:
    success: bool
    plan_id: str
    failed_stage: str
    message: str
    transit_points: int
    approach_points: int
    achieved_pose_xyzw: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    executed: bool
    target_attached: bool
    obstacle_voxels: int

    @classmethod
    def from_json(cls, payload: dict) -> "MoveItPlanningResult":
        return cls(
            success=bool(payload["success"]),
            plan_id=str(payload.get("plan_id", "")),
            failed_stage=str(payload.get("failed_stage", "")),
            message=str(payload.get("message", "")),
            transit_points=int(payload.get("transit_points", 0)),
            approach_points=int(payload.get("approach_points", 0)),
            achieved_pose_xyzw=tuple(map(float, payload.get("achieved_pose_xyzw", ()))),
            position_error_m=float(payload.get("position_error_m", float("inf"))),
            orientation_error_rad=float(payload.get("orientation_error_rad", float("inf"))),
            executed=bool(payload.get("executed", False)),
            target_attached=bool(payload.get("target_attached", False)),
            obstacle_voxels=int(payload.get("obstacle_voxels", 0)),
        )


def model_tcp_to_moveit_pose(model_pose, model_tcp_to_robot_tcp) -> np.ndarray:
    """Convert TCD-PRG model TCP [xyz,xyzw] to MoveIt's physical tcp_link pose."""
    transform = pose7_to_matrix(model_pose) @ np.asarray(model_tcp_to_robot_tcp, np.float64)
    return matrix_to_pose7(transform)


def voxel_centers(
    xyz_m: np.ndarray,
    instance_id: np.ndarray,
    target_instance: int,
    voxel_size_m: float,
    target_voxel_size_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-instance solid voxels and return environment then target."""
    xyz = np.asarray(xyz_m, np.float64)
    instance = np.asarray(instance_id, np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or instance.shape != (len(xyz),):
        raise ValueError("scene XYZ and instance_id must have shapes [N,3] and [N]")
    target_size = voxel_size_m if target_voxel_size_m is None else target_voxel_size_m
    if (not np.isfinite(xyz).all() or not 0.005 <= voxel_size_m <= 0.10
            or not 0.005 <= target_size <= 0.10):
        raise ValueError("scene points must be finite and voxel size must lie in [0.005,0.10]")
    def solid(points_xyz: np.ndarray, ids: np.ndarray, size: float) -> np.ndarray:
        blocks = []
        for object_id in np.unique(ids):
            points = points_xyz[ids == object_id]
            if not len(points):
                continue
            if int(object_id) < 0:
                keys = np.floor(points / size).astype(np.int64)
                blocks.append(np.unique(keys, axis=0))
                continue
            keys = np.floor(points / size).astype(np.int64)
            columns = []
            for column in np.unique(keys[:, :2], axis=0):
                observed = keys[np.all(keys[:, :2] == column, axis=1), 2]
                z = np.arange(int(observed.min()), int(observed.max()) + 1, dtype=np.int64)
                columns.append(np.column_stack((
                    np.full(len(z), column[0]), np.full(len(z), column[1]), z,
                )))
            if sum(len(column) for column in columns) > 50_000:
                raise ValueError(f"instance {object_id} produces too many collision voxels")
            blocks.append(np.concatenate(columns, axis=0))
        if not blocks:
            return np.empty((0, 3), np.float64)
        unique = np.unique(np.concatenate(blocks), axis=0)
        return (unique.astype(np.float64) + 0.5) * size

    target_mask = instance == int(target_instance)
    return solid(xyz[~target_mask], instance[~target_mask], voxel_size_m), solid(
        xyz[target_mask], instance[target_mask], target_size
    )


def windows_to_wsl(path: str | Path) -> str:
    value = Path(path).resolve()
    drive = value.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, received {value}")
    suffix = value.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


class WSLMoveItPlanner:
    """Call a persistent MoveIt service through a short-lived WSL ROS client."""

    def __init__(
        self,
        *,
        distro: str = "Ubuntu-24.04",
        ros_workspace: str = "/mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws",
        scratch_root: str | Path = r"D:\Codex运行垃圾\TCD-PRG\motion_planning",
        timeout_s: float = 60.0,
    ) -> None:
        self.distro = distro
        self.ros_workspace = ros_workspace.rstrip("/")
        self.scratch_root = Path(scratch_root)
        self.timeout_s = float(timeout_s)
        self._cleanup_stale_requests()

    def _cleanup_stale_requests(self) -> None:
        """Remove abandoned IPC files without touching requests still in flight."""
        if not self.scratch_root.is_dir():
            return
        cutoff = time.time() - max(self.timeout_s + 30.0, 120.0)
        for pattern in ("request-*.npz", "result-*.json"):
            for path in self.scratch_root.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue

    def plan_grasp(
        self,
        *,
        grasp_pose_model_xyzw,
        model_tcp_to_robot_tcp,
        scene_xyz_m,
        scene_instance_id,
        target_instance: int,
        pregrasp_distance_m: float = 0.10,
        obstacle_voxel_size_m: float = 0.02,
        target_voxel_size_m: float = 0.01,
        collision_padding_m: float = 0.0,
        table_z_m: float = 0.0,
        execute: bool = False,
    ) -> MoveItPlanningResult:
        pose = model_tcp_to_moveit_pose(grasp_pose_model_xyzw, model_tcp_to_robot_tcp)
        environment, target = voxel_centers(
            scene_xyz_m, scene_instance_id, target_instance, obstacle_voxel_size_m,
            target_voxel_size_m,
        )
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_path = self.scratch_root / f"request-{request_id}.npz"
        result_path = self.scratch_root / f"result-{request_id}.json"
        np.savez_compressed(
            request_path,
            operation=np.uint8(0),
            plan_id=np.asarray(""),
            grasp_pose_xyzw=pose,
            environment_centers_m=environment,
            target_centers_m=target,
            obstacle_voxel_size_m=np.float64(obstacle_voxel_size_m),
            target_voxel_size_m=np.float64(target_voxel_size_m),
            collision_padding_m=np.float64(collision_padding_m),
            pregrasp_distance_m=np.float64(pregrasp_distance_m),
            touch_links=np.asarray([
                "left_finger", "left_finger_pad", "right_finger", "right_finger_pad",
            ]),
            attach_target_after_execute=np.bool_(True),
            add_table=np.bool_(True),
            table_z_m=np.float64(table_z_m),
            table_size_m=np.asarray([1.2, 1.2, 0.04], np.float64),
        )
        wsl_request = windows_to_wsl(request_path)
        wsl_result = windows_to_wsl(result_path)
        command = (
            "source /opt/ros/jazzy/setup.bash && "
            f"source {shlex.quote(self.ros_workspace + '/install/setup.bash')} && "
            "ros2 run tcd_prg_motion_planner plan_grasp_request.py "
            f"--request {shlex.quote(wsl_request)} --result {shlex.quote(wsl_result)} "
            f"--timeout {self.timeout_s:.6g}"
        )
        try:
            result = self._run(command, result_path)
            return self.execute_plan(result.plan_id) if execute and result.success else result
        finally:
            with suppress(OSError):
                request_path.unlink()
            with suppress(OSError):
                result_path.unlink()

    def execute_plan(self, plan_id: str) -> MoveItPlanningResult:
        """Execute the exact trajectories retained by a successful planning call."""
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_path = self.scratch_root / f"request-{request_id}.npz"
        result_path = self.scratch_root / f"result-{request_id}.json"
        np.savez_compressed(request_path, operation=np.uint8(1), plan_id=np.asarray(plan_id))
        command = (
            "source /opt/ros/jazzy/setup.bash && "
            f"source {shlex.quote(self.ros_workspace + '/install/setup.bash')} && "
            "ros2 run tcd_prg_motion_planner plan_grasp_request.py "
            f"--request {shlex.quote(windows_to_wsl(request_path))} "
            f"--result {shlex.quote(windows_to_wsl(result_path))} --timeout {self.timeout_s:.6g}"
        )
        try:
            return self._run(command, result_path)
        finally:
            with suppress(OSError):
                request_path.unlink()
            with suppress(OSError):
                result_path.unlink()

    def _run(self, command: str, result_path: Path) -> MoveItPlanningResult:
        completed = subprocess.run(
            ["wsl.exe", "-d", self.distro, "bash", "-lc", command], text=True,
            encoding="utf-8", errors="replace", capture_output=True,
            timeout=self.timeout_s + 15.0, check=False,
        )
        if not result_path.is_file():
            raise MoveItPlanningError(
                f"MoveIt client failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        result = MoveItPlanningResult.from_json(json.loads(result_path.read_text(encoding="utf-8")))
        if completed.returncode not in (0, 2):
            raise MoveItPlanningError(
                f"MoveIt client exited with {completed.returncode}: {completed.stderr.strip()}"
            )
        return result
