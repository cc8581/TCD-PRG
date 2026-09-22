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
from .collision_geometry import build_collision_geometry, pack_meshes, voxel_centers


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
    collision_geometry: str = ""
    obstacle_meshes: int = 0
    obstacle_triangles: int = 0

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
            collision_geometry=str(payload.get("collision_geometry", "")),
            obstacle_meshes=int(payload.get("obstacle_meshes", 0)),
            obstacle_triangles=int(payload.get("obstacle_triangles", 0)),
        )


def model_tcp_to_moveit_pose(model_pose, model_tcp_to_robot_tcp) -> np.ndarray:
    """Convert TCD-PRG model TCP [xyz,xyzw] to MoveIt's physical tcp_link pose."""
    transform = pose7_to_matrix(model_pose) @ np.asarray(model_tcp_to_robot_tcp, np.float64)
    return matrix_to_pose7(transform)


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
        collision_geometry: str = "hybrid",
        mesh_max_points: int = 2500,
        alpha_radius_m: float = 0.025,
        table_z_m: float = 0.0,
        execute: bool = False,
    ) -> MoveItPlanningResult:
        pose = model_tcp_to_moveit_pose(grasp_pose_model_xyzw, model_tcp_to_robot_tcp)
        geometry = build_collision_geometry(
            scene_xyz_m,
            scene_instance_id,
            target_instance,
            mode=collision_geometry,
            voxel_size_m=obstacle_voxel_size_m,
            target_voxel_size_m=target_voxel_size_m,
            collision_padding_m=collision_padding_m,
            mesh_max_points=mesh_max_points,
            alpha_radius_m=alpha_radius_m,
        )
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_path = self.scratch_root / f"request-{request_id}.npz"
        result_path = self.scratch_root / f"result-{request_id}.json"
        payload = {
            "operation": np.uint8(0),
            "plan_id": np.asarray(""),
            "grasp_pose_xyzw": pose,
            # Keep the historic key names so old/new ROS clients remain easy to inspect.
            "environment_centers_m": geometry.environment_voxels,
            "target_centers_m": geometry.target_voxels,
            "collision_geometry": np.asarray(geometry.mode),
            "obstacle_voxel_size_m": np.float64(obstacle_voxel_size_m),
            "target_voxel_size_m": np.float64(target_voxel_size_m),
            "collision_padding_m": np.float64(collision_padding_m),
            "pregrasp_distance_m": np.float64(pregrasp_distance_m),
            "touch_links": np.asarray([
                "left_finger", "left_finger_pad", "right_finger", "right_finger_pad",
            ]),
            "attach_target_after_execute": np.bool_(True),
            "add_table": np.bool_(True),
            "table_z_m": np.float64(table_z_m),
            "table_size_m": np.asarray([1.2, 1.2, 0.04], np.float64),
        }
        payload.update(pack_meshes("environment", geometry.environment_meshes))
        payload.update(pack_meshes("target", geometry.target_meshes))
        np.savez_compressed(request_path, **payload)
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
