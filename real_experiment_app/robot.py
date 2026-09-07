from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import numpy as np

from .transforms import (model_pose_to_robot_pose, offset_model_pose,
                         pose7_to_matrix, push_pose)


class FR5Robot:
    def __init__(self, settings: dict, sdk_windows_root: Path, tcp_transform: np.ndarray):
        root = str(sdk_windows_root.resolve())
        if root not in sys.path: sys.path.insert(0, root)
        self.settings, self.tcp_transform = settings, tcp_transform
        self.controller = None

    def _build_controller(self):
        from grasp_system.robot.robot_controller import RobotController
        from grasp_system.config.robot_config import RobotConfig
        settings = self.settings
        RobotConfig.TOOL_ID = int(settings["tool_id"])
        RobotConfig.USER_ID = int(settings.get("user_id", 0))
        RobotConfig.VEL_DEFAULT = float(settings["speed_percent"])
        RobotConfig.GRIPPER_OPEN_POS = int(settings["gripper_open_position"])
        RobotConfig.GRIPPER_CLOSE_POS = int(settings["gripper_closed_position"])
        RobotConfig.GRIPPER_SPEED = int(settings["gripper_speed"])
        RobotConfig.GRIPPER_FORCE = int(settings["gripper_force"])
        RobotConfig.GRIPPER_OPEN_FORCE = int(settings["gripper_open_force"])
        RobotConfig.GRIPPER_MAX_TIME_MS = int(settings["gripper_max_time_ms"])
        RobotConfig.GRIPPER_INDEX = int(settings.get("gripper_index", 1))
        RobotConfig.GRIPPER_COMPANY = int(settings.get("gripper_company", 4))
        RobotConfig.GRIPPER_DEVICE = int(settings.get("gripper_device", 0))
        RobotConfig.GRIPPER_SOFTVERSION = int(settings.get("gripper_softversion", 0))
        RobotConfig.GRIPPER_BUS = int(settings.get("gripper_bus", 0))
        RobotConfig.GRIPPER_CONFIG_ON_CONNECT = bool(settings.get("gripper_config_on_connect", True))
        RobotConfig.GRIPPER_RESET_ON_CONNECT = bool(settings.get("gripper_reset_on_connect", True))
        RobotConfig.GRIPPER_ACTIVATE_ON_CONNECT = bool(settings.get("gripper_activate_on_connect", True))
        RobotConfig.GRIPPER_ACTIVATE_DELAY = float(settings.get("gripper_activate_delay_s", 3.0))
        self.controller = RobotController(str(settings["ip"]))

    def connect(self) -> bool:
        try:
            self._build_controller()
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError("未找到 Fairino Python SDK，无法连接 FR5") from error
        if not self.controller.connect(): return False
        # Connection and servo power are intentionally separate operator steps.
        return True

    def disconnect(self) -> None:
        if self.controller is not None:
            if self.controller.is_connected:
                try:
                    self.stop()
                except Exception:
                    pass
            self.controller.disconnect()
            self.controller = None
    def stop(self) -> None:
        if self.controller is None:
            return
        # Motion calls in the legacy controller hold their command lock while
        # waiting. Use a separate RPC client so a stop request cannot queue
        # behind the movement it is meant to stop.
        self.controller._stop_requested.set()
        from fairino import Robot
        safety_rpc = Robot.RPC(str(self.settings["ip"]))
        errors = []
        for name, args in (("StopMotion", ()), ("ProgramStop", ()), ("RobotEnable", (0,))):
            try:
                code = getattr(safety_rpc, name)(*args)
                if code:
                    errors.append(f"{name}={code}")
            except Exception as error:
                errors.append(f"{name}={type(error).__name__}: {error}")
        if errors:
            raise RuntimeError("FR5 紧急停止未全部成功: " + "; ".join(errors))

    def enable(self) -> None:
        self._require_controller()
        error = self.controller.set_auto_mode()
        if error == 0:
            error = self.controller.enable_robot()
        if error:
            raise RuntimeError(f"FR5 使能失败，错误码: {error}")

    def disable(self) -> None:
        self._require_controller()
        error = self.controller.disable_robot()
        if error:
            raise RuntimeError(f"FR5 下使能失败，错误码: {error}")

    def clear_errors(self) -> None:
        self._require_controller()
        error = self.controller._rpc.ResetAllError()
        if error:
            raise RuntimeError(f"FR5 清除故障失败，错误码: {error}")

    def pause(self) -> None:
        self._require_controller()
        error = self.controller._rpc.PauseMotion()
        if error:
            raise RuntimeError(f"FR5 暂停失败，错误码: {error}")

    def resume(self) -> None:
        self._require_controller()
        error = self.controller._rpc.ResumeMotion()
        if error:
            raise RuntimeError(f"FR5 继续失败，错误码: {error}")

    def home(self) -> None:
        self._require_controller()
        joints = np.asarray(self.settings["home_joints_deg"], dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("安全原点必须包含 6 个有限关节角")
        error = self.controller.move_j(
            joints.tolist(), vel=float(self.settings.get("home_speed_percent", 10))
        )
        if error:
            raise RuntimeError(f"FR5 回安全原点失败，错误码: {error}")

    def status(self) -> dict[str, Any]:
        self._require_controller()
        state = dict(self.controller._get_state_snapshot())
        state["joints_deg"] = self.controller.get_joint_pos()
        state["tcp_pose_mm_deg"] = self.controller.get_current_pose()
        state["gripper"] = self.controller.get_gripper_state()
        return state

    def _require_controller(self):
        if self.controller is None or not self.controller.is_connected:
            raise RuntimeError("FR5 尚未连接")

    def _move(self, model_pose, speed=None):
        pose = model_pose_to_robot_pose(model_pose, self.tcp_transform).tolist()
        if not self.controller.check_pose_reachable(pose):
            raise RuntimeError(f"FR5 IK has no solution: {pose}")
        error = self.controller.move_l(pose, vel=speed or self.settings["speed_percent"])
        if error: raise RuntimeError(f"FR5 MoveL failed: {error}")

    def _set_width(self, width_m: float):
        maximum = float(self.settings["gripper_max_width_m"])
        command_width = max(0., min(maximum, width_m-float(self.settings["gripper_close_margin_m"])))
        closed = float(self.settings["gripper_closed_position"])
        opened = float(self.settings["gripper_open_position"])
        position_percent = int(round(closed+(opened-closed)*command_width/maximum))
        error = self.controller._move_gripper(position_percent, int(self.settings["gripper_force"]))
        if error: raise RuntimeError(f"AG-160-95 command failed: {error}")

    def _grasp(self, action):
        pose = np.asarray(action["grasp_pose_world"], np.float64)
        pre = offset_model_pose(pose, [0,0,-float(self.settings["pregrasp_distance_m"])])
        self.controller.gripper_open(); self._move(pre); self._move(pose)
        self._set_width(float(action["grasp_width_m"]))
        lift = pose.copy(); lift[2] += float(self.settings["lift_distance_m"])
        self._move(lift)

    def execute(self, action: dict[str,Any]) -> None:
        self._validate_action(action)
        kind = int(action["action_type"])
        if kind in (1,2):
            self._grasp(action)
            if kind == 1:
                destination = list(map(float, self.settings["removal_pose_mm_rpy_deg"]))
                error = self.controller.move_l(destination, vel=self.settings["speed_percent"])
                if error: raise RuntimeError(f"PICK_REMOVE transport failed: {error}")
                self.controller.gripper_open()
            return
        if kind == 0:
            contact = np.asarray(action["push_contact_world"],np.float64)
            direction = np.asarray(action["push_direction_world"],np.float64)
            direction /= max(np.linalg.norm(direction),1e-12)
            final = push_pose(contact,direction)
            pre = push_pose(contact-direction*float(self.settings["push_retreat_m"]),direction)
            end = push_pose(contact+direction*float(self.settings["push_distance_m"]),direction)
            self.controller.gripper_close(); self._move(pre); self._move(final); self._move(end)
            self._move(pre)
            return
        raise ValueError(f"Unknown action type {kind}")

    def _validate_action(self, action: dict[str, Any]) -> None:
        kind = int(action["action_type"])
        if kind in (1, 2):
            pose = np.asarray(action.get("grasp_pose_world"), dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("抓取位姿必须是有限的 XYZ+四元数")
            position = pose[:3]
            width = float(action.get("grasp_width_m", -1))
            if not 0 <= width <= float(self.settings["gripper_max_width_m"]):
                raise ValueError("预测夹爪宽度超出物理范围")
            if not 0.95 <= float(np.linalg.norm(pose[3:])) <= 1.05:
                raise ValueError("抓取四元数未归一化")
            positions = [
                position,
                offset_model_pose(pose, [0, 0, -float(self.settings["pregrasp_distance_m"])])[:3],
                position + np.asarray([0, 0, float(self.settings["lift_distance_m"])]),
            ]
            if kind == 1:
                positions.append(np.asarray(self.settings["removal_pose_mm_rpy_deg"][:3], dtype=np.float64) * .001)
        elif kind == 0:
            position = np.asarray(action.get("push_contact_world"), dtype=np.float64)
            direction = np.asarray(action.get("push_direction_world"), dtype=np.float64)
            if position.shape != (3,) or direction.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError("推动接触点或方向无效")
            direction /= np.linalg.norm(direction)
            positions = [
                position,
                position - direction * float(self.settings["push_retreat_m"]),
                position + direction * float(self.settings["push_distance_m"]),
            ]
        else:
            raise ValueError(f"Unknown action type {kind}")
        low = np.asarray(self.settings["motion_workspace_min_m"], dtype=np.float64)
        high = np.asarray(self.settings["motion_workspace_max_m"], dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("机械臂运动安全区配置必须为 XYZ")
        for point in positions:
            if np.any(point < low) or np.any(point > high):
                raise RuntimeError(f"动作轨迹点超出机械臂运动安全区: {np.asarray(point).tolist()}")


def build_robot(config):
    settings = config.raw["robot"]
    return FR5Robot(settings, config.resolve(settings["sdk_windows_root"]), config.tcp_transform)
