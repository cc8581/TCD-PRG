import sys
from types import SimpleNamespace

import numpy as np
import pytest

from real_experiment_app.robot import FR5Robot
from tcd_prg.constants import ActionType


def robot():
    value = FR5Robot.__new__(FR5Robot)
    value.settings = {
        "motion_workspace_min_m": [0.1, -0.5, 0.0],
        "motion_workspace_max_m": [0.9, 0.5, 0.7],
        "gripper_max_width_m": 0.095,
        "pregrasp_distance_m": 0.1,
        "lift_distance_m": 0.1,
        "push_retreat_m": 0.05,
        "push_distance_m": 0.15,
        "removal_pose_mm_rpy_deg": [400, 300, 300, 0, 0, 0],
    }
    value.tcp_transform = np.eye(4)
    return value


def test_push_checks_entire_commanded_segment():
    value = robot()
    value.settings["push_distance_m"] = 0.01  # obsolete config must not shorten the fixed stroke
    action = {"action_type": int(ActionType.PUSH), "push_contact_world": [0.8, 0, .2],
              "push_direction_world": [1, 0, 0]}
    with pytest.raises(RuntimeError, match="实际机器人 TCP"):
        value._validate_action(action)


def test_grasp_checks_width_quaternion_and_lift():
    action = {"action_type": int(ActionType.TASK_GRASP),
              "grasp_pose_world": [0.5, 0, .65, 0, 0, 0, 1], "grasp_width_m": .04}
    with pytest.raises(RuntimeError, match="实际机器人 TCP"):
        robot()._validate_action(action)
    action["grasp_pose_world"] = [0.5, 0, .4, 0, 0, 0, 2]
    with pytest.raises(ValueError, match="四元数"):
        robot()._validate_action(action)


def test_workspace_gate_uses_compensated_robot_tcp():
    value = robot()
    value.tcp_transform[:3, 3] = [0.2, 0.0, 0.0]
    action = {
        "action_type": int(ActionType.TASK_GRASP),
        "grasp_pose_world": [0.8, 0, .4, 0, 0, 0, 1],
        "grasp_width_m": .04,
    }
    with pytest.raises(RuntimeError, match="实际机器人 TCP"):
        value._validate_action(action)


def test_robot_pose_move_applies_workspace_and_ik_gate():
    value = robot()
    calls = []
    value.controller = SimpleNamespace(
        check_pose_reachable=lambda pose: calls.append(("ik", pose)) or True,
        move_l=lambda pose, vel: calls.append(("move", pose, vel)) or 0,
    )
    value.settings["speed_percent"] = 10
    value._move_robot_pose([400, 300, 300, 0, 0, 0])
    assert [item[0] for item in calls] == ["ik", "move"]
    with pytest.raises(RuntimeError, match="实际机器人 TCP"):
        value._move_robot_pose([1000, 300, 300, 0, 0, 0])


def test_gripper_error_is_not_ignored():
    with pytest.raises(RuntimeError, match="open failed"):
        FR5Robot._require_gripper_success(7, "open")


def test_stop_uses_independent_rpc_and_disables(monkeypatch):
    calls = []
    class RPC:
        def StopMotion(self):
            calls.append("stop")
            return 0

        def ProgramStop(self):
            calls.append("program")
            return 0

        def RobotEnable(self, state):
            calls.append(("enable", state))
            return 0
    monkeypatch.setitem(sys.modules, "fairino", SimpleNamespace(
        Robot=SimpleNamespace(RPC=lambda ip: RPC())
    ))
    value = robot()
    value.settings["ip"] = "192.168.58.2"
    value.controller = SimpleNamespace(
        _stop_requested=SimpleNamespace(set=lambda: calls.append("event")))
    value.stop()
    assert calls == ["event", "stop", "program", ("enable", 0)]
