"""Non-learning FR5/AG-160-95 certification and execution boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from tcd_prg.constants import ActionType, PUSH_DISTANCE_M


class MotionPlanner(Protocol):
    def collision_free(self, action: Any) -> tuple[bool, str]: ...

    def ik_reachable(self, action: Any) -> tuple[bool, str]: ...


class RobotClient(Protocol):
    def execute_task_grasp(self, action: Any) -> bool: ...

    def execute_pick_remove(self, action: Any) -> bool: ...

    def execute_push(self, action: Any) -> bool: ...


class FR5AG16095Executor:
    def __init__(self, urdf: str | Path, planner: MotionPlanner, robot: RobotClient) -> None:
        self.urdf = Path(urdf)
        if not self.urdf.exists():
            raise FileNotFoundError(self.urdf)
        self.planner, self.robot = planner, robot

    def certify(self, action: Any) -> tuple[bool, str]:
        kind = int(action["action_type"] if isinstance(action, dict) else action.action_type)
        if kind == int(ActionType.PUSH):
            distance = float(action["push_distance_m"] if isinstance(action, dict) else action.push_distance_m)
            if abs(distance - PUSH_DISTANCE_M) > 1e-6:
                return False, "push_distance_not_0.15m"
        ik, reason = self.planner.ik_reachable(action)
        if not ik:
            return False, f"ik:{reason}"
        collision_free, reason = self.planner.collision_free(action)
        if not collision_free:
            return False, f"collision_or_path:{reason}"
        return True, "ok"

    def execute(self, action: Any) -> bool:
        kind = int(action["action_type"] if isinstance(action, dict) else action.action_type)
        if kind == int(ActionType.TASK_GRASP):
            return self.robot.execute_task_grasp(action)
        if kind == int(ActionType.PICK_REMOVE):
            return self.robot.execute_pick_remove(action)
        if kind == int(ActionType.PUSH):
            return self.robot.execute_push(action)
        raise ValueError(f"Unknown action type {kind}")

