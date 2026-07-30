"""Interactive PyBullet viewer for the combined robot model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pybullet as p
import pybullet_data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pybullet_model import joint_indices, load_robot, set_gripper  # noqa: E402


def main() -> None:
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("PyBullet GUI connection failed")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = load_robot()
    joints = joint_indices(robot)
    pose = {"j1": 0.0, "j2": -0.85, "j3": 1.45, "j4": -2.10, "j5": -1.57, "j6": 0.0}
    for name, value in pose.items():
        p.resetJointState(robot, joints[name], value)
    slider = p.addUserDebugParameter("AG closure (0=open, 1=closed)", 0.0, 1.0, 0.0)
    p.resetDebugVisualizerCamera(1.35, 38, -24, [0.0, 0.0, 0.48])
    try:
        while p.isConnected():
            set_gripper(robot, p.readUserDebugParameter(slider))
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
