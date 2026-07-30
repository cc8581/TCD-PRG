"""PyBullet loading and gripper-command helpers."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pybullet as p


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "urdf" / "fr5_ag160_95.urdf"
OPEN_RAD = 0.0
CLOSED_RAD = 0.93
COUPLED_GRIPPER_JOINTS = (
    "left_outer_knuckle_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
AG_LINKS = (
    "ag95_base_link",
    "ag95_body",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_finger",
    "left_finger_pad",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_finger",
    "right_finger_pad",
)


def joint_indices(body: int) -> dict[str, int]:
    return {
        p.getJointInfo(body, index)[1].decode("utf-8"): index
        for index in range(p.getNumJoints(body))
    }


def link_indices(body: int) -> dict[str, int]:
    return {
        p.getJointInfo(body, index)[12].decode("utf-8"): index
        for index in range(p.getNumJoints(body))
    }


def disable_ag_internal_collisions(body: int) -> None:
    """Disable only gripper-internal loop contacts; object contacts remain active."""
    links = link_indices(body)
    for name_a, name_b in combinations(AG_LINKS, 2):
        p.setCollisionFilterPair(body, body, links[name_a], links[name_b], 0)


def load_robot(
    base_position=(0.0, 0.0, 0.0),
    base_orientation=(0.0, 0.0, 0.0, 1.0),
    use_fixed_base: bool = True,
) -> int:
    body = p.loadURDF(
        str(URDF),
        basePosition=base_position,
        baseOrientation=base_orientation,
        useFixedBase=use_fixed_base,
        flags=(
            p.URDF_USE_INERTIA_FROM_FILE
            | p.URDF_USE_SELF_COLLISION
            | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        ),
    )
    disable_ag_internal_collisions(body)
    return body


def set_gripper(
    body: int,
    closure: float,
    use_motor: bool = True,
    max_effort: float = 50.0,
) -> float:
    """Command normalized closure: 0.0 is fully open, 1.0 fully closed."""
    closure = max(0.0, min(1.0, float(closure)))
    target = OPEN_RAD + closure * (CLOSED_RAD - OPEN_RAD)
    joints = joint_indices(body)
    for name in COUPLED_GRIPPER_JOINTS:
        index = joints[name]
        if use_motor:
            p.setJointMotorControl2(
                body,
                index,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=max_effort,
                maxVelocity=2.1,
            )
        else:
            p.resetJointState(body, index, target)
    return target
