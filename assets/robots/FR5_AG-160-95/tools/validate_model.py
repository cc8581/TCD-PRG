"""Validate and render the FR5 + AG-160-95 URDF in PyBullet."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "urdf" / "fr5_ag160_95.urdf"
VALIDATION_DIR = ROOT / "validation"
OPEN_RAD = 0.0
CLOSED_RAD = 0.93
ARM_POSE = {
    "j1": 0.0,
    "j2": -0.85,
    "j3": 1.45,
    "j4": -2.10,
    "j5": -1.57,
    "j6": 0.0,
}
COUPLED_GRIPPER_JOINTS = (
    "left_outer_knuckle_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
AG_COLLISION_LINKS = (
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


def _decode(value: bytes) -> str:
    return value.decode("utf-8")


def _joint_map(body: int) -> dict[str, int]:
    return {
        _decode(p.getJointInfo(body, index)[1]): index
        for index in range(p.getNumJoints(body))
    }


def _link_map(body: int) -> dict[str, int]:
    return {
        _decode(p.getJointInfo(body, index)[12]): index
        for index in range(p.getNumJoints(body))
    }


def disable_ag_internal_collisions(body: int, links: dict[str, int]) -> None:
    """Disable false contacts inside the URDF-open-loop gripper linkage."""
    for name_a, name_b in combinations(AG_COLLISION_LINKS, 2):
        p.setCollisionFilterPair(body, body, links[name_a], links[name_b], 0)


def _set_configuration(
    body: int, joints: dict[str, int], closed: bool, display_pose: bool = True
) -> None:
    for name, value in ARM_POSE.items():
        p.resetJointState(body, joints[name], value if display_pose else 0.0)
    gripper_value = CLOSED_RAD if closed else OPEN_RAD
    # PyBullet does not enforce URDF <mimic>; all six linkage joints must be set.
    for name in COUPLED_GRIPPER_JOINTS:
        p.resetJointState(body, joints[name], gripper_value)


def _render(
    body: int,
    links: dict[str, int],
    label: str,
    output_path: Path,
    closeup: bool = False,
) -> None:
    if closeup:
        position, orientation = p.getLinkState(body, links["ag95_base_link"], 1)[4:6]
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        target = np.asarray(position) + rotation @ np.asarray([0.0, 0.0, 0.115])
        eye = target + rotation @ np.asarray([0.0, -0.38, 0.0])
        up = rotation @ np.asarray([0.0, 0.0, 1.0])
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    else:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.48],
            distance=1.55,
            yaw=38,
            pitch=-24,
            roll=0,
            upAxisIndex=2,
        )
    projection = p.computeProjectionMatrixFOV(
        fov=46.0, aspect=4.0 / 3.0, nearVal=0.05, farVal=4.0
    )
    width, height, rgba, _depth, _segmentation = p.getCameraImage(
        width=960,
        height=720,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=[-1.0, -1.0, 2.0],
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4))
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 16, 360, 55), fill=(245, 245, 245))
    draw.text((30, 26), label, fill=(20, 20, 20))
    image.save(output_path)


def _whole_robot_aabb(body: int) -> list[list[float]]:
    bounds = [p.getAABB(body, link) for link in range(-1, p.getNumJoints(body))]
    lower = np.min([bound[0] for bound in bounds], axis=0)
    upper = np.max([bound[1] for bound in bounds], axis=0)
    return [lower.tolist(), upper.tolist()]


def _position_in_ag_frame(body: int, links: dict[str, int], name: str) -> list[float]:
    base_position, base_orientation = p.getLinkState(
        body, links["ag95_base_link"], 1
    )[4:6]
    position = p.getLinkState(body, links[name], 1)[4]
    rotation = np.asarray(p.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)
    relative = rotation.T @ (np.asarray(position) - np.asarray(base_position))
    return relative.tolist()


def _ik_smoke_test(body: int, joints: dict[str, int], links: dict[str, int]) -> dict:
    _set_configuration(body, joints, closed=False, display_pose=False)
    target = np.asarray([0.45, 0.0, 0.35])
    solution = p.calculateInverseKinematics(
        body,
        links["tcp_link"],
        target.tolist(),
        maxNumIterations=300,
        residualThreshold=1e-6,
    )
    movable = [
        index
        for index in range(p.getNumJoints(body))
        if p.getJointInfo(body, index)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
    ]
    for joint_index, value in zip(movable, solution):
        p.resetJointState(body, joint_index, value)
    actual = np.asarray(p.getLinkState(body, links["tcp_link"], 1)[4])
    error = float(np.linalg.norm(actual - target))
    if error > 1e-4:
        raise AssertionError(f"IK smoke test failed: {error:.9f} m")
    return {
        "target_xyz_m": target.tolist(),
        "actual_xyz_m": actual.tolist(),
        "position_error_m": error,
    }


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        body = p.loadURDF(
            str(URDF),
            useFixedBase=True,
            flags=(
                p.URDF_USE_INERTIA_FROM_FILE
                | p.URDF_USE_SELF_COLLISION
                | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            ),
        )
        if body < 0:
            raise RuntimeError("PyBullet failed to load the URDF")

        joints = _joint_map(body)
        links = _link_map(body)
        expected_joints = {
            "j1",
            "j2",
            "j3",
            "j4",
            "j5",
            "j6",
            "ag_mount_joint",
            *COUPLED_GRIPPER_JOINTS,
            "tcp_link_joint",
        }
        missing = expected_joints.difference(joints)
        if missing:
            raise AssertionError(f"Missing joints: {sorted(missing)}")
        if p.getNumJoints(body) != 21:
            raise AssertionError(f"Expected 21 joints, got {p.getNumJoints(body)}")

        controlled_info = p.getJointInfo(body, joints["left_outer_knuckle_joint"])
        if abs(controlled_info[8] - OPEN_RAD) > 1e-9 or abs(
            controlled_info[9] - CLOSED_RAD
        ) > 1e-9:
            raise AssertionError("AG controlled joint limits are incorrect")

        disable_ag_internal_collisions(body, links)
        report = {
            "urdf": str(URDF),
            "pybullet_body_id": body,
            "joint_count": p.getNumJoints(body),
            "joint_names": list(joints),
            "controlled_gripper_joint": "left_outer_knuckle_joint",
            "coupled_gripper_joints": list(COUPLED_GRIPPER_JOINTS),
            "command_range_rad": [OPEN_RAD, CLOSED_RAD],
            "configurations": {},
        }

        for closed, state_name in ((False, "open"), (True, "closed")):
            _set_configuration(body, joints, closed)
            p.performCollisionDetection()
            self_contacts = [
                {
                    "link_a": contact[3],
                    "link_b": contact[4],
                    "distance_m": contact[8],
                }
                for contact in p.getContactPoints(body, body)
            ]
            if self_contacts:
                raise AssertionError(
                    f"Unexpected self contacts after AG filtering: {self_contacts}"
                )
            left_pad = _position_in_ag_frame(body, links, "left_finger_pad")
            right_pad = _position_in_ag_frame(body, links, "right_finger_pad")
            report["configurations"][state_name] = {
                "command_rad": CLOSED_RAD if closed else OPEN_RAD,
                "left_pad_center_in_ag_frame_m": left_pad,
                "right_pad_center_in_ag_frame_m": right_pad,
                "pad_center_separation_m": abs(left_pad[0] - right_pad[0]),
                "self_contacts_after_filter": self_contacts,
                "robot_aabb_m": _whole_robot_aabb(body),
            }
            _render(
                body,
                links,
                f"FR5 + AG-160-95 | {state_name.upper()}",
                VALIDATION_DIR / f"fr5_ag160_95_{state_name}.png",
            )
            _render(
                body,
                links,
                f"AG-160-95 PARALLEL LINKAGE | {state_name.upper()}",
                VALIDATION_DIR / f"ag160_95_{state_name}_closeup.png",
                closeup=True,
            )

        report["ik_smoke_test"] = _ik_smoke_test(body, joints, links)
        report_path = VALIDATION_DIR / "validation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"PASS: {URDF}")
        print(f"Joints: {p.getNumJoints(body)}")
        print(f"AG command range: {OPEN_RAD:.2f} .. {CLOSED_RAD:.2f} rad")
        print(f"IK error: {report['ik_smoke_test']['position_error_m']:.9g} m")
        print(f"Report: {report_path}")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
