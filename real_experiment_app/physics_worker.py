"""Headless PyBullet checks using point-cloud-derived compound convex bodies."""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
ASSET = PROJECT / "assets" / "robots" / "FR5_AG-160-95"
sys.path.insert(0, str(ASSET))


def _instances(path):
    with np.load(path, allow_pickle=False) as data:
        xyz, labels = data["xyz_m"], data["instance_id"]
    return {
        int(i): xyz[labels == i]
        for i in np.unique(labels)
        if int(i) >= 0 and (labels == i).sum() >= 8
    }


def _compound(p, points, parts, mass, settings, client):
    """Build a deterministic compound of point-derived convex hulls."""
    axis = int(np.argmax(np.ptp(points, axis=0)))
    order = points[np.argsort(points[:, axis])]
    chunks = [x for x in np.array_split(order, min(parts, max(1, len(order) // 24))) if len(x)]
    origin = np.median(points, axis=0)
    shapes = []
    for chunk in chunks:
        # GEOM_MESH without the concave flag is converted by Bullet to a convex
        # hull. Quantiles reject isolated depth outliers without inflating an
        # oblique object into an axis-aligned box.
        lo, hi = np.quantile(chunk, (0.01, 0.99), axis=0)
        kept = chunk[np.all((chunk >= lo) & (chunk <= hi), axis=1)]
        hull_points = kept if len(kept) >= 4 else chunk
        shapes.append(
            p.createCollisionShape(
                p.GEOM_MESH,
                vertices=(np.asarray(hull_points) - origin).tolist(),
                physicsClientId=client,
            )
        )
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=shapes[0],
        basePosition=origin.tolist(),
        linkMasses=[0.0] * (len(shapes) - 1),
        linkCollisionShapeIndices=shapes[1:],
        linkVisualShapeIndices=[-1] * (len(shapes) - 1),
        linkPositions=[[0.0, 0.0, 0.0]] * (len(shapes) - 1),
        linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * (len(shapes) - 1),
        linkInertialFramePositions=[[0.0, 0.0, 0.0]] * (len(shapes) - 1),
        linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * (len(shapes) - 1),
        linkParentIndices=[0] * (len(shapes) - 1),
        linkJointTypes=[p.JOINT_FIXED] * (len(shapes) - 1),
        linkJointAxis=[[0.0, 0.0, 0.0]] * (len(shapes) - 1),
        physicsClientId=client,
    )
    for link in range(-1, len(shapes) - 1):
        p.changeDynamics(
            body,
            link,
            lateralFriction=float(settings["object_friction"]),
            restitution=float(settings["restitution"]),
            physicsClientId=client,
        )
    return body, origin


def _scene(p, clouds, dynamic, settings, client):
    plane = settings["table_plane_base"]
    normal = np.asarray(plane["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    offset = float(plane["offset_m"])
    table_shape = p.createCollisionShape(
        p.GEOM_PLANE, planeNormal=normal.tolist(), physicsClientId=client
    )
    table = p.createMultiBody(
        0, table_shape, basePosition=(-offset * normal).tolist(), physicsClientId=client
    )
    p.changeDynamics(
        table, -1, lateralFriction=float(settings["table_friction"]), physicsClientId=client
    )
    bodies, origins = {}, {}
    for key, points in clouds.items():
        bodies[key], origins[key] = _compound(
            p,
            points,
            int(settings["convex_parts"]),
            float(settings["object_mass_kg"]) if dynamic else 0,
            settings,
            client,
        )
    return table, bodies, origins


def _gripper(p, pose, width, client):
    from tools.pybullet_model import AG_LINKS, link_indices, load_robot, set_gripper

    robot = load_robot(use_fixed_base=True)
    links = link_indices(robot)
    for index in range(-1, p.getNumJoints(robot)):
        name = (
            p.getBodyInfo(robot)[0].decode()
            if index < 0
            else p.getJointInfo(robot, index)[12].decode()
        )
        p.setCollisionFilterGroupMask(
            robot,
            index,
            1 if name in AG_LINKS else 0,
            1 if name in AG_LINKS else 0,
            physicsClientId=client,
        )
    if width is None:
        set_gripper(robot, 1.0, use_motor=False)
    else:
        # Solve against the URDF kinematics instead of assuming a linear
        # width-to-joint mapping.
        desired = max(0.0, float(width))
        best = (float("inf"), 0.0)
        for closure in np.linspace(0.0, 1.0, 81):
            set_gripper(robot, closure, use_motor=False)
            left = np.asarray(
                p.getLinkState(robot, links["left_finger_pad"], physicsClientId=client)[0]
            )
            right = np.asarray(
                p.getLinkState(robot, links["right_finger_pad"], physicsClientId=client)[0]
            )
            error = abs(float(np.linalg.norm(left - right)) - desired)
            if error < best[0]:
                best = (error, float(closure))
        set_gripper(robot, best[1], use_motor=False)
    tcp = links["tcp_link"]
    current_pos, current_quat = p.getLinkState(robot, tcp, physicsClientId=client)[:2]
    inv = p.invertTransform(current_pos, current_quat)
    base_pos, base_quat = p.multiplyTransforms(pose[:3], pose[3:7], *inv)
    p.resetBasePositionAndOrientation(robot, base_pos, base_quat, physicsClientId=client)
    return robot, links


def _grasp_contact_report(p, robot, table, bodies, target, allowed, client):
    neighbor_ids = set()
    table_collision = False
    target_non_finger_collision = False
    for body_key, body in [(None, table), *bodies.items()]:
        for contact in p.getContactPoints(robot, body, physicsClientId=client):
            gripper_link = int(contact[3])
            if body_key is None:
                table_collision = True
            elif body_key == target:
                target_non_finger_collision |= gripper_link not in allowed
            else:
                neighbor_ids.add(int(body_key))
    free = not (table_collision or target_non_finger_collision or neighbor_ids)
    return {
        "collision_free": free,
        "reason": ("table" if table_collision else
                   "target_non_finger" if target_non_finger_collision else
                   "neighbor" if neighbor_ids else ""),
        "table_collision": table_collision,
        "target_non_finger_collision": target_non_finger_collision,
        "neighbor_ids": sorted(neighbor_ids),
    }


def _grasp_one(args):
    candidate, scene_path, target, settings = args
    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        clouds = _instances(scene_path)
        table, bodies, _ = _scene(p, clouds, False, settings, client)
        robot, links = _gripper(
            p, candidate["grasp_pose_world"], candidate["grasp_width_m"], client
        )
        p.performCollisionDetection(physicsClientId=client)
        allowed = {
            links[name]
            for name in ("left_finger", "left_finger_pad", "right_finger", "right_finger_pad")
        }
        return {
            "candidate_index": candidate["candidate_index"],
            **_grasp_contact_report(p, robot, table, bodies, target, allowed, client),
        }
    finally:
        p.disconnect(client)


def _min_gap(a, b):
    return float(cKDTree(a[:, :2]).query(b[:, :2], k=1)[0].min())


def transformed_cloud(points, origin, final_position, final_quaternion):
    """Apply a PyBullet rigid-body pose to points stored around body origin."""
    x, y, z, w = np.asarray(final_quaternion, dtype=float)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (np.asarray(points) - np.asarray(origin)) @ rotation.T + np.asarray(final_position)


def _push_one(args):
    batch_index, candidate, scene_path, target, settings = args
    import pybullet as p

    client = p.connect(p.DIRECT)
    started = time.perf_counter()
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(float(settings["simulation_time_step_s"]), physicsClientId=client)
        clouds = _instances(scene_path)
        table, bodies, origins = _scene(p, clouds, True, settings, client)
        acted = int(candidate["acted_object"])
        direction = np.asarray(candidate["push_direction_world"], float)
        direction[2] = 0
        direction /= max(np.linalg.norm(direction), 1e-9)
        contact = np.asarray(candidate["push_contact_world"], float)
        yaw = math.atan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, math.pi / 2, yaw])
        # Settle the independently reconstructed scene before measuring the
        # common initial state used by every candidate.
        for _ in range(int(settings["settle_steps"])):
            p.stepSimulation(physicsClientId=client)
        initial_clouds = {}
        for key, body in bodies.items():
            position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=client)
            initial_clouds[key] = transformed_cloud(
                clouds[key], origins[key], position, orientation
            )
        robot, _ = _gripper(p, [*contact, *quat], None, client)
        p.performCollisionDetection(physicsClientId=client)
        initial_collision = bool(p.getContactPoints(robot, table, physicsClientId=client))
        initial_collision |= any(
            key != acted and bool(p.getContactPoints(robot, body, physicsClientId=client))
            for key, body in bodies.items()
        )
        if initial_collision:
            return {
                "batch_index": batch_index,
                "candidate_index": candidate["candidate_index"],
                "effective": False,
                "distance_gain_m": 0.0,
                "direction_projection_m": 0.0,
                "reason": "initial_gripper_penetration",
                "simulation_s": time.perf_counter() - started,
            }
        before_gap = _min_gap(initial_clouds[acted], initial_clouds[target])
        steps = max(
            1,
            math.ceil(
                0.15
                / (float(settings["push_speed_m_s"]) * float(settings["simulation_time_step_s"]))
            ),
        )
        base0, orientation = p.getBasePositionAndOrientation(robot, physicsClientId=client)
        for step in range(1, steps + 1):
            position = np.asarray(base0) + direction * (0.15 * step / steps)
            p.resetBasePositionAndOrientation(robot, position, orientation, physicsClientId=client)
            p.stepSimulation(physicsClientId=client)
        for _ in range(int(settings["settle_steps"])):
            p.stepSimulation(physicsClientId=client)
        final_pos, final_quat = p.getBasePositionAndOrientation(
            bodies[acted], physicsClientId=client
        )
        target_pos, target_quat = p.getBasePositionAndOrientation(
            bodies[target], physicsClientId=client
        )
        moved = transformed_cloud(clouds[acted], origins[acted], final_pos, final_quat)
        moved_target = transformed_cloud(clouds[target], origins[target], target_pos, target_quat)
        after_gap = _min_gap(moved, moved_target)
        initial_acted_center = np.mean(initial_clouds[acted], axis=0)
        displacement = np.mean(moved, axis=0) - initial_acted_center
        projection = float(displacement[:2] @ direction[:2])
        return {
            "batch_index": batch_index,
            "candidate_index": candidate["candidate_index"],
            "effective": after_gap > before_gap
            and projection > float(settings["effective_displacement_m"]),
            "distance_gain_m": after_gap - before_gap,
            "direction_projection_m": projection,
            "simulation_s": time.perf_counter() - started,
        }
    finally:
        p.disconnect(client)


def main():
    print(json.dumps({"ready": True}), flush=True)
    pool = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["command"] == "close":
                    return
                settings = request["settings"]
                if request["command"] == "grasp":
                    jobs = [
                        (x, request["scene"], request["target"], settings)
                        for x in request["candidates"]
                    ]
                    result = [_grasp_one(x) for x in jobs]
                elif request["command"] == "push":
                    jobs = [
                        (i, x, request["scene"], request["target"], settings)
                        for i, x in enumerate(request["candidates"])
                    ]
                    if pool is None:
                        pool = ProcessPoolExecutor(max_workers=int(settings["parallel_workers"]))
                    result = list(pool.map(_push_one, jobs))
                else:
                    raise ValueError("unknown command")
                print(json.dumps({"ok": True, "result": result}), flush=True)
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
                        }
                    ),
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
