"""Batch exact FR5/AG-160-95 IK, approach-path and collision certification."""

from __future__ import print_function

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as pb


ARM_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
BASE_POSITION = (0.50, 0.68, 0.0)
BASE_YAW_DEG = -90.0
PREGRASP_M = 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-root", required=True)
    parser.add_argument("--runtime-mesh-root", required=True)
    return parser.parse_args()


def normalize(value, fallback):
    norm = float(np.sqrt(np.sum(value * value)))
    return np.asarray(fallback, dtype=np.float64) if norm < 1e-9 else value / norm


class Certifier(object):
    def __init__(self, robot_root, runtime_mesh_root, request):
        sys.path.insert(0, str(robot_root))
        from tools.pybullet_model import joint_indices, link_indices, load_robot, set_gripper

        self.set_gripper = set_gripper
        self.mesh_root = runtime_mesh_root
        table_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=(0.5, 0.5, 0.02))
        self.table = pb.createMultiBody(0, table_shape, -1, (0.5, 0.0, -0.02))
        base_quat = pb.getQuaternionFromEuler((0, 0, math.radians(BASE_YAW_DEG)))
        self.robot = load_robot(BASE_POSITION, base_quat)
        joints, links = joint_indices(self.robot), link_indices(self.robot)
        self.tcp = links["tcp_closed_front_link"]
        self.pad_links = {links["left_finger_pad"], links["right_finger_pad"]}
        self.arm = [joints[name] for name in ARM_NAMES]
        self.movable = [index for index in range(pb.getNumJoints(self.robot))
                        if pb.getJointInfo(self.robot, index)[2]
                        in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC)]
        self.lower, self.upper = [], []
        for index in self.movable:
            info = pb.getJointInfo(self.robot, index)
            low, high = float(info[8]), float(info[9])
            if high <= low:
                low, high = -math.pi, math.pi
            self.lower.append(low); self.upper.append(high)
        self.ranges = [high - low for low, high in zip(self.lower, self.upper)]
        self.home = [0.0] * len(self.movable)
        self.scene_bodies = []
        model_ids = request["object_model_ids"].astype(str)
        scales = request["object_scales"].astype(np.float64)
        poses = request["object_pose"].astype(np.float64)
        present = request["object_present"].astype(bool)
        for model_id, scale, pose, is_present in zip(model_ids, scales, poses, present):
            if not is_present:
                self.scene_bodies.append(-1)
                continue
            mesh = runtime_mesh_root / ("vhacd_v3_%s.obj" % model_id)
            shape = pb.createCollisionShape(pb.GEOM_MESH, fileName=str(mesh), meshScale=[scale] * 3)
            body = pb.createMultiBody(0, shape, -1, pose[:3].tolist(), pose[3:].tolist())
            self.scene_bodies.append(body)
        self.support_bodies = []
        for present_support, pose, size in zip(
            request["support_present"].astype(bool), request["support_pose"], request["support_size"]
        ):
            if not present_support:
                continue
            shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=(size * 0.5).tolist())
            self.support_bodies.append(pb.createMultiBody(
                0, shape, -1, pose[:3].tolist(), pose[3:].tolist()
            ))

    def solve(self, pose, rest):
        solution = pb.calculateInverseKinematics(
            self.robot, self.tcp, pose[:3].tolist(), pose[3:].tolist(),
            lowerLimits=self.lower, upperLimits=self.upper, jointRanges=self.ranges,
            restPoses=rest, maxNumIterations=120, residualThreshold=5e-5,
        )
        if len(solution) < len(self.movable):
            return None, "ik_no_solution"
        for index, value in zip(self.movable, solution):
            pb.resetJointState(self.robot, index, float(value))
        state = pb.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)
        position_error = float(np.linalg.norm(np.asarray(state[4]) - pose[:3]))
        angle_error = 2.0 * math.acos(min(1.0, max(-1.0, abs(float(np.dot(state[5], pose[3:]))))))
        if position_error > 0.005 or angle_error > math.radians(15):
            return None, "ik_pose_error"
        for index in self.arm:
            value = float(pb.getJointState(self.robot, index)[0])
            info = pb.getJointInfo(self.robot, index)
            if not float(info[8]) + math.radians(1) <= value <= float(info[9]) - math.radians(1):
                return None, "joint_limit_margin"
        return list(solution[: len(self.movable)]), "ok"

    def collision_free(self, acted_object, allow_pad_contact):
        pb.performCollisionDetection()
        if pb.getClosestPoints(self.robot, self.table, distance=0.0005):
            return False, "table_collision"
        for index, body in enumerate(self.scene_bodies):
            if body < 0:
                continue
            contacts = pb.getClosestPoints(self.robot, body, distance=0.0005)
            if not contacts:
                continue
            if index == acted_object and allow_pad_contact:
                if all(int(contact[3]) in self.pad_links for contact in contacts):
                    continue
            return False, "scene_collision"
        for body in self.support_bodies:
            if pb.getClosestPoints(self.robot, body, distance=0.0005):
                return False, "support_collision"
        return True, "ok"

    @staticmethod
    def grasp_waypoints(pose):
        rotation = np.asarray(pb.getMatrixFromQuaternion(pose[3:])).reshape(3, 3)
        pre = pose.copy(); pre[:3] -= rotation[:, 2] * PREGRASP_M
        result = [pre]
        for alpha in (0.33, 0.66, 1.0):
            waypoint = pose.copy(); waypoint[:3] = pre[:3] * (1 - alpha) + pose[:3] * alpha
            result.append(waypoint)
        return result

    @staticmethod
    def push_waypoints(contact, direction):
        direction = normalize(direction, (1, 0, 0))
        tool_z = np.asarray((0.0, 0.0, -1.0))
        tool_y = direction
        tool_x = normalize(np.cross(tool_y, tool_z), (0, 1, 0))
        rotation = np.column_stack((tool_x, tool_y, tool_z))
        quaternion = np.asarray(pb.getQuaternionFromEuler((0, 0, 0)), dtype=np.float64)
        # Convert the proper rotation without SciPy.
        trace = np.trace(rotation)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2; quaternion = np.array([
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s, 0.25 * s])
        else:
            # PyBullet can robustly convert an equivalent basis via a view matrix;
            # use the explicit matrix-to-quaternion branches for non-positive trace.
            diagonal = int(np.argmax(np.diag(rotation)))
            if diagonal == 0:
                s = math.sqrt(1 + rotation[0,0]-rotation[1,1]-rotation[2,2])*2
                quaternion=np.array([.25*s,(rotation[0,1]+rotation[1,0])/s,
                    (rotation[0,2]+rotation[2,0])/s,(rotation[2,1]-rotation[1,2])/s])
            elif diagonal == 1:
                s=math.sqrt(1+rotation[1,1]-rotation[0,0]-rotation[2,2])*2
                quaternion=np.array([(rotation[0,1]+rotation[1,0])/s,.25*s,
                    (rotation[1,2]+rotation[2,1])/s,(rotation[0,2]-rotation[2,0])/s])
            else:
                s=math.sqrt(1+rotation[2,2]-rotation[0,0]-rotation[1,1])*2
                quaternion=np.array([(rotation[0,2]+rotation[2,0])/s,
                    (rotation[1,2]+rotation[2,1])/s,.25*s,(rotation[1,0]-rotation[0,1])/s])
        start = np.asarray(contact, dtype=np.float64) - direction * 0.01
        # PUSH uses one deterministic side-entry primitive. Approach selection
        # is execution geometry, not a learned or candidate-level decision.
        pre = start - direction * PREGRASP_M
        positions = [pre, start, start + direction * 0.075, start + direction * 0.15]
        return [np.r_[position, quaternion] for position in positions]

    def certify(self, kind, acted, pose, width, contact, direction):
        if int(kind) == 0:
            self.set_gripper(self.robot, 1.0, use_motor=False)
            waypoints = self.push_waypoints(contact, direction)
        else:
            closure = (0.095 - min(0.095, max(0.0, float(width)))) / 0.095
            self.set_gripper(self.robot, closure, use_motor=False)
            waypoints = self.grasp_waypoints(pose)
        previous = self.home
        for waypoint_index, waypoint in enumerate(waypoints):
            solution, reason = self.solve(waypoint, previous)
            if solution is None:
                return False, reason
            if previous is not self.home and np.max(np.abs(np.asarray(solution[:6])-np.asarray(previous[:6]))) > math.radians(45):
                return False, "joint_discontinuity"
            final_contact = waypoint_index == len(waypoints) - 1
            clear, reason = self.collision_free(int(acted), final_contact)
            if not clear:
                return False, reason
            previous = solution
        return True, "ok"


def main():
    args = parse_args()
    client = pb.connect(pb.DIRECT)
    try:
        with np.load(args.request, allow_pickle=False) as request:
            certifier = Certifier(Path(args.robot_root), Path(args.runtime_mesh_root), request)
            success, reasons = [], []
            for values in zip(request["action_type"], request["acted_object"],
                              request["pose_world"], request["width_m"],
                              request["contact_world"], request["direction_world"]):
                ok, reason = certifier.certify(*values); success.append(ok); reasons.append(reason)
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        np.savez_compressed(temporary, success=np.asarray(success, bool), reasons=np.asarray(reasons))
        temporary.replace(output)
    finally:
        pb.disconnect(client)


if __name__ == "__main__":
    main()
