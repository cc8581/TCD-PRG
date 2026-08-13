from __future__ import annotations

import math
import numpy as np


def quaternion_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64)
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], np.float64)


def matrix_to_quaternion_xyzw(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, np.float64)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s,
             (r[1,0]-r[0,1])/s, 0.25*s]
    elif r[0,0] > r[1,1] and r[0,0] > r[2,2]:
        s = math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2
        q = [0.25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s,
             (r[2,1]-r[1,2])/s]
    elif r[1,1] > r[2,2]:
        s = math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2
        q = [(r[0,1]+r[1,0])/s, 0.25*s, (r[1,2]+r[2,1])/s,
             (r[0,2]-r[2,0])/s]
    else:
        s = math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2
        q = [(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, 0.25*s,
             (r[1,0]-r[0,1])/s]
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def pose7_to_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, np.float64)
    if pose.shape != (7,):
        raise ValueError("pose must be [x,y,z,qx,qy,qz,qw]")
    result = np.eye(4)
    result[:3, :3] = quaternion_xyzw_to_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def matrix_to_pose7(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, np.float64)
    return np.r_[transform[:3, 3], matrix_to_quaternion_xyzw(transform[:3, :3])]


def rpy_degrees_to_matrix(rpy) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(rpy, np.float64))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def matrix_to_rpy_degrees(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, np.float64)
    pitch = math.atan2(-r[2,0], math.hypot(r[0,0], r[1,0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(r[2,1], r[2,2])
        yaw = math.atan2(r[1,0], r[0,0])
    else:
        roll = math.atan2(-r[1,2], r[1,1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def xyz_rpy_to_matrix(values, translation_scale: float = 1.0) -> np.ndarray:
    values = np.asarray(values, np.float64)
    result = np.eye(4)
    result[:3, :3] = rpy_degrees_to_matrix(values[3:])
    result[:3, 3] = values[:3] * translation_scale
    return result


def model_pose_to_robot_pose(model_pose, model_tcp_to_robot_tcp) -> np.ndarray:
    """Return FR5 [mm,mm,mm,deg,deg,deg] for the configured robot TCP.

    ``model_tcp_to_robot_tcp`` is T_model_tcp_from_robot_tcp. Therefore the
    commanded robot TCP frame is T_base_model @ T_model_robot.
    """
    target = pose7_to_matrix(model_pose) @ np.asarray(model_tcp_to_robot_tcp, np.float64)
    return np.r_[target[:3, 3] * 1000.0, matrix_to_rpy_degrees(target[:3, :3])]


def offset_model_pose(model_pose, local_xyz_m) -> np.ndarray:
    transform = pose7_to_matrix(model_pose)
    transform[:3, 3] += transform[:3, :3] @ np.asarray(local_xyz_m, np.float64)
    return matrix_to_pose7(transform)


def push_pose(contact, direction) -> np.ndarray:
    z = np.asarray(direction, np.float64)
    z /= max(np.linalg.norm(z), 1e-12)
    reference = np.array([0., 0., 1.])
    if abs(float(np.dot(z, reference))) > 0.9:
        reference = np.array([0., 1., 0.])
    x = np.cross(reference, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.r_[np.asarray(contact, np.float64), matrix_to_quaternion_xyzw(np.column_stack((x,y,z)))]

