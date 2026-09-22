#!/usr/bin/env python3
"""Send one NPZ scene/grasp request to the persistent MoveIt planner service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from tcd_prg_motion_planner.srv import PlanGrasp


def scalar(data, name: str, default):
    return np.asarray(data[name]).item() if name in data else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    with np.load(args.request, allow_pickle=False) as data:
        operation = int(scalar(data, "operation", 0))
        if operation not in (
            PlanGrasp.Request.PLAN,
            PlanGrasp.Request.EXECUTE_SAVED,
            PlanGrasp.Request.CLEAR_SAVED,
        ):
            raise ValueError(f"unsupported PlanGrasp operation: {operation}")
        request = PlanGrasp.Request()
        request.operation = operation
        request.plan_id = str(scalar(data, "plan_id", ""))
        if operation != PlanGrasp.Request.PLAN:
            pose = np.asarray([0, 0, 0, 0, 0, 0, 1], np.float64)
            environment = target = np.empty((0, 3), np.float64)
        else:
            pose = np.asarray(data["grasp_pose_xyzw"], np.float64)
            environment = np.asarray(data["environment_centers_m"], np.float64).reshape(-1, 3)
            target = np.asarray(data["target_centers_m"], np.float64).reshape(-1, 3)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("grasp_pose_xyzw must be finite [x,y,z,qx,qy,qz,qw]")
        if not np.isfinite(environment).all() or not np.isfinite(target).all():
            raise ValueError("collision centers contain non-finite values")
        request.grasp_pose.header.frame_id = "base_link"
        request.grasp_pose.pose.position.x = float(pose[0])
        request.grasp_pose.pose.position.y = float(pose[1])
        request.grasp_pose.pose.position.z = float(pose[2])
        request.grasp_pose.pose.orientation.x = float(pose[3])
        request.grasp_pose.pose.orientation.y = float(pose[4])
        request.grasp_pose.pose.orientation.z = float(pose[5])
        request.grasp_pose.pose.orientation.w = float(pose[6])
        request.pregrasp_distance_m = float(scalar(data, "pregrasp_distance_m", 0.10))
        request.obstacle_voxel_size_m = float(scalar(data, "obstacle_voxel_size_m", 0.02))
        request.target_voxel_size_m = float(scalar(data, "target_voxel_size_m", 0.01))
        request.collision_padding_m = float(scalar(data, "collision_padding_m", 0.01))
        request.environment_centers = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in environment]
        request.target_centers = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in target]
        request.touch_links = [str(value) for value in data.get(
            "touch_links", [
                "left_finger", "left_finger_pad", "right_finger", "right_finger_pad"
            ]
        )]
        request.attach_target_after_execute = bool(
            scalar(data, "attach_target_after_execute", True)
        )
        request.add_table = bool(scalar(data, "add_table", True))
        request.table_z_m = float(scalar(data, "table_z_m", 0.0))
        table_size = np.asarray(data.get("table_size_m", [1.2, 1.2, 0.04]), np.float64)
        if table_size.shape != (3,) or not np.isfinite(table_size).all():
            raise ValueError("table_size_m must contain three finite values")
        request.table_size_m.x = float(table_size[0])
        request.table_size_m.y = float(table_size[1])
        request.table_size_m.z = float(table_size[2])

    rclpy.init()
    node = rclpy.create_node("tcd_prg_plan_grasp_client")
    client = node.create_client(PlanGrasp, "/tcd_prg/plan_grasp")
    if not client.wait_for_service(timeout_sec=args.timeout):
        raise RuntimeError("/tcd_prg/plan_grasp service is unavailable")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
    if not future.done():
        raise TimeoutError("MoveIt grasp planning timed out")
    response = future.result()
    if response is None:
        raise RuntimeError("MoveIt grasp planning returned no response")
    achieved = response.achieved_pose.pose
    result = {
        "success": bool(response.success),
        "target_attached": bool(response.target_attached),
        "plan_id": response.plan_id,
        "failed_stage": response.failed_stage,
        "message": response.message,
        "transit_points": len(response.transit_trajectory.joint_trajectory.points),
        "approach_points": len(response.approach_trajectory.joint_trajectory.points),
        "achieved_pose_xyzw": [
            achieved.position.x,
            achieved.position.y,
            achieved.position.z,
            achieved.orientation.x,
            achieved.orientation.y,
            achieved.orientation.z,
            achieved.orientation.w,
        ],
        "position_error_m": response.position_error_m,
        "orientation_error_rad": response.orientation_error_rad,
        "executed": bool(response.trajectory_executed),
        "execution_completed": bool(response.execution_completed),
        "obstacle_voxels": len(request.environment_centers) + len(request.target_centers),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.result:
        Path(args.result).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    node.destroy_node()
    rclpy.shutdown()
    if not response.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
