#!/usr/bin/env python3
"""Send one NPZ scene/grasp request to the persistent MoveIt planner service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from shape_msgs.msg import Mesh, MeshTriangle
from tcd_prg_motion_planner.srv import PlanGrasp


def scalar(data, name: str, default):
    return np.asarray(data[name]).item() if name in data else default


def unpack_meshes(data, prefix: str) -> list[Mesh]:
    """Rebuild variable-length triangle meshes from pickle-free NPZ arrays."""
    vertex_name = f"{prefix}_mesh_vertices_m"
    triangle_name = f"{prefix}_mesh_triangles"
    vertex_offset_name = f"{prefix}_mesh_vertex_offsets"
    triangle_offset_name = f"{prefix}_mesh_triangle_offsets"
    if vertex_name not in data:
        return []

    vertices = np.asarray(data[vertex_name], np.float64).reshape(-1, 3)
    triangles = np.asarray(data[triangle_name], np.int64).reshape(-1, 3)
    vertex_offsets = np.asarray(data[vertex_offset_name], np.int64).reshape(-1)
    triangle_offsets = np.asarray(data[triangle_offset_name], np.int64).reshape(-1)
    if not np.isfinite(vertices).all():
        raise ValueError(f"{prefix} mesh vertices contain non-finite values")
    if len(vertex_offsets) != len(triangle_offsets) or len(vertex_offsets) < 1:
        raise ValueError(f"{prefix} mesh offset arrays are inconsistent")
    if vertex_offsets[0] != 0 or triangle_offsets[0] != 0:
        raise ValueError(f"{prefix} mesh offsets must begin at zero")
    if vertex_offsets[-1] != len(vertices) or triangle_offsets[-1] != len(triangles):
        raise ValueError(f"{prefix} mesh offsets do not cover packed arrays")
    if np.any(np.diff(vertex_offsets) < 0) or np.any(np.diff(triangle_offsets) < 0):
        raise ValueError(f"{prefix} mesh offsets must be monotonic")

    meshes: list[Mesh] = []
    for index in range(len(vertex_offsets) - 1):
        v0, v1 = int(vertex_offsets[index]), int(vertex_offsets[index + 1])
        t0, t1 = int(triangle_offsets[index]), int(triangle_offsets[index + 1])
        local_vertices = vertices[v0:v1]
        local_triangles = triangles[t0:t1]
        if not len(local_vertices) or not len(local_triangles):
            raise ValueError(f"{prefix} mesh {index} is empty")
        if local_triangles.min(initial=0) < 0 or local_triangles.max(initial=-1) >= len(local_vertices):
            raise ValueError(f"{prefix} mesh {index} has an out-of-range triangle index")
        mesh = Mesh()
        mesh.vertices = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in local_vertices]
        mesh.triangles = []
        for a, b, c in local_triangles:
            triangle = MeshTriangle()
            triangle.vertex_indices = [int(a), int(b), int(c)]
            mesh.triangles.append(triangle)
        meshes.append(mesh)
    return meshes


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
            environment_meshes = target_meshes = []
            collision_geometry = ""
        else:
            pose = np.asarray(data["grasp_pose_xyzw"], np.float64)
            environment = np.asarray(
                data.get("environment_centers_m", np.empty((0, 3))), np.float64
            ).reshape(-1, 3)
            target = np.asarray(
                data.get("target_centers_m", np.empty((0, 3))), np.float64
            ).reshape(-1, 3)
            environment_meshes = unpack_meshes(data, "environment")
            target_meshes = unpack_meshes(data, "target")
            collision_geometry = str(scalar(data, "collision_geometry", "voxel"))
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
        request.collision_geometry = collision_geometry
        request.obstacle_voxel_size_m = float(scalar(data, "obstacle_voxel_size_m", 0.02))
        request.target_voxel_size_m = float(scalar(data, "target_voxel_size_m", 0.01))
        request.collision_padding_m = float(scalar(data, "collision_padding_m", 0.01))
        request.environment_centers = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in environment]
        request.target_centers = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in target]
        request.environment_meshes = environment_meshes
        request.target_meshes = target_meshes
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
    all_meshes = list(request.environment_meshes) + list(request.target_meshes)
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
        "collision_geometry": request.collision_geometry,
        "obstacle_voxels": len(request.environment_centers) + len(request.target_centers),
        "obstacle_meshes": len(all_meshes),
        "obstacle_triangles": sum(len(mesh.triangles) for mesh in all_meshes),
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
