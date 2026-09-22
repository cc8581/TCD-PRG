"""Application adapter that filters grasp candidates through the MoveIt module."""

from __future__ import annotations

from dataclasses import asdict

from .client import WSLMoveItPlanner


class GraspMotionPlanningStage:
    def __init__(self, app_config, planner: WSLMoveItPlanner | None = None):
        self.config = app_config
        settings = app_config.raw.get("motion_planning", {})
        self.settings = settings
        self.planner = planner or WSLMoveItPlanner(
            distro=str(settings.get("wsl_distro", "Ubuntu-24.04")),
            ros_workspace=str(settings.get(
                "ros_workspace",
                "/mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws",
            )),
            timeout_s=float(settings.get("timeout_s", 90.0)),
        )

    def plan_first(self, scene, candidates, target_instance: int):
        diagnostics = []
        robot = self.config.raw["robot"]
        settings = self.settings
        for candidate in candidates:
            result = self.planner.plan_grasp(
                grasp_pose_model_xyzw=candidate["grasp_pose_world"],
                model_tcp_to_robot_tcp=self.config.tcp_transform,
                scene_xyz_m=scene.xyz_m,
                scene_instance_id=scene.instance_id,
                target_instance=target_instance,
                pregrasp_distance_m=float(robot["pregrasp_distance_m"]),
                obstacle_voxel_size_m=float(settings.get("voxel_size_m", 0.02)),
                target_voxel_size_m=float(settings.get("target_voxel_size_m", 0.01)),
                collision_padding_m=float(settings.get("collision_padding_m", 0.0)),
                collision_geometry=str(settings.get("collision_geometry", "hybrid")),
                mesh_max_points=int(settings.get("mesh_max_points", 2500)),
                alpha_radius_m=float(settings.get("alpha_radius_m", 0.025)),
                table_z_m=float(settings.get("table_z_m", 0.0)),
                execute=False,
            )
            record = {"candidate_index": int(candidate["candidate_index"]), **asdict(result)}
            diagnostics.append(record)
            if result.success:
                selected = dict(candidate)
                selected["moveit_plan_id"] = result.plan_id
                selected["moveit_plan"] = record
                return selected, tuple(diagnostics)
        return None, tuple(diagnostics)

    def execute_saved(self, plan_id: str):
        return self.planner.execute_plan(plan_id)
