from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    config = MoveItConfigsBuilder(
        "fr5_ag160_95", package_name="fr5_ag160_95_moveit_config"
    ).to_moveit_configs()
    return LaunchDescription([
        Node(
            package="tcd_prg_motion_planner",
            executable="scene_grasp_planner",
            output="screen",
            parameters=[config.to_dict()],
        )
    ])
