cmake_minimum_required(VERSION 3.8)
project(tcd_prg_motion_planner)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(shape_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanGrasp.srv"
  DEPENDENCIES geometry_msgs moveit_msgs
)

add_executable(scene_grasp_planner src/scene_grasp_planner.cpp)
ament_target_dependencies(scene_grasp_planner
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  shape_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(scene_grasp_planner "${cpp_typesupport_target}")

install(TARGETS scene_grasp_planner DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/plan_grasp_request.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
