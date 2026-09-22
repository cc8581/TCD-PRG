#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tcd_prg_motion_planner/srv/plan_grasp.hpp>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
using MoveGroup = moveit::planning_interface::MoveGroupInterface;
using Request = tcd_prg_motion_planner::srv::PlanGrasp::Request;
using Response = tcd_prg_motion_planner::srv::PlanGrasp::Response;

geometry_msgs::msg::Pose normalizedPose(const geometry_msgs::msg::Pose& input)
{
  geometry_msgs::msg::Pose pose = input;
  const double norm = std::sqrt(
    pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
  if (!std::isfinite(norm) || norm < 1e-8) {
    throw std::invalid_argument("grasp quaternion has zero or non-finite norm");
  }
  pose.orientation.x /= norm;
  pose.orientation.y /= norm;
  pose.orientation.z /= norm;
  pose.orientation.w /= norm;
  return pose;
}

geometry_msgs::msg::Pose makePregrasp(const geometry_msgs::msg::Pose& grasp, double distance)
{
  const auto pose = normalizedPose(grasp);
  const Eigen::Quaterniond rotation(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  geometry_msgs::msg::Pose result = pose;
  const Eigen::Vector3d offset = rotation * Eigen::Vector3d::UnitZ() * distance;
  result.position.x -= offset.x();
  result.position.y -= offset.y();
  result.position.z -= offset.z();
  return result;
}

moveit::core::RobotState endState(
  const moveit::core::RobotState& start,
  const moveit_msgs::msg::RobotTrajectory& trajectory)
{
  moveit::core::RobotState result(start);
  const auto& joint_trajectory = trajectory.joint_trajectory;
  if (joint_trajectory.points.empty()) {
    throw std::runtime_error("planned trajectory contains no joint points");
  }
  result.setVariablePositions(
    joint_trajectory.joint_names, joint_trajectory.points.back().positions);
  result.update();
  return result;
}

geometry_msgs::msg::Pose poseFromState(
  const moveit::core::RobotState& state, const std::string& link)
{
  const Eigen::Isometry3d transform = state.getGlobalLinkTransform(link);
  const Eigen::Quaterniond quaternion(transform.rotation());
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  return pose;
}

std::pair<double, double> poseError(
  const geometry_msgs::msg::Pose& actual, const geometry_msgs::msg::Pose& target)
{
  const double dx = actual.position.x - target.position.x;
  const double dy = actual.position.y - target.position.y;
  const double dz = actual.position.z - target.position.z;
  const double position = std::sqrt(dx * dx + dy * dy + dz * dz);
  Eigen::Quaterniond a(
    actual.orientation.w, actual.orientation.x, actual.orientation.y, actual.orientation.z);
  Eigen::Quaterniond b(
    target.orientation.w, target.orientation.x, target.orientation.y, target.orientation.z);
  a.normalize();
  b.normalize();
  const double dot = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return {position, 2.0 * std::acos(dot)};
}

bool validMesh(const shape_msgs::msg::Mesh& mesh)
{
  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    return false;
  }
  for (const auto& vertex : mesh.vertices) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) || !std::isfinite(vertex.z)) {
      return false;
    }
  }
  for (const auto& triangle : mesh.triangles) {
    for (const auto index : triangle.vertex_indices) {
      if (index >= mesh.vertices.size()) {
        return false;
      }
    }
  }
  return true;
}

std::string planId()
{
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  std::ostringstream stream;
  stream << std::hex << ticks;
  return stream.str();
}
}  // namespace

class SceneGraspPlanner
{
  struct SavedPlan
  {
    MoveGroup::Plan transit;
    MoveGroup::Plan approach;
    MoveGroup::Plan gripper_close;
    geometry_msgs::msg::Pose target_pose;
    moveit_msgs::msg::AllowedCollisionMatrix original_acm;
    moveit_msgs::msg::AllowedCollisionMatrix contact_acm;
    std::vector<std::string> touch_links;
    bool attach_target{true};
  };

public:
  explicit SceneGraspPlanner(const rclcpp::Node::SharedPtr& node)
  : node_(node), arm_(node_, "arm"), gripper_(node_, "gripper")
  {
    arm_.setPoseReferenceFrame("base_link");
    arm_.setEndEffectorLink("tcp_link");
    arm_.setPlanningPipelineId("ompl");
    arm_.setPlannerId("RRTConnectkConfigDefault");
    arm_.setPlanningTime(8.0);
    arm_.setNumPlanningAttempts(10);
    arm_.setGoalPositionTolerance(0.002);
    arm_.setGoalOrientationTolerance(0.01);
    arm_.setMaxVelocityScalingFactor(0.20);
    arm_.setMaxAccelerationScalingFactor(0.20);
  }

  void handle(const std::shared_ptr<Request> request, std::shared_ptr<Response> response)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    response->success = false;
    response->failed_stage = "validation";
    try {
      validate(*request);
      if (request->operation == Request::PLAN) {
        updateScene(*request);
        plan(*request, *response);
      } else if (request->operation == Request::EXECUTE_SAVED) {
        executeSaved(*request, *response);
      } else {
        saved_.erase(request->plan_id);
        response->success = true;
        response->message = "saved plan cleared";
      }
    } catch (const std::exception& error) {
      response->message = error.what();
      RCLCPP_ERROR(node_->get_logger(), "Grasp planning failed: %s", error.what());
    }
  }

private:
  void validate(const Request& request) const
  {
    if (request.operation != Request::PLAN &&
        request.operation != Request::EXECUTE_SAVED &&
        request.operation != Request::CLEAR_SAVED) {
      throw std::invalid_argument("operation is not a supported PlanGrasp operation");
    }
    if (request.operation != Request::PLAN) {
      if (request.plan_id.empty()) {
        throw std::invalid_argument("plan_id is required for saved-plan operations");
      }
      return;
    }
    if (request.grasp_pose.header.frame_id != "base_link") {
      throw std::invalid_argument("grasp_pose.frame_id must be base_link");
    }
    const auto& p = request.grasp_pose.pose.position;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      throw std::invalid_argument("grasp position contains a non-finite value");
    }
    (void)normalizedPose(request.grasp_pose.pose);
    if (!(request.pregrasp_distance_m > 0.0 && request.pregrasp_distance_m <= 0.30)) {
      throw std::invalid_argument("pregrasp_distance_m must lie in (0, 0.30]");
    }
    const std::unordered_set<std::string> valid_geometry_modes = {
      "voxel", "hybrid", "convex_hull", "alpha_shape", "obb"};
    if (valid_geometry_modes.find(request.collision_geometry) == valid_geometry_modes.end()) {
      throw std::invalid_argument("collision_geometry is not supported");
    }
    const auto valid_centers = [](const auto& centers) {
      return std::all_of(centers.begin(), centers.end(), [](const auto& point) {
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
      });
    };
    if (!valid_centers(request.environment_centers) || !valid_centers(request.target_centers)) {
      throw std::invalid_argument("collision centers must be finite");
    }
    const auto valid_meshes = [](const auto& meshes) {
      return std::all_of(meshes.begin(), meshes.end(), [](const auto& mesh) {
        return validMesh(mesh);
      });
    };
    if (!valid_meshes(request.environment_meshes) || !valid_meshes(request.target_meshes)) {
      throw std::invalid_argument("collision meshes are empty, non-finite, or malformed");
    }
    if (!request.environment_centers.empty() &&
        !(request.obstacle_voxel_size_m >= 0.005 && request.obstacle_voxel_size_m <= 0.10)) {
      throw std::invalid_argument("obstacle_voxel_size_m must lie in [0.005, 0.10]");
    }
    if (!request.target_centers.empty() &&
        !(request.target_voxel_size_m >= 0.005 && request.target_voxel_size_m <= 0.10)) {
      throw std::invalid_argument("target_voxel_size_m must lie in [0.005, 0.10]");
    }
    if (!(request.collision_padding_m >= 0.0 && request.collision_padding_m <= 0.05)) {
      throw std::invalid_argument("collision_padding_m must lie in [0, 0.05]");
    }
    if ((request.target_centers.empty() && request.target_meshes.empty()) ||
        request.touch_links.empty()) {
      throw std::invalid_argument("target geometry and touch_links are required");
    }
    std::unordered_set<std::string> unique_touch_links;
    for (const auto& link : request.touch_links) {
      if (!arm_.getRobotModel()->hasLinkModel(link)) {
        throw std::invalid_argument("touch_links contains unknown robot link: " + link);
      }
      if (!unique_touch_links.insert(link).second) {
        throw std::invalid_argument("touch_links contains a duplicate robot link: " + link);
      }
    }
    if (request.add_table &&
        (!std::isfinite(request.table_z_m) || !std::isfinite(request.table_size_m.x) ||
         !std::isfinite(request.table_size_m.y) || !std::isfinite(request.table_size_m.z) ||
         request.table_size_m.x <= 0.0 || request.table_size_m.y <= 0.0 ||
         request.table_size_m.z <= 0.0)) {
      throw std::invalid_argument("table position and dimensions must be finite and positive");
    }
  }

  void updateScene(const Request& request)
  {
    scene_.removeCollisionObjects({"tcd_environment", "tcd_target", "tcd_table"});
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    const auto add_geometry = [&](const auto& centers, const auto& meshes,
                                  const std::string& id, double voxel_size) {
      if (centers.empty() && meshes.empty()) {
        return;
      }
      moveit_msgs::msg::CollisionObject object;
      object.header.frame_id = "base_link";
      object.id = id;
      object.operation = moveit_msgs::msg::CollisionObject::ADD;
      object.primitives.reserve(centers.size());
      object.primitive_poses.reserve(centers.size());
      const double size = voxel_size + 2.0 * request.collision_padding_m;
      for (const auto& center : centers) {
        shape_msgs::msg::SolidPrimitive voxel;
        voxel.type = shape_msgs::msg::SolidPrimitive::BOX;
        voxel.dimensions = {size, size, size};
        geometry_msgs::msg::Pose pose;
        pose.position = center;
        pose.orientation.w = 1.0;
        object.primitives.push_back(voxel);
        object.primitive_poses.push_back(pose);
      }
      object.meshes = meshes;
      object.mesh_poses.resize(meshes.size());
      for (auto& pose : object.mesh_poses) {
        pose.orientation.w = 1.0;
      }
      objects.push_back(std::move(object));
    };
    add_geometry(
      request.environment_centers, request.environment_meshes,
      "tcd_environment", request.obstacle_voxel_size_m);
    add_geometry(
      request.target_centers, request.target_meshes,
      "tcd_target", request.target_voxel_size_m);
    if (request.add_table) {
      moveit_msgs::msg::CollisionObject table;
      table.header.frame_id = "base_link";
      table.id = "tcd_table";
      table.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive box;
      box.type = shape_msgs::msg::SolidPrimitive::BOX;
      box.dimensions = {
        request.table_size_m.x, request.table_size_m.y, request.table_size_m.z};
      geometry_msgs::msg::Pose pose;
      pose.position.x = 0.5;
      pose.position.y = 0.0;
      pose.position.z = request.table_z_m - request.table_size_m.z / 2.0;
      pose.orientation.w = 1.0;
      table.primitives.push_back(box);
      table.primitive_poses.push_back(pose);
      objects.push_back(std::move(table));
    }
    if (!objects.empty() && !scene_.applyCollisionObjects(objects)) {
      throw std::runtime_error("MoveIt rejected the collision scene");
    }
    std::size_t triangle_count = 0;
    for (const auto& mesh : request.environment_meshes) {
      triangle_count += mesh.triangles.size();
    }
    for (const auto& mesh : request.target_meshes) {
      triangle_count += mesh.triangles.size();
    }
    RCLCPP_INFO(
      node_->get_logger(),
      "Collision geometry=%s env_voxels=%zu target_voxels=%zu meshes=%zu triangles=%zu",
      request.collision_geometry.c_str(), request.environment_centers.size(),
      request.target_centers.size(),
      request.environment_meshes.size() + request.target_meshes.size(), triangle_count);
  }

  void plan(const Request& request, Response& response)
  {
    auto current = arm_.getCurrentState(3.0);
    if (!current) {
      throw std::runtime_error("current robot state is unavailable");
    }
    const auto grasp = normalizedPose(request.grasp_pose.pose);
    const auto pregrasp = makePregrasp(grasp, request.pregrasp_distance_m);
    response.failed_stage = "current_to_pregrasp";
    arm_.setStartState(*current);
    arm_.setPoseTarget(pregrasp, "tcp_link");
    MoveGroup::Plan transit;
    const bool transit_ok = static_cast<bool>(arm_.plan(transit));
    arm_.clearPoseTargets();
    if (!transit_ok) {
      throw std::runtime_error("OMPL could not plan current state to pregrasp");
    }

    const auto original_acm = getAllowedCollisionMatrix();
    collision_detection::AllowedCollisionMatrix acm(original_acm);
    for (const auto& link : request.touch_links) {
      acm.setEntry("tcd_target", link, true);
    }
    moveit_msgs::msg::AllowedCollisionMatrix contact_acm;
    acm.getMessage(contact_acm);

    auto pregrasp_state = endState(*current, transit.trajectory);
    response.failed_stage = "pregrasp_to_grasp";
    applyAllowedCollisionMatrix(contact_acm);
    arm_.setStartState(pregrasp_state);
    arm_.setPoseTarget(grasp, "tcp_link");
    MoveGroup::Plan approach;
    MoveGroup::Plan gripper_close;
    bool approach_ok = false;
    try {
      approach_ok = static_cast<bool>(arm_.plan(approach));
      arm_.clearPoseTargets();
    } catch (...) {
      arm_.clearPoseTargets();
      try {
        applyAllowedCollisionMatrix(original_acm);
      } catch (...) {
      }
      throw;
    }
    if (!approach_ok) {
      applyAllowedCollisionMatrix(original_acm);
      throw std::runtime_error("OMPL could not plan ACM-limited pregrasp-to-grasp approach");
    }

    auto grasp_state = endState(pregrasp_state, approach.trajectory);
    response.failed_stage = "validate_gripper_close";
    gripper_.setStartState(grasp_state);
    gripper_.setNamedTarget("closed");
    bool close_ok = false;
    try {
      close_ok = static_cast<bool>(gripper_.plan(gripper_close));
      gripper_.clearPoseTargets();
      applyAllowedCollisionMatrix(original_acm);
    } catch (...) {
      gripper_.clearPoseTargets();
      try {
        applyAllowedCollisionMatrix(original_acm);
      } catch (...) {
      }
      throw;
    }
    if (!close_ok) {
      throw std::runtime_error("gripper cannot close without contacting non-target geometry");
    }
    auto achieved = poseFromState(grasp_state, "tcp_link");
    auto [position_error, orientation_error] = poseError(achieved, grasp);
    if (position_error > 0.003 || orientation_error > 0.02) {
      throw std::runtime_error("planned endpoint does not reach the requested grasp pose");
    }

    const std::string id = planId();
    saved_[id] = SavedPlan{
      transit, approach, gripper_close, grasp, original_acm, contact_acm,
      request.touch_links, request.attach_target_after_execute};
    response.success = true;
    response.plan_id = id;
    response.failed_stage.clear();
    response.message = "segmented plan saved";
    response.transit_trajectory = transit.trajectory;
    response.approach_trajectory = approach.trajectory;
    response.achieved_pose.header.frame_id = "base_link";
    response.achieved_pose.header.stamp = node_->now();
    response.achieved_pose.pose = achieved;
    response.position_error_m = position_error;
    response.orientation_error_rad = orientation_error;
  }

  moveit_msgs::msg::AllowedCollisionMatrix getAllowedCollisionMatrix()
  {
    using Service = moveit_msgs::srv::GetPlanningScene;
    auto client = node_->create_client<Service>("/get_planning_scene");
    if (!client->wait_for_service(std::chrono::seconds(3))) {
      throw std::runtime_error("get_planning_scene service is unavailable");
    }
    auto request = std::make_shared<Service::Request>();
    request->components.components =
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
    auto future = client->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
      throw std::runtime_error("timed out reading allowed collision matrix");
    }
    return future.get()->scene.allowed_collision_matrix;
  }

  void applyAllowedCollisionMatrix(const moveit_msgs::msg::AllowedCollisionMatrix& acm)
  {
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    diff.allowed_collision_matrix = acm;
    if (!scene_.applyPlanningScene(diff)) {
      throw std::runtime_error("MoveIt rejected allowed collision matrix update");
    }
  }

  void executeSaved(const Request& request, Response& response)
  {
    const auto found = saved_.find(request.plan_id);
    if (found == saved_.end()) {
      throw std::invalid_argument("saved plan_id is unknown or expired");
    }
    const SavedPlan plan = found->second;
    response.plan_id = request.plan_id;
    response.failed_stage = "open_gripper";
    gripper_.setNamedTarget("open");
    if (gripper_.move() != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error("failed to open gripper before saved trajectory execution");
    }
    response.failed_stage = "execute_transit";
    applyAllowedCollisionMatrix(plan.original_acm);
    if (arm_.execute(plan.transit) != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error("failed to execute saved current-to-pregrasp trajectory");
    }
    response.failed_stage = "execute_approach";
    applyAllowedCollisionMatrix(plan.contact_acm);
    moveit::core::MoveItErrorCode execution;
    try {
      execution = arm_.execute(plan.approach);
    } catch (...) {
      try {
        applyAllowedCollisionMatrix(plan.original_acm);
      } catch (...) {
      }
      throw;
    }
    if (execution != moveit::core::MoveItErrorCode::SUCCESS) {
      applyAllowedCollisionMatrix(plan.original_acm);
      throw std::runtime_error("failed to execute saved pregrasp-to-grasp trajectory");
    }
    response.trajectory_executed = true;
    auto achieved = arm_.getCurrentPose("tcp_link").pose;
    auto [position_error, orientation_error] = poseError(achieved, plan.target_pose);
    if (position_error > 0.005 || orientation_error > 0.03) {
      throw std::runtime_error("executed endpoint does not reach the planned grasp pose");
    }
    response.failed_stage = "close_gripper";
    if (gripper_.execute(plan.gripper_close) != moveit::core::MoveItErrorCode::SUCCESS) {
      applyAllowedCollisionMatrix(plan.original_acm);
      throw std::runtime_error("failed to close gripper at grasp pose");
    }
    if (plan.attach_target) {
      response.failed_stage = "attach_target";
      if (!arm_.attachObject("tcd_target", "tcp_link", plan.touch_links)) {
        applyAllowedCollisionMatrix(plan.original_acm);
        throw std::runtime_error("failed to attach target collision object to tcp_link");
      }
      response.target_attached = true;
    }
    applyAllowedCollisionMatrix(plan.original_acm);
    response.success = true;
    response.execution_completed = true;
    response.failed_stage.clear();
    response.message = "saved plan executed";
    response.transit_trajectory = plan.transit.trajectory;
    response.approach_trajectory = plan.approach.trajectory;
    response.achieved_pose.header.frame_id = "base_link";
    response.achieved_pose.header.stamp = node_->now();
    response.achieved_pose.pose = achieved;
    response.position_error_m = position_error;
    response.orientation_error_rad = orientation_error;
    saved_.erase(found);
  }

  rclcpp::Node::SharedPtr node_;
  MoveGroup arm_;
  MoveGroup gripper_;
  moveit::planning_interface::PlanningSceneInterface scene_;
  std::mutex mutex_;
  std::unordered_map<std::string, SavedPlan> saved_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(
    "tcd_prg_scene_grasp_planner",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  auto planner = std::make_shared<SceneGraspPlanner>(node);
  // Planning waits for fresh joint-state callbacks. Keep the service callback
  // reentrant so those subscriptions can run in the multithreaded executor.
  auto service_group = node->create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  auto service = node->create_service<tcd_prg_motion_planner::srv::PlanGrasp>(
    "/tcd_prg/plan_grasp",
    [planner](const std::shared_ptr<Request> request, std::shared_ptr<Response> response) {
      planner->handle(request, response);
    }, rclcpp::ServicesQoS(), service_group);
  (void)service;
  RCLCPP_INFO(node->get_logger(), "Ready on /tcd_prg/plan_grasp");
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
