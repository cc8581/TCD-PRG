# TCD-PRG collision-aware grasp planning

This module turns a predicted TCD-PRG grasp into a collision-checked MoveIt 2
trajectory for the FR5 + AG-160-95 model. Each segmented instance is converted
to solidified collision voxels; the target remains a separate collision object.
OMPL plans current-to-pregrasp and pregrasp-to-grasp trajectories. During the
final approach and gripper close, the ACM permits only the configured finger and
finger-pad links to contact the target. Contact with all non-target geometry
remains forbidden. A successful plan is retained by ID and execution reuses the
same arm and gripper trajectories before attaching the target to `tcp_link`.

## Build (WSL2)

```bash
cd /mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## Start MoveIt and the reusable service

Run in two WSL terminals:

```bash
source /opt/ros/jazzy/setup.bash
source /mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws/install/setup.bash
ros2 launch fr5_ag160_95_moveit_config demo.launch.py
```

```bash
source /opt/ros/jazzy/setup.bash
source /mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws/install/setup.bash
ros2 launch tcd_prg_motion_planner scene_grasp_planner.launch.py
```

## Dataset end-to-end validation

From the repository root, using the same Python environment as the application:

```powershell
python -m real_experiment_app.motion_planning.dataset_validation `
  --scene-id 0 --state-id 0 --task-index 0 --execute `
  --output D:\Codex运行垃圾\TCD-PRG\motion_planning\validation.json
```

`--execute` drives the MoveIt fake controller and verifies the reported TCP
position and orientation against the model-predicted grasp. It does not command
the physical FR5 controller.

The formal application calls the same module after grasp candidate generation.
Candidates are submitted in score order, the first fully valid arm-and-gripper
plan is stored in the decision, and execution uses that saved plan ID. If every
candidate fails, the decision stops with an explicit MoveIt failure reason.
