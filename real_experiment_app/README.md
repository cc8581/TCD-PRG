# TCD-PRG minimal real experiment app

This folder is independent of training and offline validation. The first
version implements the minimal repeated workflow:

`RGB-D capture -> instance segmentation -> point-cloud fusion -> task input -> TCD-PRG prediction -> operator-confirmed FR5 action -> capture again`.

## Start the production application

From the project root:

```powershell
D:\Anaconda\install\python.exe -m real_experiment_app.main
```

The application has no simulation entry in the delivered UI. It connects to
the configured Mech-Eye cameras, FR5 and AG-160-95. Keep the robot workspace
clear and verify the controller emergency stop before connecting.

## Real devices

Edit `configs/real_experiment.yaml`:

1. Open **设备设置** to configure FR5 IP, tool/user coordinates, motion speed,
   camera enable/IP settings, and AG-160-95 parameters. Settings are persisted
   atomically to `configs/real_experiment.yaml` and take effect on reconnect.
2. Configure the single Mech-Eye camera IP and external calibration. Stage B
   uses logical reference view 2, so the physical camera's `model_view_index`
   remains 2. This mapping does not imply three physical cameras.
3. Fill every enabled camera's calibrated 4x4 `camera_to_robot_base` matrix.
   Translation must be in metres. An absent matrix deliberately blocks startup.
4. Configure the instance segmentation command. It receives `--input` and
   `--output`; output NPZ must contain an `instance_image` array matching the
   RGB image. Optional `category_keys` and `category_values` arrays provide
   per-instance category IDs. Local IDs are associated across views using the
   calibrated 3D centroids and mean RGB colors.
5. Tune `model_tcp_to_robot_tcp`. It is the fixed transform
   `T_model_tcp_from_robot_tcp`, entered as `[x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg]`.
   The same six values can be adjusted and applied at runtime from the collapsed
   TCP compensation group on the main screen. Runtime changes are intentionally
   not written back automatically; copy confirmed values into the YAML config.

The real FR5 path reuses the previously validated Fairino and AG-160-95
controller initialization sequence: `SetGripperConfig -> reset -> activate ->
MoveGripper`. It retains controller IK checks and requires explicit confirmation
before every predicted action. The device card also exposes manual gripper
initialize/open/close controls for commissioning.

Connecting leaves FR5 servo power disabled. The operator must explicitly enable
it before action execution. The UI provides state refresh, fault reset, enable,
disable, pause/resume, and confirmed return-to-home controls. Software stop uses
an independent RPC channel to send `StopMotion`, `ProgramStop`, and
`RobotEnable(0)` even while the normal motion channel is waiting. Closing the
application also stops and disables FR5. The configured motion workspace gates
every generated pregrasp, contact, push endpoint, lift, and removal waypoint.

## Model and closed-loop contract

The app loads the three stage checkpoints independently: Stage A perception,
Stage B task grasping, and Stage C push evaluation. The Stage-C checkpoint must
contain its training FPS protocol; the app applies the same compiled FPS before
push scoring. Startup intentionally fails until all three configured files
exist, so the current formal Stage-C run must first publish
`push_evaluator_best.pt`.

For a new task, capture a scene and select the target instance, category, and
functional region. After a predicted PUSH or PICK_REMOVE is executed, the old
scene and action are invalidated. Capture again and run prediction: the policy
re-identifies the original physical target from its stored 3D/appearance prompt
and recommends the next action. Category and functional region stay locked for
the whole task. A TASK_GRASP ends the task and requires an operator reset before
another target can be selected.

## Coordinate contract

- Fused XYZ and model output translations: metres in FR5 base coordinates.
- Model grasp quaternion: `qx,qy,qz,qw`.
- FR5 Cartesian commands: millimetres and degrees.
- Model local `+Z`: approach direction. Pregrasp is generated along local `-Z`.
- TCP compensation is composed as
  `T_base_robot_tcp = T_base_model_tcp @ T_model_tcp_robot_tcp`.

The app intentionally does not implement grasp certification, automatic result
judgement, experiment recording, metrics, or unattended execution.
