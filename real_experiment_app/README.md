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
2. Enable one or more cameras and set each IP.
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

## Coordinate contract

- Fused XYZ and model output translations: metres in FR5 base coordinates.
- Model grasp quaternion: `qx,qy,qz,qw`.
- FR5 Cartesian commands: millimetres and degrees.
- Model local `+Z`: approach direction. Pregrasp is generated along local `-Z`.
- TCP compensation is composed as
  `T_base_robot_tcp = T_base_model_tcp @ T_model_tcp_robot_tcp`.

The app intentionally does not implement grasp certification, automatic result
judgement, experiment recording, metrics, or unattended execution.
