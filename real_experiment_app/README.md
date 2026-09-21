# TCD-PRG 仿真与真机实验平台

运行流程以 `docs/real_experiment_workflow_v2.md` 为准。仿真与真机共享“实例分割 → 目标确认 → 目标抓取预测”阶段；抓取不可行时才进入“压覆及邻接阻挡诊断 → 规则生成 PUSH → Push Evaluator 评分”分支。两个分支同时显示，当前分支高亮。

## 启动

在项目根目录执行：

```powershell
D:\Anaconda\install\python.exe -m real_experiment_app.main
```

界面顶部的滑动开关在“仿真”和“真机”之间切换。右侧有两套独立功能区：仿真显示本地场景及验证按钮，隐藏真机设备按钮；真机显示“连接相机”“连接机械臂”及采集、执行按钮，隐藏仿真功能区。设备使能、夹爪与回原点等操作位于“设备高级操作”折叠区。连接前确认机器人工作区和急停状态。

## 本地场景验证

在仿真模式下点击 **刷新场景**，列出 `configs/local_paths.yaml` 中
`dataset_root` 下已发布的 `scene_XXXX`。刷新不会自动选择场景，且会清除旧选择。
明确选择场景、状态编号和任务编号，再点击 **加载本地场景**。
加载只读取 `observation_cache_dir`（通常为 `F:\TCD-PRG-observations`）中的既有点云缓存；
缓存缺失时结果区会列出同一任务已缓存的状态索引，供手动更改；系统不会自动切换、
渲染新缓存或修改训练缓存。成功加载后手动点击 **2 实例分割**。
界面会将数据集目标映射到预测实例，并填入可核对的类别和功能区。
确认后依次点击 **3 确认目标**、**4 目标抓取预测**；抓取不可行时继续点击
PUSH 分支的压覆与邻接阻挡诊断、规则生成和评分。仿真仅展示结果，不连接或驱动真机。
切换到真机模式会清空当前场景与决策；任务运行或设备已连接时禁止切换。

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
push scoring. Corresponding stages require their configured checkpoint files,
so the formal Stage-C run must first publish
`push_evaluator_best.pt`.

For a new real-device task, capture a scene, run instance segmentation, then
select the target instance, category and functional region. A valid task grasp
goes directly to operator-confirmed execution. If task grasp fails, infer the
obstruction, generate PUSH rule candidates, score them with Push Evaluator,
then confirm the selected PUSH before sending it to FR5. The old scene and
action are invalidated after PUSH; acquire and segment again before continuing.
There is no PICK_REMOVE or PyBullet decision stage in this workflow.

## Coordinate contract

- Fused XYZ and model output translations: metres in FR5 base coordinates.
- Model grasp quaternion: `qx,qy,qz,qw`.
- FR5 Cartesian commands: millimetres and degrees.
- Model local `+Z`: approach direction. Pregrasp is generated along local `-Z`.
- TCP compensation is composed as
  `T_base_robot_tcp = T_base_model_tcp @ T_model_tcp_robot_tcp`.

The app intentionally does not implement grasp certification, automatic result
judgement, experiment recording, metrics, or unattended execution.
