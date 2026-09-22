# TCD-PRG collision-aware grasp planning

This module turns a predicted TCD-PRG grasp into a collision-checked MoveIt 2
trajectory for the FR5 + AG-160-95 model. The target remains a separate
collision object. OMPL plans current-to-pregrasp and pregrasp-to-grasp
trajectories. During the final approach and gripper close, the ACM permits only
the configured finger and finger-pad links to contact the target. Contact with
all non-target geometry remains forbidden. A successful plan is retained by ID
and execution reuses the same arm and gripper trajectories before attaching the
target to `tcp_link`.

## Collision geometry

The collision representation is selected by
`motion_planning.collision_geometry` in
`real_experiment_app/configs/real_experiment.yaml`.

- `hybrid` — **recommended/default**. Segmented environment objects use closed
  convex-hull triangle meshes; the target uses a tighter alpha-shape mesh when
  stable. Unknown/background points and degenerate point sets remain conservative
  legacy voxels. Alpha failure automatically falls back to a convex hull and then
  voxels. This keeps environment planning robust while preserving more target
  shape detail for grasp contact and attached-object collision checking.
- `convex_hull` — one convex hull per instance, including unknown IDs. This is
  smooth and fast, but can bridge disconnected/concave unknown geometry.
- `alpha_shape` — a 3-D alpha-complex boundary for a tighter concave surface.
  Sparse or ill-conditioned instances automatically fall back to a convex hull,
  then to voxels. Tune `alpha_radius_m`; this mode is intentionally not the
  default because depth clouds are partial and noisy.
- `obb` — a PCA-oriented bounding box per segmented instance. It is very cheap
  and stable, but more conservative than a hull.
- `voxel` — the previous behavior: per-instance solidified box voxels only.

`collision_padding_m` still enlarges voxel primitives. Mesh generation applies
an approximate outward radial padding before hull/alpha reconstruction; OBB
extents are expanded exactly by the requested amount. The default remains zero.

The table is still a native MoveIt box primitive. The target object can contain
one or more mesh/voxel shapes under the same `tcd_target` collision-object ID, so
existing ACM contact policy and `attachObject("tcd_target", ...)` behavior are
preserved.

Recommended settings:

```yaml
motion_planning:
  collision_geometry: hybrid
  voxel_size_m: 0.02
  target_voxel_size_m: 0.01
  collision_padding_m: 0.0
  alpha_radius_m: 0.025
  mesh_max_points: 2500
```

The repository's former implementation did **not** publish a MoveIt OctoMap in
this path: it converted segmented point clouds to many `BOX` primitives inside
`tcd_environment` and `tcd_target`. The visual/planning artifact is nevertheless
the same class of voxel stair-stepping problem. The new mesh modes replace most
of those boxes with `shape_msgs/Mesh` geometry while retaining the voxel path as
an explicit fallback.

## Build (WSL2)

The service definition now depends on `shape_msgs/Mesh`, so rebuild the ROS
workspace after applying this patch:

```bash
cd /mnt/d/pycharm/Project/TCD-PRG/real_experiment_app/motion_planning/ros2_ws
source /opt/ros/jazzy/setup.bash
rm -rf build/tcd_prg_motion_planner install/tcd_prg_motion_planner
colcon build --packages-select tcd_prg_motion_planner
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

The planner log reports the active representation and geometry size, for
example:

```text
Collision geometry=hybrid env_voxels=... target_voxels=... meshes=... triangles=...
```

## Dataset end-to-end validation

From the repository root, using the same Python environment as the application:

```powershell
python -m real_experiment_app.motion_planning.dataset_validation `
  --scene-id 0 --state-id 0 --task-index 0 --execute `
  --output D:\Codex运行垃圾\TCD-PRG\motion_planning\validation.json
```

The validator now reads the same `motion_planning` geometry settings as the
formal application, so changing `collision_geometry` in the YAML affects both
paths consistently.

`--execute` drives the MoveIt fake controller and verifies the reported TCP
position and orientation against the model-predicted grasp. It does not command
the physical FR5 controller.

The formal application calls the same module after grasp candidate generation.
Candidates are submitted in score order, the first fully valid arm-and-gripper
plan is stored in the decision, and execution uses that saved plan ID. If every
candidate fails, the decision stops with an explicit MoveIt failure reason.
