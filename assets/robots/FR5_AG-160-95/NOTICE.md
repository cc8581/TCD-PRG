# Sources and attribution

- FR5 meshes and the original `fr5v6.urdf` were taken from the user-provided
  `fr5_description` asset folder.
- `AG95.step` is the user-provided DH-Robotics AG-160-95 CAD assembly.
- The AG parallel-linkage URDF structure and optimized per-link meshes were
  adapted from [`ian-chuang/dh_ag95_gripper_ros2`](https://github.com/ian-chuang/dh_ag95_gripper_ros2),
  whose included license is copied to
  `source/third_party/dh_ag95_gripper_ros2/LICENSE`.
- The combined URDF, mount transform, TCP frames, one-kilogram mass
  normalization, PyBullet collision filtering, validation and examples in this
  directory are project-specific work.
