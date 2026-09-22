# Path-planning implementation review

Date: 2026-09-22

## Scope reviewed

- Windows Python client and temporary-file lifecycle
- ROS 2 service schema and request validation
- MoveIt planning-scene construction, ACM handling, saved-plan execution and attachment
- Formal controller and decision-flow integration
- Isolated predictor-worker recovery used by the UI perception action

## Resolved findings

1. The target is retained as `tcd_target`; environment and target geometry are sent separately.
2. Final approach and gripper close use an ACM that permits only the configured finger and pad links to contact `tcd_target`. The original ACM is restored on success and exception paths.
3. A candidate is accepted only after current-to-pregrasp, pregrasp-to-grasp, and gripper-close planning succeed. These exact saved trajectories are executed by plan ID, then the target is attached to `tcp_link`.
4. `executed` comes from the service's observed trajectory-execution state. Full workflow completion and target attachment are separate response fields.
5. The ROS service rejects unknown operations, missing IDs, invalid frames, non-finite poses/centers, invalid voxel/table dimensions, missing target geometry, and unknown or duplicate touch links.
6. Request/result files are deleted in `finally`; abandoned files older than the active timeout window are removed when the client starts.
7. Formal decision code submits collision-free candidates in order, stores the first complete MoveIt plan, executes that same plan, and stops with an explicit reason when every candidate fails.
8. Visible segmented surfaces are solidified per observed XY column. The target uses a separate 10 mm voxel size; environment voxels default to 20 mm. This avoids treating one global object height range as occupied in every column.
9. Network query IDs and physical segmentation instance IDs now have separate fields. Candidate matching uses the query ID; collision geometry uses the physical instance ID.
10. The UI perception crash was caused by writing to a dead predictor subprocess pipe. The client now detects that state, reports it clearly, and the controller recreates the worker and retries perception once.

## Verification evidence

- Python regression suite: `112 passed`.
- ROS package build: `tcd_prg_motion_planner` completed successfully with `colcon build`.
- Dataset scene 0 exercised prediction and sequential candidate planning with target and environment collision geometry. Under the strict contact policy, all candidates in the final run were rejected safely, mostly at the final approach stage.
- An earlier FakeSystem run executed the saved transit and approach trajectories to a predicted pose. It exposed a non-target collision during gripper close; close validation was then moved into planning, preventing that unsafe candidate from being selected.
- Stale request/result IPC files were reduced to zero by the implemented cleanup path.

## Remaining validation boundary

No tested dataset candidate has yet completed the new strict sequence through gripper close and target attachment. The current code therefore has verified failure handling and partial FakeSystem execution, but no end-to-end success evidence for the final stricter contract. Physical FR5 execution is also untested because the robot is unavailable. These are validation gaps, not grounds to relax the ACM or ignore non-target collisions.
