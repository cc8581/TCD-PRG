# Unified data, coordinates and label semantics

## Sample unit

The sampling key is `(scene_id, state_id, task_index, group_index)`. Each
`UnifiedSample` owns a `SceneObservation`, `StateLabels`, one
`ActionCandidateGroup`, and zero or more `SequenceLabels`. Identifiers are local
keys, not learned continuous quantities. Stable object UUIDs are
`scene_XXXX/object_YY`, equivalent to `(scene_id, object_index)`.

Padded tensors use explicit `point_mask`, `object_mask`, `object_active` and
`candidate_mask`. Object count, category count, point count, view count and
functional-region vocabulary are not hardcoded to the current dataset.

## Observation

- XYZ and translations: world frame, metres, float32.
- RGB: float32 `[0,1]`.
- pose quaternion: `(qx,qy,qz,qw)` after translation.
- `instance_id`: grouping/association only.
- `target_mask`: external mask equality with target instance.
- `object_present`: body still exists in the state.
- `object_active`: body may still be manipulated.
- after PICK_REMOVE: `present=true`, `active=false`.
- cameras: exactly the allowed external sensor profiles; Oracle is rejected.

Intermediate states are reconstructed from source scene, state object poses,
present/active state, model IDs/scales, render seed and camera profile. The GPU
training loop only reads saved/cache observations.

## Grasp and gripper

World grasp poses are `[tx,ty,tz,qx,qy,qz,qw]`. Rotation matrices are proper
right-handed matrices. The model's approach vector is the candidate local +Z
axis; in-plane rotation is quantized around that axis. Depth is in metres.

`grasp_width_m` means total AG-160-95 finger opening. The formal range is
0–0.095 m. Exact gripper collision points are sampled from the supplied URDF in
the `tcp_link` frame and transformed to each candidate pose. Width is not fixed
and must not be confused with the 0.15 m PUSH displacement.

## Relations

Canonical physical channels are `near`, `contact`, `support`, `press`,
`occlude`. `press(i,j)` is the transpose semantic of `support`. Native
`block_path` contributes to the task relation `block_grasp_approach`, not a
physical edge. Task channels are `block_task_region`, `block_task_grasp`, and
`block_grasp_approach`, directed from object to the `TASK_GRASP` node.

Direct blockers have a task edge. Indirect blockers are propagated through
support/press dependencies. Disabling indirect reasoning leaves direct task
edges intact. Invalid topology only masks topology-order loss; it does not mask
action, outcome, policy or result supervision.

## Candidate status and validity

- `POSITIVE`: evaluated successful action.
- `NEGATIVE`: evaluated unsuccessful action.
- `UNKNOWN_UNTESTED`: no outcome statement; never a negative.

Every regression/result field has its own validity mask, including
`after_state_valid`, `after_pose_valid`, `potential_after_valid`,
`acted_object_motion_valid` and `target_motion_valid`. NaN is storage for
not-applicable values and is removed by its mask before loss/metric arithmetic.

PUSH has an exact 0.15 m displacement. Top and side refer to closed-gripper
approach modes. PICK_REMOVE is a reliable high-level manipulation primitive;
transport and safe placement are certified/executed by the non-learning layer
and are not inferred as universally feasible from a positive macro label.
