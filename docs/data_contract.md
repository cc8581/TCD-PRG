# Unified data, coordinates and label semantics

## Variable-length PTv3 point protocol

Formal observations reconstruct the three allowed cameras at the configured
render resolution. `dataset.scene_points=0` preserves this complete fused
observation instead of forcing every scene to 16,384 points. At collation,
Pointcept-compatible `GridSample` selects one representative from each 5 mm
voxel: using Pointcept's randomized train rule and deterministically during
validation, offline evaluation, candidate generation, and deployment. Scenes
remain variable length and are padded only at the dense-head boundary with an
explicit `point_mask`; PTv3 itself receives packed `coord/grid_coord/feat/batch`.
Decoder features are aligned back to all retained supervision points through
the voxel inverse map. Multi-megapixel source-camera pixels are resized before
reconstruction and are never padded into a training batch.

## Sample unit

The sampling key is `(scene_id, state_id, task_index, group_index)`. Each
`UnifiedSample` owns a `SceneObservation`, `StateLabels`, one
`ActionCandidateGroup`, and zero or more `SequenceLabels`. Identifiers are local
keys, not learned continuous quantities. Stable object UUIDs are
`scene_XXXX/object_YY`, equivalent to `(scene_id, object_index)`.

Published scenes are deterministically shuffled with `training.seed`. The
configured `training.data_fraction` is selected first, then those scenes are
partitioned according to `training.split_ratios` into train/val or
train/val/test. Splitting is scene-level, so states and action groups from one
scene cannot cross partitions. `training_index.h5` remains an efficient action
group lookup table, but its legacy split column is ignored.

Padded tensors use explicit `point_mask`, `object_mask`, `object_active` and
`candidate_mask`. Object count, category count, point count, view count and
functional-region vocabulary are not hardcoded to the current dataset.

## Observation

- XYZ and translations: world frame, metres, float32.
- RGB: float32 `[0,1]`.
- pose quaternion: `(qx,qy,qz,qw)` after translation.
- `instance_id`: grouping/association only.
- `target_mask`: external mask equality with target instance.
- `object_present`: the scene catalog/pose slot contains this object.
- `object_active`: the object has not been removed from the current state.
- after PICK_REMOVE: `present=true`, `active=false`.
- `physical_active = object_present & object_active`: the only object domain
  used for current geometry pooling, global-grasp assignment and exact
  collision certification.
- cameras: exactly the allowed external sensor profiles; Oracle is rejected.

Intermediate states are reconstructed from source scene, state object poses,
present/active state, model IDs/scales, render seed and camera profile. A
bounded read-through cache serves hits; DataLoader workers render misses before
the resulting batch reaches the GPU.

## Grasp and gripper

World grasp poses are `[tx,ty,tz,qx,qy,qz,qw]`. Rotation matrices are proper
right-handed matrices. The model's approach vector is the candidate local +Z
axis; in-plane rotation is quantized around that axis. The pose translation is
the dataset's canonical contact midpoint. Legacy source-library depth remains
available for audit compatibility but is not a predicted AG pose parameter.

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

The current action vocabulary permits target self-push as a recovery primitive;
successful sequence labels contain such actions. This exception is explicit in
configuration. Other learned preparation candidates are constrained by the
derived actionable frontier, with at most the configured bounded fallback.

## Candidate status and validity

- `POSITIVE`: evaluated successful action.
- `NEGATIVE`: evaluated unsuccessful action.
- `UNKNOWN_UNTESTED`: no outcome statement; never a negative.

`label_set_complete` is false by default and may be true only for an explicitly
versioned, fully certified candidate universe with no unknown rows. Without
that declaration, unmatched grasp queries are ignored. A query becomes a
negative only when it is geometrically associated with an explicit evaluated
negative; the absence of a sampled positive label is not a negative outcome.

Every regression/result field has its own validity mask, including
`after_state_valid`, `after_pose_valid`, `potential_after_valid`,
`acted_object_motion_valid` and `target_motion_valid`. NaN is storage for
not-applicable values and is removed by its mask before loss/metric arithmetic.
Family subtotals normalize by the weights of child losses with at least one
valid supervised element in the current batch; absent supervision contributes
neither a zero pseudo-target nor denominator weight.

After learned verification and deterministic certification, TASK_GRASP
candidates are deduplicated per object in SE(3). The state-level count is the
number of unique retained grasps. Configuration requires
`task_grasp_candidates >= max_required_grasp_count`.

PUSH has an exact 0.15 m displacement. Raw top/side fields are retained only
for dataset and execution compatibility; the learning policy does not predict
or supervise an approach mode. PICK_REMOVE is a reliable high-level manipulation primitive;
transport and safe placement are certified/executed by the non-learning layer
and are not inferred as universally feasible from a positive macro label.
