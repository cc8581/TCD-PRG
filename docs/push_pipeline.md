# Independent PUSH evaluation and rule proposals

Training and inference share `PushActions`: batch_index [K] int64, object [K]
int64 (PushCondition slot), contact_world [K,3] metres, direction_world [K,3]
unit horizontal vectors, push_distance [K] metres. No point index, bin label,
GT outcome, post-action state, or proposal-network feature enters this contract.

## Training
Run `python train_push_evaluator.py --perception-checkpoint PATH --output PATH`.
The default config is configs/stage/push_evaluator.yaml. Geometry is frozen and
computed online from the current augmented cloud. The evaluator trains its own
scene/action fusion layers. Only logged evaluated actions are used; no above-object
filter, contact sampling, nearest-point distance gate, or forcing is called.
UNKNOWN actions are ignored. Missing/invalid action parameters, invisible target,
or an unrepresented object are excluded (not relabelled as negative).
The dataset stroke is 0.15 m. Labels are action_improves_state; this is improvement,
not a guarantee of immediate graspability or robot feasibility.
Periodic and final validation score held-out logged actions, not sampled actions.

## Inference
Perception + target -> multiple objects above the whole target XY footprint ->
mid-height contour samples at 10 mm spacing -> upstream half-contour ->
task-away direction rotated toward contact-to-object centre ->
the same evaluator -> global score sorting, Top-32 and NMS.
The centre is estimated from observed bounds, not simulator COM. Defaults use
20/70 degree angular blending thresholds and 0.15/0.75 weights, matching the
generator's blocker-clearing rule. Self-pushing and side-only obstacles are outside
the requested inference object-selection scope. Convex contours are proposals,
not certified surface contacts. Occluded or degenerate geometry may yield no proposals.
No robot reachability or collision certification is added in this change.

`evaluate_push_pipeline.py` evaluates rule candidates separately under GT and
predicted perception. Matches to evaluated actions are used only for metrics;
unmatched candidates remain UNKNOWN. This is not simulated execution success.

Only perception + grasp + push_evaluator checkpoints are required for combined
deployment. Evaluator protocol 2 rejects old proposal-dependent weights and checks
the frozen perception fingerprint. There is no train_push.py, val_push.py, or
learned Contact/Direction/Object proposal head. The shared legacy loss-family
configuration slots remain for A/B checkpoint/config compatibility; enabling the
old PUSH objective is explicitly rejected.

## Checks
Tests cover a shared GT/rule action interface with identical scores; no generator
call during GT training; distant logged contact retention; UNKNOWN exclusion;
multiple upper objects; off-footprint object rejection; frozen geometry gradients;
checkpoint provenance; empty/invalid actions; and opposite-direction overfitting.
Synthetic overfitting verifies trainability, not real-world PUSH effectiveness.
