# GAPG audit and TCD-PRG implementation map

## Audited GAPG flow

The audited GAPG implementation uses GraspNet proposals, `Space_GraspFusion`
to score local scene–gripper geometry, and `Push_model` to rank sampled push
poses. Its policy is grasp-first, otherwise push, followed by a new observation.
The original method has no functional-region task token,
PICK_REMOVE action, explicit dependency graph, multi-positive hierarchical
router or state-dependent multi-step supervision.

TCD-PRG keeps only the baseline runtime modules actually imported by the worker
under `third_party/GAPG/`; obsolete GAPG data collection, training, demo and
UR5e/YCB simulation files are excluded. The wrapper in
`tcd_prg/baselines/gapg_wrapper.py` converts the same three-PRO-S fusion and
target instance mask to GAPG input and invokes that isolated subset in Python
3.8.

## Detailed mapping

| Design module | GAPG source reused | TCD-PRG implementation | Main input | Main output | Supervision/loss | Inference role |
|---|---|---|---|---|---|---|
| Shared encoder | official Pointcept PTv3 | `models/backbones/point_transformer_v3.py` adapter + pinned submodule | voxelized XYZ/RGB | point/object/target/global/task tokens | joint downstream gradients | exactly one scene pass; Windows non-Flash/Linux Flash |
| Functional region | none | `models/region/head.py` | target point features + task | point probability, visibility | focal BCE + Dice + visibility | constrains task grasp |
| Task grasp set | official M2T2 architecture pattern | shared PyTorch `TransformerDecoder` + task query bank | shared points + target/task/region | K complete poses, openings and qualities | geodesic Hungarian matching + smooth symmetric chordal training loss | TASK_GRASP candidates |
| Global grasp set | official M2T2 architecture pattern | same shared decoder + neutral global query bank | task-free scene points | K complete poses, openings, object assignments and scene qualities | object-balanced scene-certified Hungarian set loss | PICK_REMOVE candidates |
| Grasp verifier | PyTorch `TransformerEncoder` | `models/grasp_verifier/` | candidate-frame scene/gripper tokens + task | task-conditioned action outcome | outcome BCE | router evidence; hard gate is an ablation |
| Dependency graph | PyG `TransformerConv` | `models/dependency_graph/hgt.py` | object/task tokens + continuous predicted relation attributes | physical/task edges, hard masks and soft prior | two edge losses | soft candidate prior by default; hard/no-graph ablations |
| PUSH | PTv3 + PyTorch direction transformer | `models/push/head.py` | point/object/task/graph/geometry/steps + direction queries | object, contact, direction, per-direction utility delta | four module objectives | fixed 0.15 m action; hard feasibility is certified |
| Router | fixed grasp→push rule | `models/policy/router.py` | heterogeneous candidates + evidence | candidate score | multi-positive listwise | legal candidate selection |
| Closed loop | `grasp_push_eval.py` loop | `planners/closed_loop.py` | policy, observations, executor | H=5 result and trace | evaluated through replay/simulation | reobserve and replan |
| Exact grasp safety | PyBullet execution environment | `execution/pybullet_certifier.py` + worker | grasp + full state + FR5/AG URDF | valid/reason | deterministic, not learned | final grasp mask; PUSH motion planning is executor-owned |

At inference, verified TASK_GRASP poses are clustered per target object before
the adaptive state gate. The NMS metric uses translation, jaw-symmetric SO(3)
rotation, approach-axis angle, and AG opening. The gate compares the resulting
unique count—not the number of populated candidate slots—with the state's
`required_grasp_count`. The configured TASK_GRASP capacity must cover the
declared maximum required count. The default uses 64 queries for a maximum
required count of 20 and records query, NMS, verifier and certifier survivor
counts so gate failures can be diagnosed directly.

PUSH candidate semantics deliberately include target self-push as an explicit
recovery primitive because it occurs in successful generated sequences. It is
controlled by `model.allow_target_push_recovery`; disabling it restores strict
`active & actionable` target eligibility. Non-graph recovery for other objects
remains bounded by `graph_candidate_fallback_objects` in the hard-graph ablation.
The default soft-graph mode supplies a dependency prior instead of deleting every
candidate outside a thresholded closure. After routing, action-level PUSH NMS
removes same-object candidates with nearby contacts and directions.

The final objective exposes eleven paper-level modules. Translation, symmetric
SO(3), width, quality and object assignment remain internal diagnostics of each grasp
set loss rather than separate weighted tasks.

## Reuse, extension and replacement

Directly reused for the GAPG baseline: the minimal grasp/push network
definitions in `third_party/GAPG/`, required preprocessing utilities and
external checkpoints. Reused conceptually in the full
method: complete continuous grasp-set prediction and local gripper–scene
evaluation. Extended: three-view input wrapper, task conditioning, shared
features and closed-loop evaluation. Newly implemented task-specific components
are the unified task adapter, functional-region output head, continuous
dependency closure, utility targets, candidate evidence, exact AG geometry,
cache pipeline and unified metrics. Core representation and reasoning now use
the pinned official PTv3 source, PyTorch Transformer modules and PyG
TransformerConv instead of local attention blocks.

## Data flow and anti-leakage boundary

`SceneObservation → shared encoder → region/grasp/graph/action heads → candidate
encoder + verifier evidence → masked router → exact certification → execution →
new SceneObservation`.

Oracle camera values are permitted only inside dataset-generation artifacts.
`SceneObservation.validate()` rejects Oracle cameras. Instance IDs are equality
keys for pooling/association only. Ground-truth relation tensors are passed for
loss construction but graph message passing uses predicted edge probabilities;
sequence ordering is supervised only when `sequence_topology_valid=true`.

PICK_REMOVE destination and previous-action history are intentionally neutral in
Router inputs because deployment-path generated state caches do not contain
equivalent supervised features. Generated matching is open-world: candidates
matching both positive and negative teachers, plus all unmatched candidates,
remain UNKNOWN. A listwise policy row is effective only when it contains at
least one known positive and one known negative.
