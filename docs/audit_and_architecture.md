# GAPG audit and TCD-PRG implementation map

## Audited GAPG flow

The inherited GAPG code uses `generate_grasp.py` to obtain GraspNet proposals,
`grasp_evaluate.py` and `models/grasp_networks.py::Space_GraspFusion` to score
local scene–gripper geometry, and `push_evaluate.py` plus
`models/push_networks.py::Push_model` to rank sampled push poses. Its top-level
`grasp_push_eval.py` implements grasp-first, otherwise push, followed by a new
observation. The original method has no functional-region task token,
PICK_REMOVE action, explicit dependency graph, multi-positive hierarchical
router or state-dependent multi-step supervision.

TCD-PRG does not modify these original modules for comparison. The external
wrapper in `tcd_prg/baselines/gapg_wrapper.py` converts the same three-PRO-S
fusion and target instance mask to GAPG input and invokes the pinned source in a
Python 3.8 worker.

## Detailed mapping

| Design module | GAPG source reused | TCD-PRG implementation | Main input | Main output | Supervision/loss | Inference role |
|---|---|---|---|---|---|---|
| Shared encoder | PointNet++ concepts | `models/backbones/task_point_transformer.py` | `[B,N,3]` XYZ/RGB/masks/task | point/object/target/global/task tokens | joint downstream gradients | exactly one scene pass |
| Functional region | none | `models/region/head.py` | target point features + task | point probability, visibility | focal BCE + Dice + visibility | constrains task grasp |
| Task grasp proposal | external GraspNet concepts | `models/grasp_proposal/head.py` | shared points + target/task/region | canonical contact frame, approach, rotation, AG total opening, confidence, compatibility | masked classification/regression | dense TASK_GRASP candidates |
| Generic removal grasp | GraspNet concepts | same head under generic condition | active object points | removal grasp fields | PICK_REMOVE grasp labels | candidate proposals |
| Grasp verifier | `Space_GraspFusion` concept | `models/grasp_verifier/` | local scene + exact AG cloud + task context | six validity heads | explicit per-head valid masks | learned filtering/ranking |
| Dependency graph | none | `models/dependency_graph/hgt.py` | object/task tokens and predicted edges | physical/task edges, blockers, topology | edge/blocker/order losses | direct/indirect dependency context |
| PICK_REMOVE | none | `models/pick_remove/head.py` | object/graph/candidate tokens | object pointer, removal-grasp rank | masked pointer/listwise | reliable removal macro selection |
| PUSH | `Push_model` concept | `models/push/head.py` | point/object/task/graph/geometry/steps | object, contact, direction, low-weight potential and risks | independently masked heads | fixed 0.15 m action; approach is execution-layer geometry |
| Router | fixed grasp→push rule | `models/policy/router.py` | heterogeneous candidates + evidence | type/object/candidate scores, steps | multi-positive listwise | legal hierarchical selection |
| Closed loop | `grasp_push_eval.py` loop | `planners/closed_loop.py` | policy, observations, executor | H=5 result and trace | evaluated through replay/simulation | reobserve and replan |
| Exact safety | PyBullet execution environment | `execution/pybullet_certifier.py` + worker | candidate + full state + FR5/AG URDF | valid/reason | deterministic, not learned | final mask before execution |

## Reuse, extension and replacement

Directly reused for the GAPG baseline: original grasp/push network definitions,
preprocessing utilities and their checkpoints. Reused conceptually in the full
method: GraspNet-style discrete grasp parameterization and local gripper–scene
evaluation. Extended: three-view input wrapper, task conditioning, shared
features and closed-loop evaluation. Newly implemented: the unified adapter,
functional region head, heterogeneous task graph, PICK_REMOVE, multi-head PUSH,
hierarchical set router, exact AG geometry, cache pipeline, capability-aware
losses and unified metrics. Replaced in the full method: GAPG's fixed action rule
and per-candidate scene feature recomputation.

## Data flow and anti-leakage boundary

`SceneObservation → shared encoder → region/grasp/graph/action heads → candidate
encoder + verifier evidence → masked router → exact certification → execution →
new SceneObservation`.

Oracle camera values are permitted only inside dataset-generation artifacts.
`SceneObservation.validate()` rejects Oracle cameras. Instance IDs are equality
keys for pooling/association only. Ground-truth relation tensors are passed for
loss construction but graph message passing uses predicted edge probabilities;
sequence ordering is supervised only when `sequence_topology_valid=true`.
