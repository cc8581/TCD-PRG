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
| Task grasp set | external GraspNet/DETR concepts | `models/grasp_proposal/head.py` | shared points + target/task/region | K complete poses, openings and qualities | geodesic Hungarian matching + smooth symmetric chordal training loss | TASK_GRASP candidates |
| Global grasp set | GraspNet/DETR concepts | same module, neutral scene features | task-free scene points | K complete poses, openings, object assignments and scene qualities | object-balanced scene-certified Hungarian set loss | PICK_REMOVE candidates |
| Grasp verifier | `Space_GraspFusion` concept | `models/grasp_verifier/` | local scene + exact AG cloud + task context | overall executability | final-certifier BCE | learned filtering/ranking |
| Dependency graph | none | `models/dependency_graph/hgt.py` | object/task tokens and predicted edges | physical/task edges and derived masks | two edge losses | dependency context/frontier |
| PUSH | `Push_model` concept | `models/push/head.py` | point/object/task/graph/geometry/steps | object, contact, direction, per-direction utility delta | four module objectives | fixed 0.15 m action; hard feasibility is certified |
| Router | fixed grasp→push rule | `models/policy/router.py` | heterogeneous candidates + evidence | candidate score | multi-positive listwise | legal candidate selection |
| Closed loop | `grasp_push_eval.py` loop | `planners/closed_loop.py` | policy, observations, executor | H=5 result and trace | evaluated through replay/simulation | reobserve and replan |
| Exact safety | PyBullet execution environment | `execution/pybullet_certifier.py` + worker | candidate + full state + FR5/AG URDF | valid/reason | deterministic, not learned | final mask before execution |

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
remains bounded by `graph_candidate_fallback_objects`.

The final objective exposes eleven paper-level modules. Translation, symmetric
SO(3), width, quality and object assignment remain internal diagnostics of each grasp
set loss rather than separate weighted tasks.

## Reuse, extension and replacement

Directly reused for the GAPG baseline: original grasp/push network definitions,
preprocessing utilities and their checkpoints. Reused conceptually in the full
method: complete continuous grasp-set prediction and local gripper–scene
evaluation. Extended: three-view input wrapper, task conditioning, shared
features and closed-loop evaluation. Newly implemented: the unified adapter,
functional region head, heterogeneous task graph, utility-aware PUSH,
candidate router, exact AG geometry, cache pipeline, capability-aware
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
