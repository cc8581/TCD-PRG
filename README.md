# TCD-PRG

Task-Conditioned Dependency-Aware Push–Remove–Grasp (TCD-PRG) is a research
framework for functional-region manipulation in clutter. Given three Mech-Eye
PRO S observations, externally supplied instance masks, a target instance and a
requested functional region, it either executes a valid 6D `TASK_GRASP` or
repeatedly selects a safe `PICK_REMOVE`/fixed-distance `PUSH`, re-observes, and
re-plans for at most five preparation actions.

The formal action encoding is `0=PUSH`, `1=PICK_REMOVE`, `2=TASK_GRASP`.
`PUSH` is exactly 0.15 m in the main experiment. Gripper width is the predicted
AG-160-95 total opening in `[0, 0.095]` m; it is not the push distance.
`PICK_REMOVE` leaves an object physically present but marks it inactive.

## What is implemented

- A single expensive task-free scene geometry backbone pass, followed by a
  lightweight task-conditioning adapter for task-specific heads.
- Target-only functional-region segmentation and visibility prediction.
- Separate query-based task and global grasp heads. Each directly predicts an
  unordered set of complete `(translation, SO(3), width, quality)` grasps;
  Hungarian matching preserves multiple valid modes without pose averaging.
- Exact-URDF local scene–gripper overall-executability verifier.
- Predicted-edge heterogeneous dependency graph ending at a `TASK_GRASP` node.
- PICK_REMOVE reuses global complete grasps. PUSH predicts object, contact,
  direction and a direction-conditioned complete utility change that combines task gains with
  instability/workspace/failure costs. Hard feasibility remains certified.
- TASK_GRASP eligibility counts unique verifier/certifier survivors after
  per-object SE(3) grasp NMS, not raw candidate tensor slots.
- The generated data contains successful target self-push transitions. This is
  an explicit configurable recovery primitive (`allow_target_push_recovery`),
  while all other PUSH/PICK_REMOVE objects use the graph frontier plus the
  bounded recovery fallback.
- Candidate-only policy routing with multi-positive listwise supervision;
  UNKNOWN candidates never become negatives.
- Deterministic three-PRO-S state reconstruction, free-space-aware LRU cache,
  Windows-safe external PyBullet workers, and FR5/AG execution interfaces.
- AMP, accumulation, activation checkpointing, EMA, resume, freeze/unfreeze,
  two learning rates, DDP, deterministic seeds, validation, early stopping,
  JSONL/TensorBoard logging, and exact state-group progress counters.
- Unified offline metrics, grouped JSON/CSV exports and bootstrap confidence
  intervals; baseline/ablation configurations share one data and evaluation path.

## Repository layout

```text
tcd_prg/
  datasets/          unified contracts, capabilities, adapters and collation
  observation/       saved/rendered/cached observation providers
  rendering/         renderer contracts
  geometry/          SE(3) and exact AG-160-95 point geometry
  models/            backbone and all task heads
  losses/            independently masked multi-task losses
  trainers/          reproducible AMP/DDP trainer and EMA
  evaluators/        task/module metrics and exports
  planners/          candidate decoding and closed-loop H=5 policy
  execution/         deterministic certification/execution boundary
  baselines/         unified rules, one-shot and original GAPG wrapper
  scripts/           train/evaluate/prefetch/infer/replay entry points
  tools/             sample inspection and bounded dataset auditing
configs/             main, six ablations and eight baselines
scripts/             Python 3.8 PyBullet/GAPG workers and dependency setup
tests/               unit, contract, real-data and overfit tests
docs/                audit, data contract, reproduction and limitations
```

## Environment

The main training environment is native Windows Python 3.10. It intentionally
does not install PyBullet:

```powershell
conda env create -f environment.yml
conda activate tcd-prg
pip install -e .
```

Rendering, exact gripper sampling and action certification call an existing
PyBullet-capable Python environment. No PyBullet reinstall is performed. Set
`TCD_PYBULLET_PYTHON` to that interpreter; otherwise `python` on `PATH` is used.

External GraspNet source is pinned but not vendored:

```powershell
.\scripts\setup_third_party.ps1
```

The private repository deliberately excludes `.deps/`, GraspNet source,
datasets, caches and checkpoints.

## Data preparation

Place data under `data/`, or set `TCD_DATASET_ROOT`, `TCD_ACRONYM_ROOT` and
`TCD_FUNCTIONAL_REGION_ROOT`. No source edit is required. The training loop is
cache-only and never synchronously invokes PyBullet on the GPU path:

```powershell
tcd-prg-audit --config configs/config.yaml --states 100
tcd-prg-prefetch --config configs/config.yaml --max-groups 1000
tcd-prg-inspect --config configs/config.yaml --scene-id 0 --group-index 0
```

The prefetch command deterministically reconstructs intermediate observations
at configurable low resolution and samples them to `dataset.scene_points`.
Cache keys bind scene/state, poses, present/active masks, model IDs, scales,
camera profile, render seed, renderer version and point-sampling configuration.

Build task-unfiltered global grasp supervision and scene certification outside
the GPU loop (paths remain configurable and are not embedded in checkpoints):

```powershell
python tools/build_global_grasp_library.py `
  --acronym-root $env:TCD_ACRONYM_ROOT `
  --annotations $env:TCD_FUNCTIONAL_REGION_ROOT `
  --task-library "$env:TCD_DATASET_ROOT/task_training_labels_steps1_6_v1/grasp_library" `
  --output "$env:TCD_DATASET_ROOT/generic_grasp_library_v1"

python tools/certify_global_grasps.py --config configs/config.yaml --allow-render
```

Evaluate the global branch independently of the task graph and router:

```powershell
tcd-prg-eval-global --config configs/config.yaml --checkpoint outputs/full/last.pt `
  --scene-id 0 --state-id 0 --task-index 0 --output outputs/global/state_0.json
```

The global comparison has separate `scene_only` and `instance_assisted` tracks
and reports unified scene-executable proposal metrics before and after exact certification.
See [docs/global_grasp_protocol.md](docs/global_grasp_protocol.md).

## Training

```powershell
# One real batch forward/backward
tcd-prg-train --config configs/config.yaml --dry-run

# Full method
tcd-prg-train --config configs/config.yaml output_dir=outputs/full

# Resume
tcd-prg-train --config configs/config.yaml --resume outputs/full/last.pt

# Example ablation override
tcd-prg-train --config configs/config.yaml ablation.use_dependency_graph=false `
  output_dir=outputs/ablation_no_graph
```

Windows DDP uses `gloo` by default; Linux CUDA DDP uses `nccl`. The launchers
select one process per GPU and retain all runtime files inside this repository:

```powershell
.\scripts\train_ddp.ps1 -Gpus 2
```

See `docs/portable_training.md` for Linux and environment-variable examples.

Training units are `(scene_id, state_id, task_index, action_state_group)`, not
uniformly sampled action rows. Logs include optimizer steps, samples/states/
candidate groups seen and effective epochs. `loss_routing.json` records losses
automatically disabled by dataset capabilities or ablations.

The paper-level objective has exactly eleven modules: region, task grasp,
global grasp, physical edge, task edge, verifier overall, four PUSH objectives,
and policy candidate. Pose matching diagnostics remain internal to the two
grasp-set objectives and are not independently weighted task families.

## Evaluation and inference

```powershell
tcd-prg-eval --config configs/config.yaml --checkpoint outputs/full/best.pt `
  --split test --output-dir outputs/evaluation/full

tcd-prg-infer --config configs/config.yaml --checkpoint outputs/full/best.pt `
  --scene-id 0 --state-id 0 --task-index 0

tcd-prg-replay --config configs/config.yaml --scene-id 0 --task-index 0
```

Formal inference applies learned ranking first and then exact collision, FR5 IK
and approach-path certification. A rejection is masked and the next ranked
candidate is tried without rerunning the scene backbone. Actual robot transport
and safe placement remain behind the `RobotClient` interface.

## Reproducibility and documentation

- [Architecture and GAPG audit](docs/audit_and_architecture.md)
- [Unified data and coordinate contract](docs/data_contract.md)
- [Full reproduction procedure](docs/reproduction.md)
- [Checkpoint and experiment format](docs/checkpoints_and_outputs.md)
- [External pretrained checkpoint provenance](docs/pretrained_checkpoints.md)
- [Known limitations](docs/known_limitations.md)

Every run saves the resolved configuration, Git commit, framework version,
checkpoint RNG state and auditable per-task outputs. See `third_party.lock.yaml`
for exact external revisions.
