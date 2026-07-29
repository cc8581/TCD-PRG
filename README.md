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

- A single task-conditioned point transformer pass shared by all learned heads.
- Target-only functional-region segmentation and visibility prediction.
- Task-region grasp and generic removal-grasp proposal heads.
- Exact-URDF local scene–gripper multi-head verifier.
- Predicted-edge heterogeneous dependency graph ending at a `TASK_GRASP` node.
- PICK_REMOVE object/candidate/outcome and PUSH object/contact/direction/
  approach/outcome/potential/risk prediction.
- Masked hierarchical, flat, and fixed-priority routing with multi-positive
  listwise supervision; UNKNOWN candidates never become negatives.
- Deterministic three-PRO-S state reconstruction, content-addressed LRU cache,
  Windows-safe external PyBullet workers, and exact FR5/AG certification.
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

Rendering, exact gripper sampling and action certification call the existing
`D:\Anaconda\envs\gapg\python.exe` environment. No PyBullet reinstall is
performed. On another machine, change `observation.pybullet_python` in YAML to
an existing compatible environment.

External GraspNet source is pinned but not vendored:

```powershell
.\scripts\setup_third_party.ps1
```

The private repository deliberately excludes `.deps/`, GraspNet source,
datasets, caches and checkpoints.

## Data preparation

Set the paths in `configs/config.yaml`. The training loop is cache-only and
never synchronously invokes PyBullet on the GPU path:

```powershell
tcd-prg-audit --config configs/config.yaml --states 100
tcd-prg-prefetch --config configs/config.yaml --max-groups 1000
tcd-prg-inspect --config configs/config.yaml --scene-id 0 --group-index 0
```

The prefetch command deterministically reconstructs intermediate observations
at configurable low resolution and samples them to `dataset.scene_points`.
Cache keys bind scene/state, poses, present/active masks, model IDs, scales,
camera profile, render seed, renderer version and point-sampling configuration.

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

Windows DDP uses `gloo` by default; Linux CUDA DDP uses `nccl`:

```powershell
torchrun --standalone --nproc_per_node=2 -m tcd_prg.scripts.train `
  --config configs/config.yaml
```

Training units are `(scene_id, state_id, task_index, action_state_group)`, not
uniformly sampled action rows. Logs include optimizer steps, samples/states/
candidate groups seen and effective epochs. `loss_routing.json` records losses
automatically disabled by dataset capabilities or ablations.

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
- [Known limitations](docs/known_limitations.md)

Every run saves the resolved configuration, Git commit, framework version,
checkpoint RNG state and auditable per-task outputs. See `third_party.lock.yaml`
for exact external revisions.
