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

- A single official Pointcept Point Transformer V3 task-free scene pass,
  followed by a lightweight task-conditioning adapter for task-specific heads.
- Target-only functional-region segmentation and visibility prediction.
- Separate task/global query banks over one shared M2T2-style PyTorch decoder. Each predicts an
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
training.py          canonical formal training launcher
configs/             main, seven ablations and seven baselines
scripts/             DDP, external PyBullet workers and dependency setup
tools/               offline data preparation and profiling utilities
tests/               unit, contract, real-data and overfit tests
docs/                audit, data contract, reproduction and limitations
assets/robots/       FR5/AG-160-95 URDF, collision meshes and provenance
benchmarks/          small versioned fixtures for reproducible profiling
third_party/         pinned PTv3 and the minimal GAPG comparison subset
patches/             explicit compatibility patches for external dependencies
```

## Environment

The main training environment is native Windows Python 3.10. It intentionally
does not install PyBullet:

```powershell
conda env create -f environment.yml
conda activate tcd-prg
git submodule update --init third_party/PointTransformerV3
pip install -e .
```

Windows defaults to the official PTv3 non-Flash path. The environment installs
the CUDA 11.8 `spconv` and PyG wheels used by PTv3 and the relation-aware graph.
On a Linux CUDA-12 server, install the matching `spconv-cu12x` package and
FlashAttention, then set `backbone.enable_flash_attention=true`.

Rendering, exact gripper sampling and action certification call an existing
PyBullet-capable Python environment. No PyBullet reinstall is performed. Its
interpreter is configured in the machine-local path file described below.

External GraspNet source is pinned but not vendored:

```powershell
.\scripts\setup_third_party.ps1
```

The private repository deliberately excludes `.deps/`, GraspNet source,
datasets, caches and checkpoints.

## Data preparation

Copy `configs/local_paths.example.yaml` to `configs/local_paths.yaml` once and
set this machine's dataset and PyBullet paths there. The local file is ignored
by Git; source code and shared configuration contain no machine-specific path.
Command-line `--dataset-root`, `--acronym-root`, `--functional-region-root` and
`--pybullet-python` values take precedence. The training loop is cache-only and
never synchronously invokes PyBullet on the GPU path:

```powershell
python training.py
```

The launcher supplies the Windows RTX 3090 defaults, creates a timestamped
output directory, and accepts ordinary `key=value` overrides. Its default is
one optimizer update per batch (no gradient accumulation). Use `--resume`,
`--initialize`, or `--gpus N` when required; it always starts formal training
and has no dry-run mode.

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
$DatasetRoot = "E:\datasets\TaskOrientedClutterSceneDataset"
$AcronymRoot = "E:\datasets\ACRONYM"
$FunctionalRegionRoot = "E:\datasets\manual_function_regions_v1"
python tools/build_global_grasp_library.py `
  --acronym-root $AcronymRoot `
  --annotations $FunctionalRegionRoot `
  --task-library "$DatasetRoot/task_training_labels_steps1_6_v1/grasp_library" `
  --output "$DatasetRoot/generic_grasp_library_v1"

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
# Formal cache-only training (stop the process after startup verification when testing)
tcd-prg-train --config configs/config.yaml

# Full method
tcd-prg-train --config configs/config.yaml output_dir=outputs/full

# Resume
tcd-prg-train --config configs/config.yaml --resume outputs/full/last.pt

# Freeze the trained geometry/action stack, generate deployment-path candidates
tcd-prg-generate-policy-candidates --config configs/config.yaml `
  --checkpoint outputs/full/best.pt --split train `
  --output-dir runtime/cache/policy_candidates
tcd-prg-generate-policy-candidates --config configs/config.yaml `
  --checkpoint outputs/full/best.pt --split val `
  --output-dir runtime/cache/policy_candidates

# Train the router on generated candidates, then optionally mix clean teachers
tcd-prg-train --config configs/stage/policy_generated.yaml `
  --initialize outputs/full/best.pt output_dir=outputs/policy_generated
tcd-prg-train --config configs/stage/policy_mixed.yaml `
  --initialize outputs/full/best.pt output_dir=outputs/policy_mixed

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

The terminal prints a concise live summary at `logging.log_interval`: total and
paper-level module losses, learning rate, gradient norm, throughput, AMP skips,
effective epoch, and generated-candidate coverage when applicable. Detailed
metrics are not downsampled: every successful optimizer step is appended to
`train_metrics.jsonl`, including all loss diagnostics averaged across the full
gradient-accumulation window. `validation_metrics.jsonl` stores every validation
score plus all averaged validation terms, while `training_events.jsonl` records
start/end, AMP-skipped windows, checkpoints, validation, and early stopping.

Generated policy caches are tied to the upstream checkpoint SHA-256 and the
full candidate code/config/asset signature. Matching is tri-state: candidates
near successful sequence actions are positive, candidates near explicitly
evaluated unsuccessful actions are negative, and unmatched candidates remain
UNKNOWN and are excluded from policy loss. A candidate matching both a positive
and negative teacher is also UNKNOWN. Cache manifests report state-level positive
coverage and positive-plus-negative effective rows; pure generated training fails
fast below the configured coverage thresholds. Generate both train and validation
splits into the same cache directory before starting a generated-candidate stage.

Generated caches deliberately do not run robot approach certification. Learned
Verifier and soft graph scores are Router evidence; only physical inactivity and
malformed actions are hard masks by default. The pure generated stage skips the
frozen geometry/action heads and executes only the shared encoder, cached candidate
encoder/evidence, and Router.

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

Formal inference applies learned ranking first. Grasp candidates then receive
exact collision, FR5 IK and approach-path certification; a rejection is masked
and the next ranked candidate is tried without rerunning the scene backbone.
PUSH approach/path selection and PICK_REMOVE transport/safe placement remain
executor-owned engineering operations behind the `RobotClient` interface.

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
