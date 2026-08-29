# TCD-PRG

Task-Conditioned Push-Remove-Grasp (TCD-PRG) is a closed-loop manipulation
framework for cluttered scenes. The formal model learns three responsibilities:
sensor-only perception/target-region understanding, grasp evaluation, and
task-conditioned PUSH prediction. Action arbitration is deterministic rather than
a separately learned policy network.

The formal action encoding is `0=PUSH`, `1=PICK_REMOVE`, `2=TASK_GRASP`.
`PUSH` remains a fixed 0.15 m primitive. `TASK_GRASP` and `PICK_REMOVE` grasps
are ranked by the learned grasp branch and checked by the exact FR5/AG-160-95
execution certifier before execution. Every executed preparation action is followed
by a fresh observation and replanning.

## What is implemented

- One shared Point Transformer V3 perception pass with predicted instances, target
  identity and task-region segmentation.
- Frozen official GraspNet proposal generation, task-conditioned grasp scoring and
  AG-160-95 width adaptation.
- A task-conditioned PUSH hierarchy: object -> contact -> direction bin/residual ->
  utility. Target-relative and task-region-relative geometry are explicit inputs.
- Training-only sparse PUSH direction supervision that unions predicted top-k
  contacts with evaluated GT contacts; GT geometry never enters deployment forward.
- `PICK_REMOVE` candidates are generic grasps on non-target predicted instances,
  restricted to a lightweight target-local geometric neighborhood instead of a
  learned dependency graph.
- Fixed closed-loop action priority `TASK_GRASP > PICK_REMOVE > PUSH`; Stage B uses
  an independent binary TaskGraspEvaluator, while no learned router is used.
- Exact grasp IK/collision/approach certification remains at the execution boundary.
  PUSH trajectory planning remains executor-owned.
- Three independent training stages so PUSH training does not execute GraspNet or
  the task-grasp scorer.

## Repository layout

```text
tcd_prg/
  datasets/          unified contracts, capabilities, adapters and collation
  observation/       saved/rendered/cached observation providers
  rendering/         renderer contracts
  geometry/          SE(3), grasp NMS and frame-safe geometry
  models/            backbone and all task heads
  losses/            independently masked multi-task losses
  trainers/          reproducible AMP/DDP trainer and EMA
  evaluators/        task/module metrics and exports
  planners/          candidate decoding and closed-loop H=5 policy
  execution/         deterministic certification/execution boundary
  baselines/         unified rules, one-shot and original GAPG wrapper
  scripts/           train/evaluate/prefetch/infer/replay entry points
  tools/             sample inspection and bounded dataset auditing
train.py             canonical formal training launcher
configs/             main, staged training, retained ablations and baselines
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
the CUDA 11.8 `spconv` and PyG wheels used by PTv3.
On a Linux CUDA-12 server, install the matching `spconv-cu12x` package and
FlashAttention, then set `backbone.enable_flash_attention=true`.

Rendering and exact action certification call an existing
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
`--pybullet-python` values take precedence. Training uses a bounded read-through
cache; DataLoader workers render cache misses through the external PyBullet
runtime while the GPU consumes already prepared batches:

```powershell
python train.py --stage perception
```

The launcher supplies the Windows RTX 3090 defaults, creates a timestamped
output directory, and accepts ordinary `key=value` overrides. Its default is
one optimizer update per batch (no gradient accumulation). Use `--resume`,
the Stage-C component checkpoint arguments, or `--gpus N` when required; it always starts formal training
and has no dry-run mode.

```powershell
tcd-prg-audit --config configs/config.yaml --states 100
tcd-prg-prefetch --config configs/config.yaml --split train --max-groups 1000
tcd-prg-inspect --config configs/config.yaml --scene-id 0 --group-index 0
```

Formal training uses a bounded read-through LRU cache: a cache miss is rendered
by the external PyBullet worker and then cached. The optional prefetch command
only warms an explicitly bounded hot set; full-dataset prefetch is intentionally
unsupported because the complete observation set is larger than the cache.
It deterministically reconstructs intermediate observations at configurable
low resolution. Formal PTv3 training sets `dataset.scene_points=0`, preserving
the complete variable-length three-view union at the configured 320x200 view
resolution. The collator applies the official Pointcept GridSample rule at
5 mm before padding, the backbone packs occupied voxels across the batch, and
the decoder/inverse map restores features for every retained supervision point
used by Region, Push, and grasp-anchor heads. Raw multi-megapixel camera pixels
are not padded or sent to the GPU. A positive `scene_points` remains available
only as an explicit pre-grid hardware safety cap.
Cache keys bind scene/state, poses, present/active masks, model IDs, scales,
camera profile, render seed, renderer version and point-sampling configuration.

Evaluate a GraspNet-format prediction dump with the official GraspNet evaluator:

```powershell
tcd-prg-eval-graspnet --graspnet-root D:\GraspNet `
  --dump-folder outputs\graspnet_dump --camera kinect --split all
```

The global branch is trained directly from the published per-object
`grasp_library` and the PICK_REMOVE poses and outcomes in the action HDF5. Its
labels are open-world: only executed attempts are positive or negative, while
unexecuted and conflicting outcomes remain UNKNOWN. These sparse native labels
are not used to report proxy AP or recall; public Global Grasp comparison is
delegated to the official GraspNet protocol.

## Training

Stage A, Stage B and Stage C train independently behind public condition
contracts. Stage C uses GT-derived `PushCondition` during standalone training;
deployment composes the three checkpoints without copying Stage A/B weights into
the Stage C training checkpoint.

```powershell
# Stage A: perception / instance / target region
tcd-prg-train --config configs/stage/perception.yaml `
  output_dir=outputs/perception

# One-time Stage B-D expansion: GT condition -> frozen GraspNet -> binary geometry labels
tcd-prg-build-stageb --config configs/stage/grasp.yaml `
  --output runtime/stageb_binary --split train
tcd-prg-build-stageb --config configs/stage/grasp.yaml `
  --output runtime/stageb_binary --split val

# Stage B-T: independent binary evaluator
tcd-prg-train --config configs/stage/grasp.yaml `
  output_dir=outputs/grasp

# Stage C: independent PUSH predictor with GT-derived PushCondition
tcd-prg-train --config configs/stage/push.yaml `
  output_dir=outputs/push
```

The Stage-C checkpoint contains only the standalone PUSH module. `--resume` is
for continuing the same stage. Deployment requires compatible Stage A, Stage B
and Stage C checkpoints and checks their runtime semantics before composition.

To start a new run from every parameter in a same-stage checkpoint without
restoring its optimizer, scheduler, AMP scaler, RNG state, best metric or step,
use the pretrain-checkpoint mode:

```powershell
python train.py --stage perception `
  --pretrain-checkpoint outputs\old_perception\best.pt `
  --output-dir outputs\perception_rgb_augmented
```

The same option and `training.pretrain_checkpoint` configuration field are
supported by Stage A, Stage B and Stage C. The standalone
PUSH evaluator accepts the same option through `train_push_evaluator.py`.
Checkpoint stage, schema, parameter names and tensor shapes are checked strictly.

Formal stage configurations inherit the training-only `augmentation`
pipeline from `configs/config.yaml`. Each RGB transform has its own nested
enable flag, probability and intensity range. The pipeline can keep RGB,
zero it, convert it to three-channel grayscale, jitter color, randomly recolor
instances, perturb material/lighting, add sensor noise, or drop channels/points.
It never changes XYZ, point count, labels or point correspondence, and validation
and inference bypass it entirely.

Stage B trains only on concrete GraspNet proposals stored with strict 0/1
`task_valid` labels. Invalid data-generation attempts are dropped and never become
a third class. Each record freezes the sampled scene context, GT condition, pose
and proposal width used during label generation. Labels use dense AG CAD geometry,
and the manifest binds them to dataset, GraspNet, transfer, configuration, geometry
and code hashes. It contains no Stage-A checkpoint identity.
Both train and validation splits are mandatory for formal Stage B. Validation
aggregates candidate-level TP/FP/FN, AUROC and AUPRC over the complete split and
writes the best-F1 decision threshold to the run output. Other stages retain their
native action-state units.

## Evaluation and inference

```powershell
tcd-prg-eval --config configs/config.yaml --checkpoint outputs/perception/best.pt `
  --split test --output-dir outputs/evaluation/perception

tcd-prg-infer --config configs/config.yaml `
  --stage-a-checkpoint outputs/perception/best.pt `
  --stage-b-checkpoint outputs/grasp/best.pt `
  --stage-c-checkpoint outputs/push/best.pt `
  --scene-id 0 --state-id 0 --task-index 0

tcd-prg-replay --config configs/config.yaml --scene-id 0 --task-index 0
```

Formal inference ranks learned grasp/PUSH candidates, applies fixed action-type
priority, and exact-certifies grasp candidates. A rejected grasp is masked and the
next eligible candidate is tried without rerunning the scene backbone.
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
