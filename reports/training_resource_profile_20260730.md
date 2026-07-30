# TCD-PRG training resource profile (2026-07-30)

## Measurement scope

- GPU: NVIDIA GeForce RTX 3090 24 GB
- PyTorch: 2.2.0+cu121 on native Windows
- Main training input: batch size 1, 16,384 scene points, 4,096 target/attention points
- Effective batch: gradient accumulation 8
- AMP: float16, stable measured scale 4,096
- Activation checkpointing: enabled
- Verifier candidate micro-batch: 16
- Real unit: `(scene_id=0, state_id=0, task_index=0, group_index=0)` with 72 candidates
- Rendering is offline and excluded from steady-state GPU training time

## Parameters

| Module | Parameters | Share |
|---|---:|---:|
| dependency graph | 4,308,748 | 22.36% |
| hierarchical router | 3,684,872 | 19.12% |
| shared encoder | 3,505,920 | 18.19% |
| flat-router ablation head | 2,501,122 | 12.98% |
| PICK_REMOVE | 1,250,825 | 6.49% |
| PUSH | 864,549 | 4.49% |
| grasp verifier | 858,630 | 4.46% |
| candidate encoder | 799,488 | 4.15% |
| task grasp proposal | 531,479 | 2.76% |
| generic removal grasp proposal | 531,479 | 2.76% |
| task-region head | 361,474 | 1.88% |
| candidate evidence | 71,680 | 0.37% |
| **Instantiated total** | **19,270,266** | **100%** |

All 19.27 M instantiated parameters have `requires_grad=True`. The full
hierarchical method does not execute the flat-router ablation head, so the
gradient-bearing full-method parameter count is **16,769,144 (87.02%)** across
305 parameter tensors. The remaining 2,501,122 parameters are retained only so
the flat-router ablation can use the same model/configuration implementation.

Storage:

- complete FP32 model state: 73.51 MiB
- EMA state: 73.51 MiB
- populated AdamW state for active parameters: 127.94 MiB
- `last.pt`: 275.45 MiB
- interval checkpoint: 275.47 MiB

## Computation

PyTorch operator-dispatch FLOP counter results for one full-resolution
micro-batch:

- forward and loss: **258.65 GFLOPs**
- backward, including activation-checkpoint recomputation: **610.63 GFLOPs**
- forward + backward: **869.28 GFLOPs**
- approximate forward MAC convention: **129.33 GMACs**
- one optimizer step with 8 accumulated micro-batches: **at least 6.954 TFLOPs**

These are conservative lower bounds. The counter covers matrix multiplication,
attention and other registered PyTorch operators, but does not invent costs for
unsupported indexing, sampling, elementwise, loss and optimizer operations.

## Timing

Warm, successful full-resolution micro-batch at AMP scale 4,096:

| Stage | Time |
|---|---:|
| forward + 45 loss terms | 248.59 ms |
| backward | 116.54 ms |
| unscale + clip + AdamW | 7.17 ms |
| micro-batch including update, median | 373.10 ms |
| CPU-to-GPU transfer | 2.40 ms |

Across 10 successful repetitions, micro-batch CUDA time was 365.19 ms minimum,
373.10 ms median and 422.77 ms P95. With accumulation 8, the measured steady
GPU-compute estimate is **2.928 s per optimizer step**, or **2.732 state
groups/s**. The first scale-8,192 attempt was
correctly rejected by GradScaler; after falling to 4,096, the measured loss,
gradient norm and update were finite.

Single-worker cold input measurements were 362.2 ms for sample/HDF5/cache load
and 430.6 ms for candidate/gripper collation. The configured four persistent
workers can overlap most of this work, but real end-to-end throughput must be
confirmed on a long run because storage contention, validation and checkpoint
writes are not included in the 2.883-second GPU estimate.

Model, optimizer and checkpoint construction/loading took 3.07 s. The complete
profiling process, including 10 timing repetitions and FLOP replay, took 9.62 s. Rendering one previously
uncached 16,384-point observation took approximately 3.06 s and must remain an
offline prefetch operation.

## Memory

- baseline allocated GPU memory after model, optimizer, checkpoint and batch load: 204.80 MiB
- peak allocated GPU memory during training: **1,183.68 MiB (1.156 GiB)**
- peak reserved CUDA memory: **1,380 MiB (1.348 GiB)**
- incremental training peak above baseline: 978.88 MiB
- profiler process peak working set: **1.864 GiB**
- profiler process private memory: **4.079 GiB**

This is single-GPU, batch-1 training. It excludes PyBullet renderer processes,
FR5 certification, DataLoader worker memory and DDP replication. The measured
model itself is comfortably within the RTX 3090 24 GB capacity; data preparation
and offline observation cache capacity are the larger system-level concerns.

## Dataset-dependent duration

The action generator was still running during the snapshot. The immutable-file
snapshot contained:

- 2,480 completed training scenes
- 275,286 action-state groups
- 34,411 optimizer steps per current effective epoch at accumulation 8

At measured steady GPU throughput, one current-snapshot effective epoch is a
**27.99 GPU-hour lower bound**. The configured 100,000 optimizer steps are an
**81.34 GPU-hour / 3.39-day compute lower bound**, processing 800,000 state
groups or about 2.91 current-snapshot epochs.

The source split is 8,000 train / 1,000 validation / 1,000 test scenes. If the
current mean of 111.0 groups per scene remains stable, the final train split is
projected to contain about 888,019 groups. Under that projection:

- one final effective epoch: approximately **90.29 GPU hours (3.76 days)**
- 100,000 steps: approximately **0.90 effective epoch**

The projection is not a final dataset statistic. Recompute it after generation
finishes before selecting the paper-training step budget.

## Artifacts

- `training_resource_profile_full_20260730.json`: exact machine-readable profile
- `training_resource_profile_full_20260730.csv`: per-module parameter table
- `training_resource_profile_full_20260730.console.txt`: raw profiler console
- `training_resource_profile_smoke_20260730.json`: 2,048-point smoke profile
- `dataset_group_snapshot_20260730.json`: immutable completed-file snapshot
- `profile_smoke_training.py`: reproducible local profiler
