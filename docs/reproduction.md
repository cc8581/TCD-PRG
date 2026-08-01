# Reproduction procedure

## 1. Paths and environment

Create the Python 3.10 environment from `environment.yml`; do not install
PyBullet again on the current workstation. Verify that the configured Python
3.8 executable can import `torch`, `pybullet`, `trimesh` and `open3d`. Adjust
dataset, functional-region, ACRONYM and FR5/AG paths in `configs/config.yaml`.

Run `scripts/setup_third_party.ps1` (Windows) or `.sh` (Linux) to clone the
exact GraspNet commits. The setup is idempotent and verifies upstream remotes,
detached revisions and the Windows int64 patch. Supply external checkpoints;
they are never committed.

## 2. Audit and cache

Run the bounded 100-state audit. It checks action encodings, fixed PUSH distance,
UNKNOWN semantics, camera leakage, unified shapes and dataset capabilities. Run
the prefetcher before training. Rendering is deterministic and separated from
the GPU loop. Prewarm exact AG geometry for all observed valid widths.

## 3. Smoke gates

Set `TCD_DATASET_ROOT` and run `pytest -q`. Before a large job run:

1. one real batch forward;
2. one real batch backward (`--dry-run`);
3. ten-batch overfit test;
4. one cached state inference;
5. one labelled multi-step replay;
6. the 100-state audit.

Do not proceed if any validity-mask, coordinate, quaternion, action encoding,
cache or deterministic-seed test fails.

## 4. Main model and ablations

Use the same data snapshot, train/val/test scene split, seed, candidate sampler,
motion constraints and evaluator for every experiment. Override only the
documented switch. The six supported comparisons are task region off,
dependency graph off, indirect reasoning off, verifier off, PUSH potential
off, and hierarchical router versus fixed/flat routing.

## 5. Evaluation

Evaluate the full method and all baselines with identical H=5, sensor input,
target condition, test scenes, execution constraints and seeds. Reports include
per-category, per-region, per-sequence-length and per-occlusion groups plus
bootstrap intervals. UNKNOWN selections reduce evaluable coverage but are not
counted as failures. Report coverage alongside task success.

## 6. Repeatability record

Archive `resolved_config.yaml`, `run_metadata.json`, `loss_routing.json`, JSONL
logs, checkpoint, `metrics.json`, `per_task.csv`, dependency lock and Git commit.
For multi-seed results, keep one output directory per seed and aggregate means
and standard deviations without overwriting the per-seed reports.
