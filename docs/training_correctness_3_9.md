# TCD-PRG training-correctness patch: items 3–9

Base commit: `9f2da2526086ba6af25318a7e5872f706ddb71f8` (`main`, `standardize evaluation protocols`).

## Compatibility boundary

This patch preserves the repository's **standard-only public evaluation surface** introduced by the base commit. Oracle hits, quality-ranked pose hits and pose component errors are **training diagnostics only** and are written to a separate file:

`validation_grasp_diagnostics.jsonl`

Per-sample component diagnostics are additionally written to
`validation_grasp_diagnostic_samples.jsonl` on single-GPU runs.  DDP uses
`validation_grasp_diagnostic_samples.rankNNN.jsonl` so ranks never contend for
one large file or all-gather every sample record.

They are not added to `OfflineModelEvaluator`, `metrics.json`, the paper exporter or any standard-comparison table.

Because the base commit intentionally removed `_add_grasp_metrics()` from `OfflineModelEvaluator`, the requested GT self-consistency test is attached to the new internal pose-diagnostic implementation instead. That is now the only native TCD code path that performs pose-neighbourhood matching. The official Global Grasp result remains delegated to `graspnetAPI` and is not replaced by these diagnostics.

## Implemented items

### 3. Geometry vs quality-ranking diagnostics

For Task and Global Grasp validation rows with positive labels, the patch records:

- `task/global_oracle_hit_at_{1,5,10,64}`: raw queries ranked only by normalized nearest-GT geometric error, never by quality.
- `task/global_quality_hit_at_{1,5,10,64}`: quality-ranked predictions after diagnostic NMS. These are deliberately named `quality_hit`, not `known_hit`, to avoid reintroducing a removed public metric family.
- Top-1 translation, parallel-jaw-symmetric rotation and width errors.
- Global Top-1 object correctness.
- Best-of-query translation, rotation and width errors from the same best normalized-geometry prediction.
- The same values are retained per validation sample with `scene_id`, `state_id` and `task_index`, in addition to the validation-cycle means.

Interpretation:

- high `oracle_hit_at_64` + low `quality_hit_at_1` → pose set exists, quality ranking is the likely bottleneck;
- low `oracle_hit_at_64` → the predicted pose set itself has not reached the complete matching tolerances;
- component errors identify translation/rotation/width failure directly;
- Global object correctness separates object assignment from pose error.

### 4. Independent Global Grasp scene-state stream

The primary action stream now uses `global_grasp_mode="never"`, so direct Global Grasp loss is no longer injected according to action-group sampling frequency.

A new `GlobalStateDataset`:

1. deduplicates training units by `(scene_id, state_id)`;
2. retains only states with at least one certified positive or negative PICK_REMOVE grasp;
3. loads Global labels explicitly for every selected state.

To avoid one HDF5 open per unique state, the adapter builds a persistent
`global_grasp_states_*.npz` index by scanning each scene once. Certification uses
the same conflict semantics as `_pick_remove_grasp_records`: an `(object, source)`
grasp is known only when its executed outcomes are not contradictory.

Each action micro-batch is paired with one Global scene-state batch. The trainer executes:

1. action-stream forward/backward;
2. Global-only forward/backward using `forward_mode="global_grasp"`;
3. one shared optimizer step after gradient accumulation.

The Global stream weight defaults to `1.0`. It is not dynamically scaled from current loss magnitude.

### 5. Explicit-negative Hungarian acceptance

Hungarian assignment now proposes `(query, negative)` pairs only. A pair supervises `quality=0` only if translation, parallel-jaw rotation, width and (for Global) object assignment all pass the configured negative thresholds.

Rejected Hungarian pairs stay UNKNOWN and are not marked `matched_query=True`; they remain available to the subsequent all-negative threshold association.

Focused tests cover:

- 5 mm translation error under a 10 mm threshold → valid negative;
- 80 mm error under a 10 mm threshold → UNKNOWN;
- close Global geometry but wrong object → UNKNOWN.

### 6. Task/state-first action sampling with non-invasive coverage

Formal training no longer uses the old inverse-frequency
`torch.multinomial(... replacement=False, total_samples=len(dataset))` ordering,
and it also does **not** impose a fixed action-stratum quota such as 2/1/1/1/1.

The new `DistributedTaskStateBatchSampler` uses the hierarchy:

1. `(scene_id, task_index)` — a target-object/task-region task is sampled
   approximately uniformly;
2. unique `(scene_id, task_index, state_id)` — decision states are shuffled and
   cycled inside each selected task;
3. one complete `action_state_group` belonging to that already selected state is
   loaded as the actual dataset sample.

Therefore, a task that has many alternative successful push/remove/grasp paths
does not automatically receive more weight merely because it generated more
states/groups. A sampler epoch is sized from the number of **unique decision
states**, not the number of raw action groups.

`training.action_batch_coverage_strata` is only a best-effort secondary rule. If
a selected state owns multiple action groups, group selection may prefer one
whose stratum is currently absent from the local mini-batch. It is forbidden to
replace the selected task or state to satisfy coverage, so the coverage heuristic
cannot distort the task/state schedule. If a state has only one group, that group
is used unchanged.

This does not turn trajectories or individual action rows into samples. Policy
supervision remains state-conditioned and multi-positive: `collate_unified()`
still unions action IDs from **all successful sequences** for the task, then marks
all candidates in the selected action-state group that belong to any successful
path as positive. `HierarchicalSetPolicyLoss` therefore continues to rank the
complete known candidate set for the current decision state.

DDP first constructs one deterministic duplicate-free global task/state batch and
then slices it across ranks. Stratum coverage is applied only after that slice,
using a rank-local RNG, so it cannot make ranks disagree about the global
task/state schedule.

### 7. Representative validation

Formal configuration now defaults to:

`training.max_validation_groups: null`

so the full validation split is used.

If a bounded subset is configured, `ActionStateGroupDataset` builds a deterministic scene-diverse, stratum-quota subset without replacement. The exact selected groups are persisted to:

`<output_dir>/validation_subset.json`

On resume, the manifest is loaded and validated against the current seed/quota/source groups instead of silently choosing a new prefix/subset.

The default 256-group quota template is retained in config for optional bounded validation:

- direct_grasp: 64
- pick_remove: 64
- push: 48
- push_failure: 48
- unresolved_or_unknown: 32

If `max_validation_groups` is set to another value, update the quota so the sum matches exactly.

### 8. Matching thresholds and NMS thresholds are separate

Internal diagnostic GT matching uses independent evaluation fields:

- task translation / rotation / width thresholds;
- global translation / rotation / width thresholds.

Diagnostic NMS uses a second independent set of task/global NMS thresholds. Initial values remain numerically equal to the previous tolerances, but the parameters no longer share meaning or storage.

Global NMS suppresses only same-object predictions. Rotation uses the repository's parallel-jaw-symmetric distance.

The official GraspNet evaluator is unchanged and continues to use its own accepted protocol/NMS.

### 9. GT self-consistency and jaw-symmetry tests

The patch includes tests that construct a prediction exactly from GT and require:

- diagnostic quality hit@1 = 1;
- oracle hit@64 = 1;
- translation error = 0;
- rotation error = 0;
- width error = 0;
- Global object correctness = 1.

A second test uses `R @ diag(-1,-1,1)` and requires zero parallel-jaw rotation error and a successful hit.

## Training comparability warning

Items 4, 5 and 6 intentionally change the gradient-generating training semantics:

- Global direct supervision frequency changes from sparse action-group coupling to one unique-state batch per action batch;
- false negative quality supervision is removed;
- action-stream sampling changes to task-balanced, unique-state-first selection; stratum coverage is only a non-invasive group-choice preference.

Therefore an optimizer trajectory produced before this patch is **not statistically comparable** to a trajectory produced after it. For a clean experiment, start a new run. If a previous checkpoint is used only as initialization, use the repository's initialization path rather than interpreting the result as a continuous unchanged training run.

## Focused verification

After applying the patch:

```powershell
python -m pytest -q tests/test_grasp_negative_matching.py tests/test_grasp_diagnostics.py tests/test_task_state_sampling.py tests/test_global_state_stream.py tests/test_validation_subset.py
```

Then run the repository's full test suite before formal training.
