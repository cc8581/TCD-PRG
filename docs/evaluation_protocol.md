# Evaluation protocol

TCD-PRG uses a **standard-only** public evaluation surface. Internal proxy metrics
are not exported, printed as performance metrics, or accepted by the paper-table
exporter. Training losses remain available for optimization monitoring, but they
are not performance metrics.

## Protocols retained

### Task Region

Binary semantic segmentation metrics are computed from dataset-level confusion
counts. The primary metric is mIoU, defined as the mean of foreground and
background IoU. Standard foreground/background precision, recall, F1, IoU and
Dice are also available.

### Dependency Graph

Relation prediction follows a scene-graph PredCls-style no-graph-constraint
ranking. Every valid relation triplet is independently ranked, multiple
predicates may be retained for the same object pair, and physical self-relations
are excluded. Report ngR@20/50/100 and ngmR@20/50/100.

### Grasp Verifier

Candidate-level binary classification reports pooled AP, AUROC, Precision,
Recall, F1, Brier score and ECE over explicitly valid labels.

### Global Grasp

Global grasp comparison is not computed from TCD pose-neighbourhood labels.
Use `tcd-prg-eval-graspnet`, which delegates the complete official evaluation
path to `graspnetAPI.GraspNetEval`: 3 cm / 30 degree NMS, object association,
Top-10 per object / scene Top-50 selection, collision and force-closure scoring,
and AP aggregation. The accepted outputs are GraspNet AP and AP_mu values.

### Task Grasp

Pose-neighbourhood hits are not reported. Task-oriented grasp performance is
measured only from executed trials as task success rate:

`successful task trials / executed task-grasp trials`.

Use `tcd-prg-eval-episodes` with `task_grasp_trial=true` and the corresponding
`task_success` outcome.

### Push / Policy

Head-level object Top-K, contact error, direction error, utility error, NDCG,
selected-candidate accuracy and labelled-transition replay are removed. Executed
closed-loop evaluation follows the three VPG metrics:

1. Completion Rate: completed trials / all trials.
2. Grasp Success Rate: per-completed-trial grasp successes / grasp attempts,
   averaged over completed trials.
3. Action Efficiency: per-completed-trial object count / actions before
   completion, averaged over completed trials.

Use `tcd-prg-eval-episodes`.

## Metrics deliberately removed

The following families are no longer part of evaluation output:

- `*_known_hit_at_K`, `*_known_recall_at_K`, pose-neighbourhood AP/precision;
- matched grasp translation/rotation/width errors;
- task-grasp anchor-region precision;
- relation AP/AUROC/F1 derived from independent edge classification;
- direct/indirect/actionable blocker proxy metrics;
- region-visibility proxy metrics;
- Push object Top-K, direction/contact/utility errors and candidate NDCG;
- policy candidate NDCG, selected-candidate success/type/object correctness;
- generated-candidate proxy metrics;
- labelled replay success/recovery/preparation metrics;
- planning-time entries previously classified as diagnostics.

If a task cannot be evaluated under its accepted protocol with available data,
no surrogate metric is emitted.

## Checkpoint selection

`best.pt` continues to minimize the configured weighted validation loss. This is
an optimization/checkpoint-selection quantity, not a paper performance metric.
Final comparisons must use the task-specific standard protocols above on the
validation/test benchmark or executed trial set.
