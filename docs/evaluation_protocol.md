# Evaluation protocol

Training validation and `tcd-prg-evaluate` use `OfflineModelEvaluator`; metric
names therefore have one implementation and one aggregation rule. Losses remain
optimization diagnostics. They are not renamed as accuracy or task success.

## Aggregation

- Binary predictions use explicit thresholds from `evaluation.*` (graph edges
  use `model.graph_edge_threshold`).
- Confusion counts are summed before precision, recall, F1, IoU and Dice are
  calculated. They are not averaged batch by batch.
- AUROC, non-interpolated average precision and Brier score use scikit-learn's
  reference implementations, which correctly group equal scores. Calibration
  also reports an explicitly defined 15-bin equal-width ECE by default.
- Scalar confidence intervals resample complete scenes, not correlated states
  inside a scene. Every metric includes its contributing count and a 95% scene-
  cluster bootstrap interval by default.
- UNKNOWN candidates are excluded from supervised ranking metrics. Selection
  coverage is reported separately as `selected_candidate_known`.
- Metrics without valid observations are omitted. The exporter never creates a
  zero-valued placeholder for an unimplemented metric.

## Lightweight component metrics

The shared evaluator reports foreground region precision/recall/F1/IoU/Dice,
visibility AP/AUROC/calibration, per-relation graph AP/AUROC/confusion metrics,
derived blocker metrics, Verifier AP/AUROC/F1/calibration, Push object Top-1/3,
contact distance, direction angle, utility MAE and candidate NDCG, and policy
Top-1 known/success plus NDCG. Teacher and generated policy candidates have
separate names.

For Task and Global Grasp, predictions are confidence-ranked and greedily
one-to-one matched after the configured translation, parallel-jaw-symmetric
rotation, width and object checks. Because native grasp libraries are not an
exhaustive grasp universe, the always-valid metrics are named
`known_hit_at_K` and `known_recall_at_K`. Precision and AP are emitted only when
the data producer explicitly sets `label_set_complete=true`. A valid but unseen
grasp is therefore never silently counted as a false positive.

## Closed-loop terminology

`labelled_replay_task_success_hK` follows selected actions through known HDF5
state transitions for at most K preparation actions. UNKNOWN selections make
the episode unevaluable. This is an offline replay proxy, not physics or robot
execution success, and the name must not be shortened to `task_success_rate`.

Formal end-to-end reports must additionally execute fixed validation/test scenes
and report real rollout task success, grasp execution success, task compliance,
Push outcome success, invalid-action rate, action count, planning/execution time,
failure taxonomy and multiple seeds. Online results must live in a separate
evaluation artifact so they cannot be confused with labelled replay.

## Checkpoint selection

`best.pt` continues to minimize the configured weighted validation loss so
training remains resumable across sparse metric subsets. Performance metrics are
stored beside the loss in validation schema v3 and must be used when selecting
checkpoints for final offline/online comparison. Never select on the test split.

## Protocol references

- Pointcept semantic evaluation: global intersection/union counts and
  mIoU/mAcc/mPrecision/macro-F1.
- scikit-learn metrics: `roc_auc_score`, `average_precision_score` and
  `brier_score_loss`.
- GraspNet: pose NMS followed by confidence-ranked Precision@K/AP under an
  executable-grasp protocol.
- M2T2: precision-coverage simulation evaluation plus execution success rates.

The project-specific extensions above preserve these conventions but use names
that expose label incompleteness and labelled-transition replay explicitly.
