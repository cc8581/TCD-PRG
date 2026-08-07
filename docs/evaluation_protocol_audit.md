# TCD-PRG Domain Evaluation Protocol Audit — Standard-Only Revision

This revision removes all TCD-specific proxy/diagnostic performance metrics from
the public evaluation surface. A task is reported only when its inputs support a
protocol that is directly comparable with established work in that field.

## Standard metric whitelist

| TCD-PRG task | Protocol retained | Evaluation path |
|---|---|---|
| Task Region | binary segmentation IoU/mIoU | native offline evaluator |
| Dependency Graph | no-graph-constraint relation R@K/mR@K | native offline evaluator |
| Verifier | AP/AUROC/Precision/Recall/F1/Brier/ECE | native offline evaluator |
| Global Grasp | official GraspNet AP/AP_mu | `tcd-prg-eval-graspnet` |
| Task Grasp | executed task success rate | `tcd-prg-eval-episodes` |
| Push / Policy | VPG Completion, Grasp Success, Action Efficiency | `tcd-prg-eval-episodes` |

## Removed metric families

The patch deletes or suppresses every previously exported metric that was not
benchmark-comparable, including known-pose grasp hits/recall/AP, matched pose
errors, task-anchor precision, blocker proxy metrics, relation edge-classifier
AP/AUROC/F1, region visibility proxies, push head Top-K/contact/direction/utility
errors, NDCG, selected-candidate metrics, generated-candidate metrics, labelled
replay metrics and planning-time diagnostic fields.

The legacy `GlobalGraspEvaluator` module and `tcd-prg-eval-global` command module/entry
point are physically deleted. They are replaced by the official GraspNet evaluation
entry point.

## Why some training-time tasks have no validation performance number

Global Grasp, Task Grasp, Push and Policy cannot be scored honestly from the
native sparse action-group labels using the accepted field protocols. During
training their losses remain visible for optimization monitoring, but no proxy
performance number is printed. Their final performance is evaluated through the
external GraspNet or executed-episode protocols.

## VPG aggregation

For N test runs:

- Completion Rate = completed runs / all runs.
- Grasp Success Rate = mean over completed runs of successful grasps / grasp attempts.
- Action Efficiency = mean over completed runs of object count / total actions before completion.

Failed runs affect Completion Rate but are not inserted into the per-completion
grasp-success/action-efficiency averages.

## Task-oriented grasp aggregation

Task Success Rate = successful executed task-grasp trials / all executed
task-grasp trials. Pose closeness by itself is not a task success.

## GraspNet boundary

The TCD evaluator does not reconstruct GraspNet AP from pose-neighbourhood
labels. `tcd-prg-eval-graspnet` imports `graspnetAPI.GraspNetEval` and delegates the
official evaluation path to it, including the 3 cm / 30 degree NMS implemented
inside `eval_grasp`, per-object Top-10 / scene Top-50 selection, collision and
force-closure scoring, and AP aggregation.

## Non-negotiable export rule

`Evaluator.summarize()` and `tcd-prg-export-paper` reject/non-export any metric
outside the audited standard whitelist. There is no `--allow-diagnostic` escape
hatch in this revision.
