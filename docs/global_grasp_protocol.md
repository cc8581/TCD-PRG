# Global task-free grasp branch and comparison protocol

The global branch estimates task-independent grasps for every physically
present, visible object. It is distinct from functional-region task grasping
and from PICK_REMOVE policy supervision.

## Information boundary

The expensive scene backbone consumes only XYZ, RGB, and point validity. It
does not consume target masks, task/category/region tokens, relation graphs, or
object poses. Instance membership is applied only after neutral point features
have been computed.

Two comparison tracks must be reported separately:

- `scene_only`: the global head uses neutral point and global scene features.
  Instance masks are used only after prediction to assign grasps to objects for
  evaluation.
- `instance_assisted`: every method receives the same external instance masks.
  Methods without native mask support run once per instance crop, then merge
  predictions in world coordinates.

The configured track is part of the checkpoint architecture and must not be
changed after training.

## Labels

`generic_grasp_library_v1` is generated from the complete ACRONYM grasp set,
without functional-region purity filtering. Every record stores original
source index, object-frame contact pose, contact points, approach direction,
AG total opening, intrinsic stability, source quality, and a conversion
version.

Quality is split into two levels:

- intrinsic: stable for the isolated object and compatible with AG opening;
- scene executable: intrinsic AND collision-free AND approach-free and
  FR5-reachable in the current state.

Scene executability uses `1` positive, `0` explicit negative, and `-1`
unknown/untested. Unknown rows are ignored by confidence losses. Scene
certification is generated offline with `tools/certify_global_grasps.py`, never
synchronously inside the GPU training loop.

Existing PICK_REMOVE sequences are retained. Their old grasp labels are
associated with the new library by exact ACRONYM source index, then by
same-object SE(3) and opening proximity. Unmatched grasps remain explicit and
are not silently discarded.

## Multimodal output

The head predicts four modes per visible contact point by default. Set matching
is performed before approach, rotation, opening, and confidence losses. Target
modes are selected with farthest-first pose/opening diversity, so nearby
ACRONYM candidates do not collapse to several equivalent labels.

Outputs are contact heatmap, approach, in-plane rotation, AG total opening,
intrinsic confidence, and scene-executable confidence. There is no task
compatibility output.

## Evaluation

Raw proposal metrics are computed before the common certifier: Recall@K, AP,
minimum set-valued translation/rotation/opening error, object coverage, and
diversity. Certified metrics are reported separately: collision-free
Precision@K, IK-feasible rate, and certified/physical success.

Matching requires the same object instance, translation <= 10 mm, rotation <=
15 degrees under parallel-jaw 180-degree symmetry, and opening error <= 5 mm.
Metrics are computed after the configured common SE(3) NMS. Exact thresholds
and whether NMS is enabled are saved with every result file.
