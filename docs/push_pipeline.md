# PUSH: yanx27 PointNet++ transfer learning

The backbone is imported directly from yanx27/Pointnet_Pointnet2_pytorch, not a
locally simplified PointNet++. Model and utility source files are unchanged and
SHA-256 checked before import. This is a third-party implementation, not the
original paper authors' TensorFlow release.

## Dependency and published weights

From the project root, install the pinned upstream checkout once:

```sh
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git third_party/pointnet2_yanx27
git -C third_party/pointnet2_yanx27 checkout eb64fe0b4c24055559cea26299cb485dcb43d8dd
```

The checkout includes its MIT license and published S3DIS semantic segmentation
weights: `log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth`.
SHA-256: `5ac6896d077ca7302fc263626f8239cffc9717785f067ea94221194dfcb79625`.
These dependencies are local and ignored by the parent repository; runtime never
downloads files. The original TensorFlow checkout under pointnet2_official, if
present, is reference material only and is never imported.

`python train.py --stage push_evaluator` starts fresh PUSH training by strictly
loading every upstream network tensor, including BatchNorm buffers. PUSH fusion,
head and the 128-to-feature_dim projection start randomly. Missing or altered
upstream files/weights cause errors, never random fallback. No perception
checkpoint or frozen perception encoder is involved.

## Network and input adaptation

Upstream `models/pointnet2_sem_seg.py:get_model(13)` runs its original forward:
SA centres 1024/256/64/16, radii 0.1/0.2/0.4/0.8 m, 32 neighbours, four feature
propagation levels and original convolution/BatchNorm layers. No SA/FP block,
channel count, sampler or normalization layer has been replaced or reduced.
A pre-hook captures the 128-channel per-point features before classifier dropout;
a Linear projection connects them to the existing PUSH feature width.
The original 13-class S3DIS conv2 remains for strict state loading and has
requires_grad=False: its unused segmentation logits are not a PUSH objective.
Every feature-producing backbone parameter, projection and PUSH head is trainable.
BatchNorm statistics update during PUSH training.

The model requires nine channels. The adapter supplies metric XYZ with scene-
centred XY and Z above its lowest visible point, RGB in [0,1], and scene-bounds-
normalized XYZ in [0,1]. S3DIS originally uses indoor room/block coordinates;
this is an explicit scene-domain adaptation, not identical S3DIS preprocessing
and not a guarantee of transfer performance. RGB is not treated as normals.
World action geometry and masks stay aligned to original points. No point-count
reduction is performed. Fewer than 32 visible points are repeated to meet upstream
ball-query/BatchNorm requirements, then outputs are sliced to original count.
Only valid points enter the network.

Training uses upstream random FPS. Evaluation fixes FPS randomness inside an
isolated RNG context, leaving caller RNG unchanged and upstream code unmodified.
Each scene is encoded once per batch and shared by its actions. Full encoding
costs more time/memory than the old simplified network; reduce only C batch_size
and use accumulation if needed. C currently requires FP32 (training.amp=false).

## Objective and inference

Training uses GT object/target/functional-region conditions, complete logged PUSH
actions and binary improvement labels. UNKNOWN actions do not contribute to loss.
Rules are inference-only. Accumulation normalizes by known-action count; empty
microbatches consume no slots and epoch tails are flushed. Steps and scheduling
count optimizer updates.

Inference uses Stage A predicted conditions, geometric rule actions and the same
fine-tuned C. Object, target, local contact, path and region pooling, task embeddings
and world action geometry enter the PUSH head. No GT labels or perception features
are required by C inference. Rule contacts are proposals, not collision/reachability
certification. Scores estimate improvement, not robot execution feasibility.

## Checkpoints and stage isolation

Scene encoding batches all action-bearing scenes with the same visible point
count into one upstream PointNet++ call. Fixed-size clouds therefore share one
call per batch; variable-size clouds are grouped by count rather than padded
through an upstream network that has no padding mask. Scenes without actions are
not encoded. Training BatchNorm now uses the whole count group, so optimization
is not numerically equivalent to the former per-scene calls. Upstream random FPS
also depends on batch composition: evaluation repeats for an identical batch,
but regrouping scenes need not reproduce single-scene scores exactly. Parameter
shapes and checkpoint format are unchanged. Restart training to use this change;
an already running Python process retains its imported implementation.

PUSH protocol: 5; architecture: `yanx27_pointnet2_sem_seg_s3dis_v1`.
Protocol-4 simplified-network weights and older PUSH weights are rejected without
migration. New checkpoints contain the complete upstream network, BN buffers,
projection and head. Deployment requires upstream source but does not reread the
S3DIS pretraining checkpoint. `--pretrain-checkpoint` initializes a fine-tuned
protocol-5 PUSH evaluator; `--resume` restores its weights, optimizer, scheduler
and best selection. Neither overwrites those weights with S3DIS initialization.
Resume begins a newly shuffled pass, not exact interrupted-loader replay.

Stage A/B code, configuration and initialization layout are unchanged. The combined
model creates the backbone only for C execution/loading. Existing inactive-module
A/B handling is not compatibility with obsolete C weights.

Periodic validation selects best AP on logged actions; final validation evaluates
best on the full split. evaluate_push_pipeline.py separately measures GT-condition
and predicted-condition rule candidates, treating unmatched outcomes as UNKNOWN.
These metrics and synthetic overfitting do not prove real robot success or better
generalization than the previous network.
