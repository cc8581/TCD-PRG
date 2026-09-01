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
SA centre limits 1024/256/64/16, radii 0.1/0.2/0.4/0.8 m, 32 neighbours, four feature
propagation levels and original convolution/BatchNorm layers. No SA/FP block,
channel count, sampler or normalization layer has been replaced or reduced.
The adapter caps each SA layer's centre count at its actual input point count
for that call, then restores the original limits. Thus 1024/4096-point inputs
use 1024->256->64->16 centres; a 512-point input uses 512->256->64->16. Upstream
source and pretrained tensor shapes are unchanged; radii remain metric values.
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
World action geometry and masks stay aligned to their sampled points. During
training only, the original observation is reduced to the configured point count before the
PUSH network using `pytorch3d.ops.sample_farthest_points` and its compiled extension
(CUDA on GPU). XYZ, RGB, object, target and functional-region point fields use the
same FPS indices. Masked padding is excluded; short clouds retain only their valid
FPS selections and padded slots are masked out before the backbone. Actions and
GT outcomes are not resampled or relabeled.
Pure FPS can omit small objects/regions; their original action eligibility is
retained, and their sampled probability may be all zero. No forced point insertion
is performed. Validation and deployed inference retain their original point counts.
This training/inference density difference is intentional for this change.

Stage-C training requires a CUDA-enabled official PyTorch3D build (v0.7.7,
commit `89653419d0973396f3eff1a381ba09a07fffc2ed`); install using the upstream
Windows/Linux build instructions. Training probes compiled FPS before loading
data and fails if unavailable. There is no Python FPS fallback. This optional C
dependency is not imported by Stage A/B. The raw dataset/cache point count remains
unchanged on disk. Restart the training process to adopt changed point settings.

Both settings live in `configs/stage/push_evaluator.yaml`:

```yaml
dataset:
  scene_points: 16384  # Raw collated points; 0 retains all observed points.
training:
  push_fps_points: 4096  # Network input limit after compiled FPS; e.g. 1024.
```

`push_fps_points` must be a positive integer and must not exceed a positive
`scene_points` limit. No network-input count is hardcoded in the trainer. Setting
both to 1024 means 1024 raw points and at most 1024 valid network points; FPS
cannot recover geometry removed by earlier collation. Changing the raw point
limit also affects this C configuration's validation inputs, but does not alter
the A/B configurations. The FPS setting itself applies only during training.

For direct network inputs, fewer than 32 visible points are repeated to meet upstream
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

On Windows, launch C through `train.py --stage push_evaluator` or the root
`train_push_evaluator.py`. The C supervisor requires `pywin32` (`python -m pip
install pywin32`) and assigns the suspended training process to a kill-on-close
Job Object before it can spawn workers. Ctrl+C requests graceful shutdown, waits
up to five seconds, then closes the job to terminate remaining descendants.
Workers ignore console interrupts so queue/DLL teardown is owned by the trainer;
the C child environment sets `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` before loading
numerical libraries, preventing Intel Fortran's console handler from aborting
Python before cleanup. This does not change the parent's or A/B's environment.
training and validation DataLoader iterators are explicitly shut down in `finally`.
This also covers exceptions and forced supervisor exit. Interrupted partial
optimizer updates are not saved; resume from the latest completed checkpoint.
The Job Object guarantee is Windows-specific and does not cover kernel/driver
termination failures; a successful termination request is not itself proof that
the process has finished exiting. A/B launch and training paths are unchanged.

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
