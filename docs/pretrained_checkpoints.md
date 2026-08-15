# External pretrained checkpoints

The original GAPG and GraspNet weights are runtime dependencies and are not
committed to this private repository. They were downloaded from the links
published by the respective upstream authors and verified by strict state-dict
loading on 2026-07-30.

| Model | Upstream source | Local path | Bytes | SHA-256 | Required top-level keys |
|---|---|---:|---:|---|---|
| GAPG grasp verifier | [GAPG pretrained folder](https://drive.google.com/drive/folders/1UAgLYQEvscLsoyXO37xN8M3frhLiBtHa) | `.deps/checkpoints/gapg/grasp_model.pt` | 17,487,918 | `571A2D095735BF5E14275EB5F597DC1A95C491DFE40E0F1FB2FFA89882E9EF25` | `model`, `optimizer`, `epoch`, `best_acc` |
| GAPG push evaluator | [GAPG pretrained folder](https://drive.google.com/drive/folders/1UAgLYQEvscLsoyXO37xN8M3frhLiBtHa) | `.deps/checkpoints/gapg/push_model.pt` | 16,973,750 | `0AA413A08675608C16F045EE9FE4B862DB71090958BF69F921AF68423131701E` | `model`, `optimizer`, `epoch`, `best_acc` |
| GraspNet RealSense proposal model | [GAPG-published GraspNet file](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view) | `.deps/checkpoints/graspnet-rs.tar` | 12,468,415 | `60680087C61CBA2B6791614FEF1519071E294F6DCAF99B3F581BB95F7C51A868` | `model_state_dict`, `optimizer_state_dict`, `epoch`, `loss` |

All baseline checkpoints live under the ignored `.deps/checkpoints/` tree and
never mix with the TCD-PRG source package or training outputs.

All three files are ignored by Git. Redistribution and use remain subject to
the upstream GAPG and GraspNet licenses.

## GraspNet fine-tuning boundary

`graspnet.freeze: true` remains the default. Setting it to `false` enables the
adapter's separate official-training API, but does not add GraspNet parameters
to the TCD-PRG optimizer, EMA, DDP broadcasts or checkpoint. Call
`prepare_finetuning(device)`, create an optimizer from `finetune_parameters()`,
train only with the complete upstream dense-label schema, and save
`finetune_state_dict()` as a standalone checkpoint. Sparse task-grasp pose sets
are rejected by `official_training_loss()` because unmatched grasps are not
valid negatives.

## Verified smoke run

The full TCD-PRG training entry point was run on one real cached state at 2,048
points with AMP enabled and stopped after one successful optimizer step. The
run writes resolved configuration, Git metadata, capability-aware loss routing,
JSONL and TensorBoard metrics, an interval checkpoint, and `last.pt`.

The smoke audit confirmed finite total loss and gradient norm, 305 populated
AdamW parameter states at step 1, finite model and EMA tensors, matching
interval/final checkpoints, and successful resume through the normal training
entry point. AMP overflow retries are counted separately and do not advance the
optimizer, scheduler, EMA, or checkpoint counters.
