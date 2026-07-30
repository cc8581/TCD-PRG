# TCD-PRG resource-profile interpretation

This profile uses one real full-resolution action-state group on an RTX 3090:
batch size 1, 16,384 scene points, 4,096 attention anchors, 72 action
candidates, FP16 AMP, activation checkpointing, verifier micro-batch 16, and
gradient accumulation 8.

## Why high FLOPs and low VRAM are compatible

The measurements are internally consistent. FLOPs accumulate every arithmetic
operation over time, whereas peak VRAM only counts tensors simultaneously live.
TCD-PRG performs substantial repeated point/candidate computation but avoids a
dense 16,384 by 16,384 attention matrix:

- the point transformer selects 4,096 anchors and uses local KNN attention with
  16 neighbors;
- scene features are computed once and shared by all task heads;
- verifier candidates are processed in micro-batches of 16;
- activation checkpointing discards selected forward activations and recomputes
  them during backward, deliberately trading more computation for less memory;
- PyTorch reuses freed CUDA blocks, so reserved memory is not the sum of every
  intermediate ever produced.

The model has 19,270,266 parameters. Its FP32 model state is 77,081,064 bytes
(73.51 MiB), EMA is another 73.51 MiB, and initialized optimizer tensors are
127.94 MiB. Parameter count therefore does not imply multi-gigabyte VRAM.

## Measured full-configuration values

| Item | Measurement |
|---|---:|
| Forward lower bound | 258.65 GFLOPs |
| Backward lower bound | 610.63 GFLOPs |
| One forward/backward micro-batch lower bound | 869.28 GFLOPs |
| One optimizer step, accumulation 8 | 6.954 TFLOPs |
| Peak CUDA allocated | 1.156 GiB |
| Peak CUDA reserved | 1.348 GiB |
| CUDA allocated before the measured step | 0.200 GiB |
| Incremental step peak | 0.956 GiB |
| Process peak working set | 1.851 GiB |
| Process private memory | 4.071 GiB |
| Median successful micro-batch CUDA time | 405.83 ms |
| Estimated optimizer-step CUDA time | 3.150 s |
| Estimated throughput | 2.54 state groups/s |

The FLOP counter is a conservative operator-dispatch lower bound: unsupported
elementwise, indexing, sampling, loss, and optimizer operations are omitted.
The timing is a single-GPU kernel measurement and excludes distributed
communication.

## Effect of batch size and DDP

Batch size 1 is an important reason for the low peak. A rough activation-only
linear projection from this sample gives about 2.11 GiB allocated for batch 2,
4.02 GiB for batch 4, and 7.85 GiB for batch 8. These are planning estimates,
not guarantees: different object/candidate counts, allocator fragmentation and
DDP buckets change the real peak. Batch sizes must therefore be profiled with
the intended data distribution before being raised.

DDP does not pool GPU memory into one larger device. Each GPU holds one model,
its optimizer/EMA state and its local batch, so batch 1 should remain near the
single-GPU per-device peak plus DDP communication buckets. With two GPUs,
`batch_size=1` and `gradient_accumulation_steps=8` produce a global effective
batch of 16. To retain the previous effective batch of 8, use accumulation 4.

The raw machine-readable measurement is in
`reports/training_resource_profile_full_portable_check.json`.
