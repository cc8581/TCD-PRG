# Checkpoints and experiment outputs

Checkpoint schema version 9 stores model, optimizer, scheduler, AMP scaler, EMA,
trainer counters, resolved structured configuration, CPU RNG and all CUDA RNG
states, including per-DDP-rank RNG state. `best.pt`, periodic
`step_XXXXXXXX.pt` and `last.pt` use the same schema. Older full TCD-PRG
checkpoints are rejected with an explicit compatibility error; original GAPG
encoder weights use the separate pretrained-backbone loading path.

An experiment directory contains:

```text
resolved_config.yaml
run_metadata.json
loss_routing.json
train_metrics.jsonl
validation_metrics.jsonl
training_events.jsonl
tensorboard/
step_XXXXXXXX.pt
best.pt
last.pt
```

`train_metrics.jsonl` uses one record per successful optimizer step. It stores
all objective terms, learning rates, gradient norm, AMP state, throughput,
cumulative state counters, and the number of micro-batches/samples represented
by that optimizer step. Non-count diagnostics are averaged across the complete
gradient-accumulation window; generated candidate counts are summed.

`validation_metrics.jsonl` schema v3 stores the aggregate selection score,
every available validation loss/diagnostic, and the same component-performance
summary produced by offline evaluation. `validation_items` is the number of
state groups, not batches. See `evaluation_protocol.md` for exact definitions.
`training_events.jsonl` is an append-only audit trail for training start/end,
AMP overflow skips (including the discarded window metrics), checkpoint writes,
validation decisions, and early stopping. TensorBoard receives every successful
optimizer-step scalar and every validation scalar. Terminal output is deliberately
smaller and is controlled by `logging.log_interval`.

Evaluation adds `metrics.json` and UTF-8-BOM `per_task.csv`. The JSON contains
global metrics, grouped metrics and the exact configuration. Each metric records
mean, standard deviation, scene-clustered confidence bounds and contributing
sample count. Metrics with no valid observations are omitted rather than filled
with synthetic zero/None placeholders.

Checkpoint resume restores numerical training state; it does not silently
replace the current dataset snapshot. Keep the snapshot/audit report with the
experiment and reject a resume if its intended data provenance differs.
