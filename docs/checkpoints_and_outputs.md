# Checkpoints and experiment outputs

Checkpoint schema version 1 stores model, optimizer, scheduler, AMP scaler, EMA,
trainer counters, resolved structured configuration, CPU RNG and all CUDA RNG
states. `best.pt`, periodic `step_XXXXXXXX.pt` and `last.pt` use the same schema.

An experiment directory contains:

```text
resolved_config.yaml
run_metadata.json
loss_routing.json
train_metrics.jsonl
tensorboard/
step_XXXXXXXX.pt
best.pt
last.pt
```

Evaluation adds `metrics.json` and UTF-8-BOM `per_task.csv`. The JSON contains
global metrics, grouped metrics and the exact configuration. Each metric records
mean, standard deviation, confidence bounds and contributing sample count.

Checkpoint resume restores numerical training state; it does not silently
replace the current dataset snapshot. Keep the snapshot/audit report with the
experiment and reject a resume if its intended data provenance differs.
