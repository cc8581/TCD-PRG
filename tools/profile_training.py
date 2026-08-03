from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import TCDPRGModel
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter, create_gripper_provider
from tcd_prg.trainers.reproducibility import seed_everything

COMMON_OVERRIDES = [
    "training.max_train_groups=1",
    "training.max_validation_groups=1",
    "training.max_optimizer_steps=1",
    "training.num_workers=0",
    "training.pin_memory=false",
    "training.validation_interval=1000",
    "training.checkpoint_interval=1",
    "logging.log_interval=1",
]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def tensor_bytes(values) -> int:
    return sum(value.numel() * value.element_size() for value in values if torch.is_tensor(value))


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--dataset-snapshot",
        type=Path,
        default=PROJECT / "benchmarks" / "dataset_group_snapshot.json",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    run = checkpoint_path.parent
    dataset_snapshot_path = args.dataset_snapshot.resolve()
    mode = "full" if args.full else "smoke"
    overrides = [*COMMON_OVERRIDES, f"output_dir={run.as_posix()}", *args.overrides]
    if args.full:
        overrides.append("training.gradient_accumulation_steps=1")
    else:
        overrides.extend([
            "dataset.scene_points=2048",
            "dataset.target_points=1024",
            "backbone.patch_size=128",
            "training.gradient_accumulation_steps=1",
        ])
    output = (args.report or (
        PROJECT / "outputs" / "profiles" / f"training_resource_profile_{mode}.json"
    )).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = output.with_suffix(".csv")
    overall_started = time.perf_counter()
    config = load_config(PROJECT / "configs/config.yaml", overrides)
    seed_everything(config.training.seed, config.training.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_started = time.perf_counter()
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(adapter, split="train", max_groups=1)
    dataset_snapshot = (
        json.loads(dataset_snapshot_path.read_text(encoding="utf-8"))
        if dataset_snapshot_path.exists() else None
    )
    train_group_count = (
        dataset_snapshot["action_state_groups_by_split"].get("train", 0)
        if dataset_snapshot else None
    )
    validation_group_count = (
        dataset_snapshot["action_state_groups_by_split"].get("val", 0)
        if dataset_snapshot else None
    )
    gripper = create_gripper_provider(config, allow_generate=False)
    collator = UnifiedBatchCollator(config, gripper)
    sample_started = time.perf_counter()
    sample = dataset[0]
    sample_seconds = time.perf_counter() - sample_started
    collate_started = time.perf_counter()
    raw_batch = collator([sample])
    collate_seconds = time.perf_counter() - collate_started
    data_setup_seconds = time.perf_counter() - data_started

    model_started = time.perf_counter()
    model = TCDPRGModel(
        config.model, config.ablation, config.graph, config.router, config.backbone
    )
    backbone_parameters = list(model.encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.optimizer.backbone_learning_rate},
            {"params": other_parameters, "lr": config.optimizer.learning_rate},
        ],
        weight_decay=config.optimizer.weight_decay,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).train()
    optimizer.load_state_dict(checkpoint["optimizer"])
    objective = TCDPRGObjective(
        adapter.capabilities, config.model, config.ablation, config.losses,
        config.region_head,
    )
    model_setup_seconds = time.perf_counter() - model_started

    parameter_rows = []
    top_level = defaultdict(lambda: {"total": 0, "trainable": 0, "bytes": 0})
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        count = parameter.numel()
        top_level[group]["total"] += count
        top_level[group]["trainable"] += count if parameter.requires_grad else 0
        top_level[group]["bytes"] += count * parameter.element_size()
    for name in sorted(top_level):
        parameter_rows.append({"module": name, **top_level[name]})
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    model_bytes = tensor_bytes(model.state_dict().values())
    ema_bytes = tensor_bytes(checkpoint["ema"].values())
    optimizer_bytes = 0
    for state in checkpoint["optimizer"]["state"].values():
        optimizer_bytes += tensor_bytes(state.values())

    synchronize()
    move_started = time.perf_counter()
    batch = move(raw_batch, device)
    synchronize()
    move_seconds = time.perf_counter() - move_started
    batch_shapes = {
        key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)
    }

    # Timed full training attempt at the already validated AMP scale.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    scaler.load_state_dict(checkpoint["scaler"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    baseline_reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
    amp_retry_count = 0
    attempt_timings = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        wall_started = time.perf_counter()
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, terms = objective(model, batch)
        forward_end.record()
        scaler.scale(loss).backward()
        backward_end.record()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_end.record()
        synchronize()
        timing = {
            "forward_and_loss_ms": forward_start.elapsed_time(forward_end),
            "backward_ms": forward_end.elapsed_time(backward_end),
            "unscale_clip_optimizer_ms": backward_end.elapsed_time(optimizer_end),
            "successful_step_total_cuda_ms": forward_start.elapsed_time(optimizer_end),
            "successful_step_wall_seconds": time.perf_counter() - wall_started,
        }
        step_skipped = scaler.get_scale() < old_scale
        attempt_timings.append({**timing, "scale": float(old_scale), "skipped": step_skipped})
        if not step_skipped:
            break
        amp_retry_count += 1
    else:
        raise RuntimeError("AMP did not produce a successful step after eight scale retries")

    benchmark_timings = [timing]
    for _ in range(9):
        optimizer.zero_grad(set_to_none=True)
        wall_started = time.perf_counter()
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, terms = objective(model, batch)
        forward_end.record()
        scaler.scale(loss).backward()
        backward_end.record()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_end.record()
        synchronize()
        if scaler.get_scale() < old_scale:
            amp_retry_count += 1
            continue
        benchmark_timings.append({
            "forward_and_loss_ms": forward_start.elapsed_time(forward_end),
            "backward_ms": forward_end.elapsed_time(backward_end),
            "unscale_clip_optimizer_ms": backward_end.elapsed_time(optimizer_end),
            "successful_step_total_cuda_ms": forward_start.elapsed_time(optimizer_end),
            "successful_step_wall_seconds": time.perf_counter() - wall_started,
        })
    timing_keys = tuple(benchmark_timings[0])
    timing = {
        key: statistics.median(item[key] for item in benchmark_timings)
        for key in timing_keys
    }
    total_cuda_samples = sorted(item["successful_step_total_cuda_ms"] for item in benchmark_timings)
    benchmark_summary = {
        "successful_repeats": len(benchmark_timings),
        "total_cuda_ms_min": min(total_cuda_samples),
        "total_cuda_ms_median": statistics.median(total_cuda_samples),
        "total_cuda_ms_p95": total_cuda_samples[math.ceil(0.95 * len(total_cuda_samples)) - 1],
        "total_cuda_ms_max": max(total_cuda_samples),
    }
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0

    # Operator-dispatch FLOP accounting. Unsupported operations are not guessed,
    # so these values are conservative lower bounds.
    optimizer.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as forward_counter:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            counted_loss, _ = objective(model, batch)
    forward_flops = int(forward_counter.get_total_flops())
    with FlopCounterMode(display=False) as backward_counter:
        counted_loss.backward()
    synchronize()
    backward_flops = int(backward_counter.get_total_flops())

    accumulation = config.training.gradient_accumulation_steps
    forward_backward_ms = timing["forward_and_loss_ms"] + timing["backward_ms"]
    optimizer_step_cuda_ms_estimate = (
        accumulation * forward_backward_ms + timing["unscale_clip_optimizer_ms"]
    )
    if train_group_count:
        optimizer_steps_per_epoch = math.ceil(train_group_count / accumulation)
        epoch_gpu_compute_seconds_estimate = (
            train_group_count * forward_backward_ms
            + optimizer_steps_per_epoch * timing["unscale_clip_optimizer_ms"]
        ) / 1000.0
        epoch_no_overlap_seconds_estimate = (
            epoch_gpu_compute_seconds_estimate
            + train_group_count * (sample_seconds + collate_seconds + move_seconds)
        )
    else:
        optimizer_steps_per_epoch = None
        epoch_gpu_compute_seconds_estimate = None
        epoch_no_overlap_seconds_estimate = None

    metrics = json.loads((run / "train_metrics.jsonl").read_text(encoding="utf-8").strip())
    process_memory = psutil.Process(os.getpid()).memory_info()
    report = {
        "scope": {
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": config.training.batch_size * accumulation,
            "scene_points": config.dataset.scene_points,
            "target_points": config.dataset.target_points,
            "backbone": config.backbone.backend,
            "voxel_size_m": config.backbone.grid_size_m,
            "patch_size": config.backbone.patch_size,
            "flash_attention": config.backbone.enable_flash_attention,
            "amp_dtype": config.training.amp_dtype,
            "activation_checkpointing": config.model.activation_checkpointing,
            "candidate_micro_batch": config.model.verifier_candidate_micro_batch,
            "profiled_unit": list(dataset.units[0].__dict__.values()) if hasattr(dataset.units[0], "__dict__") else {
                "scene_id": dataset.units[0].scene_id,
                "state_id": dataset.units[0].state_id,
                "task_index": dataset.units[0].task_index,
                "group_index": dataset.units[0].group_index,
            },
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "non_trainable": total_parameters - trainable_parameters,
            "buffers": buffer_count,
            "model_state_bytes": model_bytes,
            "ema_state_bytes": ema_bytes,
            "optimizer_state_bytes": optimizer_bytes,
            "by_top_level_module": parameter_rows,
        },
        "computation": {
            "forward_flops_lower_bound": forward_flops,
            "backward_flops_lower_bound": backward_flops,
            "training_flops_lower_bound": forward_flops + backward_flops,
            "optimizer_step_flops_lower_bound": accumulation * (forward_flops + backward_flops),
            "mac_equivalent_forward_approx": forward_flops / 2,
            "counter_note": "PyTorch operator-dispatch counter; unsupported elementwise, indexing, sampling, loss, and optimizer operations are omitted.",
        },
        "timing": {
            "sample_load_seconds": sample_seconds,
            "collate_candidate_geometry_seconds": collate_seconds,
            "data_setup_seconds": data_setup_seconds,
            "cpu_to_gpu_seconds": move_seconds,
            "model_optimizer_checkpoint_setup_seconds": model_setup_seconds,
            **timing,
            "recorded_smoke_elapsed_seconds_including_amp_retries": metrics["elapsed_seconds"],
            "recorded_amp_retries": metrics["amp_skipped_steps"],
            "profile_amp_retries": amp_retry_count,
            "profile_attempt_timings": attempt_timings,
            "steady_step_benchmark": benchmark_summary,
            "optimizer_step_cuda_ms_estimate": optimizer_step_cuda_ms_estimate,
            "train_group_count_snapshot": train_group_count,
            "validation_group_count_snapshot": validation_group_count,
            "optimizer_steps_per_effective_epoch": optimizer_steps_per_epoch,
            "effective_epoch_gpu_compute_seconds_estimate": epoch_gpu_compute_seconds_estimate,
            "effective_epoch_no_overlap_seconds_estimate": epoch_no_overlap_seconds_estimate,
            "profile_process_seconds": time.perf_counter() - overall_started,
        },
        "memory": {
            "baseline_gpu_allocated_bytes": baseline_allocated,
            "baseline_gpu_reserved_bytes": baseline_reserved,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
            "process_rss_bytes": process_memory.rss,
            "process_peak_working_set_bytes": process_memory.peak_wset,
            "process_private_bytes": process_memory.private,
            "checkpoint_last_bytes": checkpoint_path.stat().st_size,
            "checkpoint_interval_bytes": (run / "step_00000001.pt").stat().st_size,
        },
        "successful_step": {
            "loss_total": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "amp_scale_before": float(old_scale),
            "amp_scale_after": float(scaler.get_scale()),
            "step_skipped": bool(step_skipped),
            "amp_retry_count": amp_retry_count,
            "finite_loss_terms": all(torch.isfinite(value).all() for value in terms.values()),
            "loss_term_count": len(terms),
        },
        "batch_shapes": batch_shapes,
        "source_smoke_metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module", "total", "trainable", "bytes"])
        writer.writeheader()
        writer.writerows(parameter_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
