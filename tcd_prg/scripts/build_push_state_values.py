"""Stream observations through frozen Stage-A/B and persist compact state values."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.datasets.push_value import StateValues, write_state_values
from tcd_prg.models import StageBCondition, TCDPRGModel, load_staged_tcd_prg
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import create_adapter


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    return value


def _pad_cat(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate singleton batches, padding only their variable point axis."""
    if not tensors or any(value.shape[0] != 1 for value in tensors):
        raise ValueError("batched state tensors must each have a singleton batch axis")
    if all(value.shape[1:] == tensors[0].shape[1:] for value in tensors):
        return torch.cat(tuple(tensors), dim=0)
    if any(value.ndim < 2 or value.shape[2:] != tensors[0].shape[2:] for value in tensors):
        raise ValueError("only the point axis may vary while batching states")
    width = max(value.shape[1] for value in tensors)
    output = tensors[0].new_zeros((len(tensors), width, *tensors[0].shape[2:]))
    for row, value in enumerate(tensors):
        output[row, : value.shape[1]] = value[0]
    return output


def _collate_batches(batches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-size point clouds without leaking label-side fields."""
    if not batches:
        raise ValueError("cannot collate an empty state batch")
    result: dict[str, Any] = {}
    for group in batches[0]:
        values = [batch[group] for batch in batches]
        if isinstance(values[0], dict):
            result[group] = {
                key: _pad_cat([value[key] for value in values])
                for key in values[0]
            }
        else:
            result[group] = values
    return result


def _state_inputs(policy: TCDPRGPolicy, observation) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    selected = policy.sensor_point_indices(observation.xyz, observation.point_valid)
    target = np.asarray(observation.target_mask[selected], bool)
    region = np.asarray(
        observation.region_target[selected]
        & observation.region_valid[selected]
        & observation.target_mask[selected],
        bool,
    )
    return policy._batch(observation), target, region


def _load_state_inputs(adapter, policy: TCDPRGPolicy, scene_id: int, state_id: int, task_index: int):
    return _state_inputs(
        policy,
        adapter.load_observation(scene_id, state_id, task_index),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.inference_batch_size < 1 or args.render_workers < 1 or args.prefetch_batches < 1:
        parser.error("batch size, render workers and prefetch batches must be positive")
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=True)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model = TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet).to(device)
    threshold = load_staged_tcd_prg(
        model, args.perception_checkpoint, args.stage_b_checkpoint, config
    )
    model.eval()
    policy = TCDPRGPolicy(model, config)
    scene_ids = tuple(args.scene_id or adapter.snapshot_scene_ids)
    output_root = Path(args.output_root)
    checkpoint_hash = _sha256(args.stage_b_checkpoint)
    render_hash = hashlib.sha256(
        json.dumps(
            {
                "renderer_version": config.observation.renderer_version,
                "camera_profile": config.observation.camera_profile,
                "width": config.observation.render_width,
                "height": config.observation.render_height,
                "scene_points": config.dataset.scene_points,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for scene_id in tqdm(scene_ids, desc="Stage-B state values", unit="scene"):
        output = output_root / f"scene_{scene_id:04d}.h5"
        if output.is_file() and not args.overwrite:
            continue
        label_path = adapter._path_by_scene[int(scene_id)]
        with h5py.File(label_path, "r", swmr=True) as handle:
            scene = handle[next(iter(handle.keys()))]
            task_indices = scene["states/task_index"][:].astype(np.int64)
        if args.max_states is not None:
            task_indices = task_indices[: args.max_states]
        graspability = np.full(len(task_indices), np.nan, np.float32)
        directly_graspable = np.zeros(len(task_indices), bool)
        valid = np.zeros(len(task_indices), bool)
        scene_started = time.perf_counter()
        load_seconds = 0.0
        inference_seconds = 0.0
        state_batches = range(0, len(task_indices), args.inference_batch_size)
        with ThreadPoolExecutor(max_workers=args.render_workers) as pool:
            futures = {
                state_id: pool.submit(
                    _load_state_inputs,
                    adapter,
                    policy,
                    int(scene_id),
                    state_id,
                    int(task_indices[state_id]),
                )
                for state_id in range(
                    min(
                        args.inference_batch_size * args.prefetch_batches,
                        len(task_indices),
                    )
                )
            }
            progress = tqdm(
                state_batches,
                total=(len(task_indices) + args.inference_batch_size - 1)
                // args.inference_batch_size,
                desc=f"scene {scene_id:04d}",
                unit="batch",
                leave=False,
            )
            for start in progress:
                stop = min(start + args.inference_batch_size, len(task_indices))
                load_started = time.perf_counter()
                prepared = [futures.pop(state_id).result() for state_id in range(start, stop)]
                prefetch_stop = min(
                    stop + args.inference_batch_size * args.prefetch_batches,
                    len(task_indices),
                )
                for state_id in range(stop, prefetch_stop):
                    if state_id in futures:
                        continue
                    futures[state_id] = pool.submit(
                        _load_state_inputs,
                        adapter,
                        policy,
                        int(scene_id),
                        state_id,
                        int(task_indices[state_id]),
                    )
                load_seconds += time.perf_counter() - load_started
                batch_load_seconds = time.perf_counter() - load_started

                visible = [index for index, (_, target, _) in enumerate(prepared) if target.any()]
                best_values = np.zeros(len(prepared), np.float32)
                if visible:
                    cpu_batch = _collate_batches([prepared[index][0] for index in visible])
                    point_count = cpu_batch["model_inputs"]["xyz"].shape[1]
                    target = torch.zeros((len(visible), point_count), dtype=torch.float32)
                    region = torch.zeros_like(target)
                    for row, index in enumerate(visible):
                        target_values, region_values = prepared[index][1:]
                        target[row, : len(target_values)] = torch.from_numpy(target_values)
                        region[row, : len(region_values)] = torch.from_numpy(region_values)
                    batch = _device(cpu_batch, device)
                    condition = StageBCondition(
                        target.to(device),
                        region.to(device),
                        torch.ones(len(visible), dtype=torch.bool, device=device),
                        batch["task_inputs"]["task_category_id"],
                        batch["task_inputs"]["task_region_id"],
                    ).validate(point_count)
                    inference_started = time.perf_counter()
                    with torch.inference_mode():
                        grasp = model.forward_task_grasp_from_condition(
                            model._sensor(batch), condition
                        )
                    inference_seconds += time.perf_counter() - inference_started
                    batch_inference_seconds = time.perf_counter() - inference_started
                    candidate_valid = grasp.get(
                        "valid",
                        torch.ones_like(grasp["task_valid_probability"], dtype=torch.bool),
                    ).bool()
                    probability = grasp["task_valid_probability"]
                    finite = candidate_valid & torch.isfinite(probability)
                    for row, index in enumerate(visible):
                        row_finite = finite[row]
                        if bool(row_finite.any()):
                            best_values[index] = float(probability[row][row_finite].max().cpu())

                graspability[start:stop] = best_values
                directly_graspable[start:stop] = best_values >= threshold
                valid[start:stop] = True
                progress.set_postfix(
                    load=f"{batch_load_seconds:.1f}s",
                    gpu=f"{batch_inference_seconds if visible else 0.0:.1f}s",
                )
        elapsed = time.perf_counter() - scene_started
        print(
            f"scene {scene_id:04d}: states={len(task_indices)} elapsed={elapsed:.1f}s "
            f"load={load_seconds:.1f}s gpu={inference_seconds:.1f}s "
            f"throughput={len(task_indices) / max(elapsed, 1e-6):.2f} states/s",
            flush=True,
        )
        write_state_values(
            output,
            StateValues(
                graspability, directly_graspable, valid, checkpoint_hash, render_hash,
                threshold,
            ),
        )


if __name__ == "__main__":
    main()
