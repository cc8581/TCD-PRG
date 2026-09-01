"""Stream observations through frozen Stage-A/B and persist compact state values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.datasets.push_value import StateValues, write_state_values
from tcd_prg.models import TCDPRGModel, load_staged_tcd_prg
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--perception-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
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
        graspability = np.full(len(task_indices), np.nan, np.float32)
        directly_graspable = np.zeros(len(task_indices), bool)
        valid = np.zeros(len(task_indices), bool)
        for state_id, task_index in enumerate(task_indices):
            observation = adapter.load_observation(int(scene_id), state_id, int(task_index))
            batch = _device(policy._batch(observation), device)
            with torch.no_grad():
                perception = model.forward_perception(batch)
                grasp = model.forward_task_grasp_from_condition(
                    perception["sensor"], perception["stageb_condition"]
                )
            candidate_valid = grasp.get(
                "valid", torch.ones_like(grasp["task_valid_probability"], dtype=torch.bool)
            )[0].bool()
            probability = grasp["task_valid_probability"][0]
            target_valid = bool(perception["stageb_condition"].target_valid[0])
            finite = candidate_valid & torch.isfinite(probability)
            if target_valid:
                best = float(probability[finite].max().cpu()) if bool(finite.any()) else 0.0
                graspability[state_id] = best
                directly_graspable[state_id] = best >= threshold
                valid[state_id] = True
        write_state_values(
            output,
            StateValues(
                graspability, directly_graspable, valid, checkpoint_hash, render_hash,
                threshold,
            ),
        )


if __name__ == "__main__":
    main()
