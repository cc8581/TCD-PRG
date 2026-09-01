"""Build finite-horizon Stage-C action targets from state-value sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.datasets.push_value import build_action_value_sidecar
from tcd_prg.runtime import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--state-value-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", type=int, action="append")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    scene_ids = tuple(args.scene_id or adapter.snapshot_scene_ids)
    state_root, output_root = Path(args.state_value_root), Path(args.output_root)
    for scene_id in tqdm(scene_ids, desc="PUSH action values", unit="scene"):
        output = output_root / f"scene_{scene_id:04d}.h5"
        if output.is_file() and not args.overwrite:
            continue
        build_action_value_sidecar(
            adapter._path_by_scene[int(scene_id)],
            state_root / f"scene_{scene_id:04d}.h5",
            output,
            gamma=args.gamma,
            horizons=config.training.push_value_horizons,
        )


if __name__ == "__main__":
    main()
