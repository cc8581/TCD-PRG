"""Pre-render observations and exact gripper geometries before GPU training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import numpy as np
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.runtime import create_adapter, create_gripper_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=True)
    unit_iterator = adapter.iter_action_groups(args.split)
    units = list(islice(unit_iterator, args.max_groups)) if args.max_groups else list(unit_iterator)
    unique_states = sorted({unit[:3] for unit in units})
    widths = []
    for scene_id, _, _, group_index in tqdm(units, desc="scan grasp widths"):
        group = adapter.load_action_group(scene_id, group_index)
        value = group.action_parameters["grasp_width_m"]
        widths.extend(value[np.isfinite(value)].tolist())
    provider = create_gripper_provider(config, allow_generate=True)
    provider.prewarm(np.asarray(widths, dtype=np.float32))

    def render(unit: tuple[int, int, int]):
        return adapter.load_observation(*unit).xyz.shape[0]

    with ThreadPoolExecutor(max_workers=config.cache.prefetch_workers) as pool:
        futures = {pool.submit(render, unit): unit for unit in unique_states}
        for future in tqdm(as_completed(futures), total=len(futures), desc="render states"):
            future.result()
    print(
        f"prefetch complete: groups={len(units)}, states={len(unique_states)}, "
        f"gripper_widths={len(set(round(x, 7) for x in widths))}"
    )


if __name__ == "__main__":
    main()
