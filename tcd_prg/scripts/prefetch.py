"""Pre-render observations and exact gripper geometries before GPU training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.runtime import create_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--max-groups", type=int, required=True,
        help="Bounded hot-set size; full-dataset prefetch is intentionally unsupported.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=True)
    unit_iterator = adapter.iter_action_groups(args.split)
    units = list(islice(unit_iterator, args.max_groups))
    unique_states = sorted({unit[:3] for unit in units})

    def render(unit: tuple[int, int, int]):
        return adapter.load_observation(*unit).xyz.shape[0]

    with ThreadPoolExecutor(max_workers=config.cache.prefetch_workers) as pool:
        futures = {pool.submit(render, unit): unit for unit in unique_states}
        for future in tqdm(as_completed(futures), total=len(futures), desc="render states"):
            future.result()
    print(
        f"prefetch complete: groups={len(units)}, states={len(unique_states)}"
    )


if __name__ == "__main__":
    main()
