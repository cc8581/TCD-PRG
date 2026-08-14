#!/usr/bin/env python3
"""Record an externally measured tabletop plane in the experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from real_experiment_app.config import AppConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persist a tabletop normal and one tabletop point in robot-base metres."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "real_experiment.yaml",
    )
    parser.add_argument("--normal", type=float, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--point-m", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    config.record_table_plane(args.normal, args.point_m)
    plane = config.raw["fusion"]["table_plane_base"]
    print(f"Recorded table plane in {config.path}: {plane}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
