"""Opinionated formal TCD-PRG training launcher for Windows and Linux.

Run from any directory with::

    python training.py

Extra Hydra/OmegaConf dot-list values are appended after the safe defaults, so
the caller can override any setting without editing this file. This launcher
never enables a dry-run and never starts rendering or data generation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
DEFAULT_CONFIG = PROJECT / "configs" / "config.yaml"
DEFAULT_PATHS_CONFIG = PROJECT / "configs" / "local_paths.yaml"

# Defaults sized for the current Windows RTX 3090 workstation. Command-line
# dot-list overrides are applied afterwards and therefore take precedence.
DEFAULT_OVERRIDES = (
    "backbone.backend=point_transformer_v3",
    "backbone.enable_flash_attention=false",
    "backbone.grid_size_m=0.005",
    "backbone.patch_size=256",
    "dataset.scene_points=16384",
    "training.device=cuda",
    "training.amp=true",
    "training.amp_dtype=float16",
    "training.batch_size=1",
    "training.gradient_accumulation_steps=1",
    "training.num_workers=4",
    "training.max_optimizer_steps=100000",
    "training.validation_interval=1000",
    "training.checkpoint_interval=1000",
    "logging.log_interval=20",
    "optimizer.learning_rate=0.0001",
    "optimizer.backbone_learning_rate=0.00002",
)


def _load_local_paths(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    import yaml

    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError(f"Local path config must be a string mapping: {path}")
    return values


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    local = _load_local_paths(args.paths_config.resolve())
    dataset = _project_relative_path(
        args.dataset_root
        or local.get(
            "dataset_root", PROJECT / "data" / "TaskOrientedClutterSceneDataset"
        )
    )
    acronym = _project_relative_path(
        args.acronym_root
        or local.get("acronym_root", PROJECT / "data" / "ACRONYM")
    )
    functional_region = _project_relative_path(
        args.functional_region_root
        or local.get("functional_region_root", PROJECT / "data" / "manual_function_regions_v1")
    )
    pybullet_python = str(
        args.pybullet_python or local.get("pybullet_python", sys.executable)
    )
    for name, path in (
        ("dataset", dataset),
        ("ACRONYM", acronym),
        ("functional regions", functional_region),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Configured {name} path does not exist: {path}. "
                f"Copy configs/local_paths.example.yaml to {args.paths_config} "
                "and fill in this machine's paths, or pass the matching --*-root option."
            )
    return dataset.resolve(), acronym.resolve(), functional_region.resolve(), pybullet_python


def _project_relative_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def _quoted_override(name: str, value: str | Path) -> str:
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start formal PTv3 TCD-PRG training with workstation defaults."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--acronym-root", type=Path)
    parser.add_argument("--functional-region-root", type=Path)
    parser.add_argument("--pybullet-python")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional key=value overrides; these take precedence over defaults.",
    )
    args = parser.parse_args()
    if args.gpus <= 0:
        parser.error("--gpus must be positive")
    if args.resume and args.initialize:
        parser.error("--resume and --initialize are mutually exclusive")
    return args


def _training_arguments(
    args: argparse.Namespace, path_overrides: tuple[str, ...] = ()
) -> list[str]:
    output = args.output_dir
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT / "outputs" / f"ptv3_full_{stamp}"
    output = output.resolve()
    arguments = ["--config", str(args.config.resolve())]
    if args.resume:
        arguments.extend(("--resume", str(args.resume.resolve())))
    if args.initialize:
        arguments.extend(("--initialize", str(args.initialize.resolve())))
    arguments.extend(
        (
            *DEFAULT_OVERRIDES,
            *path_overrides,
            _quoted_override("output_dir", output),
            *args.overrides,
        )
    )
    return arguments


def main() -> None:
    args = _parse_args()
    if args.gpus > 1:
        import torch

        available = torch.cuda.device_count()
        if available < args.gpus:
            raise RuntimeError(
                f"Requested {args.gpus} GPUs, but PyTorch detects only {available}. "
                "Multi-GPU training was not started."
            )
    dataset, acronym, functional_region, pybullet_python = _resolve_paths(args)
    ptv3_source = PROJECT / "third_party" / "PointTransformerV3" / "model.py"
    if not ptv3_source.is_file():
        raise FileNotFoundError(
            "Official PTv3 source is missing. Run: git submodule update --init "
            "third_party/PointTransformerV3"
        )
    path_overrides = (
        _quoted_override("dataset.root", dataset),
        _quoted_override("dataset.acronym_root", acronym),
        _quoted_override("dataset.functional_region_root", functional_region),
        _quoted_override("observation.pybullet_python", pybullet_python),
    )
    training_args = _training_arguments(args, path_overrides)
    print("TCD-PRG formal training", flush=True)
    print(f"  platform={sys.platform} gpus={args.gpus}", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  acronym={acronym}", flush=True)
    print(f"  functional_regions={functional_region}", flush=True)
    print(f"  pybullet_python={pybullet_python}", flush=True)
    print(f"  config={args.config.resolve()}", flush=True)
    print(
        "  backbone=PointTransformerV3 flash=false points=16384 "
        "batch=1 accumulation=1",
        flush=True,
    )
    os.chdir(PROJECT)
    if args.gpus == 1:
        from tcd_prg.scripts.train import main as train_main

        sys.argv = ["tcd-prg-train", *training_args]
        train_main()
        return
    if sys.platform == "win32":
        command = [
            sys.executable,
            str(PROJECT / "scripts" / "launch_ddp_windows.py"),
            "--nproc-per-node",
            str(args.gpus),
            *training_args,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.gpus}",
            "-m",
            "tcd_prg.scripts.train",
            *training_args,
        ]
    subprocess.run(command, check=True, cwd=PROJECT)


if __name__ == "__main__":
    main()
