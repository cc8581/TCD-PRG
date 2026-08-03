"""Opinionated formal TCD-PRG training launcher for Windows and Linux.

Run from any directory with::

    python scripts/start_training.py

Extra Hydra/OmegaConf dot-list values are appended after the safe defaults, so
the caller can override any setting without editing this file. This launcher
never enables a dry-run and never starts rendering or data generation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "config.yaml"
DEFAULT_GAPG_PYTHON = Path(r"D:\Anaconda\envs\gapg\python.exe")

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
    "training.gradient_accumulation_steps=8",
    "training.num_workers=4",
    "training.max_optimizer_steps=100000",
    "training.validation_interval=1000",
    "training.checkpoint_interval=1000",
    "logging.log_interval=20",
    "optimizer.learning_rate=0.0001",
    "optimizer.backbone_learning_rate=0.00002",
)


def _discover_data_paths() -> tuple[Path, Path, Path]:
    configured = os.environ.get("TCD_DATASET_ROOT")
    if configured:
        dataset = Path(configured)
        grasp_root = dataset.parents[1]
    else:
        cc_root = Path(r"G:\cc")
        candidates = []
        if cc_root.is_dir():
            candidates = [
                child
                for child in cc_root.iterdir()
                if (child / "self-built-task-oriented-clusster-sense-dataset").is_dir()
            ]
        if not candidates:
            raise FileNotFoundError(
                "Cannot locate the generated dataset. Set TCD_DATASET_ROOT, "
                "TCD_ACRONYM_ROOT and TCD_FUNCTIONAL_REGION_ROOT."
            )
        grasp_root = candidates[0]
        dataset = (
            grasp_root
            / "self-built-task-oriented-clusster-sense-dataset"
            / "TaskOrientedClutterSceneDataset"
        )
    source_root = dataset.parent
    acronym = Path(os.environ.get("TCD_ACRONYM_ROOT", grasp_root / "ACRONYM"))
    functional_region = Path(
        os.environ.get(
            "TCD_FUNCTIONAL_REGION_ROOT",
            source_root
            / "Grasp_20_class_object_3D_model"
            / "data"
            / "manual_function_regions_v1",
        )
    )
    for name, path in (
        ("dataset", dataset),
        ("ACRONYM", acronym),
        ("functional regions", functional_region),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Configured {name} path does not exist: {path}")
    return dataset, acronym, functional_region


def _configure_environment() -> tuple[Path, Path, Path]:
    dataset, acronym, functional_region = _discover_data_paths()
    os.environ["TCD_DATASET_ROOT"] = str(dataset)
    os.environ["TCD_ACRONYM_ROOT"] = str(acronym)
    os.environ["TCD_FUNCTIONAL_REGION_ROOT"] = str(functional_region)
    if "TCD_PYBULLET_PYTHON" not in os.environ and DEFAULT_GAPG_PYTHON.is_file():
        os.environ["TCD_PYBULLET_PYTHON"] = str(DEFAULT_GAPG_PYTHON)
    return dataset, acronym, functional_region


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start formal PTv3 TCD-PRG training with workstation defaults."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def _training_arguments(args: argparse.Namespace) -> list[str]:
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
    arguments.extend((*DEFAULT_OVERRIDES, f"output_dir={output}", *args.overrides))
    return arguments


def main() -> None:
    args = _parse_args()
    dataset, acronym, functional_region = _configure_environment()
    ptv3_source = PROJECT / "third_party" / "PointTransformerV3" / "model.py"
    if not ptv3_source.is_file():
        raise FileNotFoundError(
            "Official PTv3 source is missing. Run: git submodule update --init "
            "third_party/PointTransformerV3"
        )
    training_args = _training_arguments(args)
    print("TCD-PRG formal training", flush=True)
    print(f"  platform={sys.platform} gpus={args.gpus}", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  acronym={acronym}", flush=True)
    print(f"  functional_regions={functional_region}", flush=True)
    print(f"  config={args.config.resolve()}", flush=True)
    print(
        "  backbone=PointTransformerV3 flash=false points=16384 "
        "batch=1 accumulation=8",
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
    subprocess.run(command, check=True, cwd=PROJECT, env=os.environ.copy())


if __name__ == "__main__":
    main()
