"""TCD-PRG training launcher for Windows and Linux.

Run from any directory with::

    python train.py

Named command-line options override the YAML only when explicitly supplied.
This launcher never enables a dry-run and never starts data generation.
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
# 公共配置只保存可移植参数；每台机器的数据盘路径写在被 Git 忽略的 local_paths.yaml 中。
DEFAULT_CONFIG = PROJECT / "configs" / "config.yaml"
DEFAULT_PATHS_CONFIG = PROJECT / "configs" / "local_paths.yaml"

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


def _launcher_defaults(paths_config: Path) -> dict[str, str | Path | None]:
    """Build visible launcher defaults without committing machine-specific paths."""

    local = _load_local_paths(paths_config.resolve())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 路径优先级：显式命令行参数 > local_paths.yaml > 仓库内约定目录。
    return {
        "dataset_root": _project_relative_path(
            local.get("dataset_root", PROJECT / "data" / "TaskOrientedClutterSceneDataset")
        ),
        "acronym_root": _project_relative_path(
            local.get("acronym_root", PROJECT / "data" / "ACRONYM")
        ),
        "functional_region_root": _project_relative_path(
            local.get(
                "functional_region_root",
                PROJECT / "data" / "manual_function_regions",
            )
        ),
        "pybullet_python": local.get("pybullet_python", sys.executable),
        "observation_cache_dir": _project_relative_path(
            local.get("observation_cache_dir", PROJECT / "runtime" / "cache" / "observations")
        ),
        "gpus": 1,
        "output_dir": PROJECT / "outputs" / f"ptv3_full_{stamp}",
        "resume": None,
        "initialize": None,
    }


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str, Path]:
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
        or local.get("functional_region_root", PROJECT / "data" / "manual_function_regions")
    )
    pybullet_python = str(
        args.pybullet_python or local.get("pybullet_python", sys.executable)
    )
    observation_cache = _project_relative_path(
        args.observation_cache_dir
        or local.get("observation_cache_dir", PROJECT / "runtime" / "cache" / "observations")
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
    return (
        dataset.resolve(), acronym.resolve(), functional_region.resolve(),
        pybullet_python, observation_cache.resolve(),
    )


def _project_relative_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def _quoted_override(name: str, value: str | Path) -> str:
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 先只读取路径配置位置，使自定义 --paths-config 也能参与其余参数的默认值计算。
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    defaults = _launcher_defaults(bootstrap_args.paths_config)
    parser = argparse.ArgumentParser(
        description="Start TCD-PRG training using YAML configuration defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    # 下面这些值均可直接覆盖；不传时使用 local_paths.yaml 中的本机配置。
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None,
        help="Override training.batch_size.",
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=None,
        help="Override training.num_workers.",
    )
    parser.add_argument(
        "--validation-num-workers", "--validation_num_workers",
        dest="validation_num_workers", type=int, default=None,
        help="Override training.validation_num_workers (0 avoids a validation worker pool).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps", type=int, default=None,
        help="Override training.gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--max-optimizer-steps", "--max_optimizer_steps",
        dest="max_optimizer_steps", type=int, default=None,
        help="Override training.max_optimizer_steps.",
    )
    parser.add_argument(
        "--validation-interval", "--validation_interval",
        dest="validation_interval", type=int, default=None,
        help="Override training.validation_interval.",
    )
    parser.add_argument("--dataset-root", type=Path, default=defaults["dataset_root"], help="Root directory of the task-oriented scene dataset.")
    parser.add_argument("--acronym-root", type=Path, default=defaults["acronym_root"], help="Root directory of the ACRONYM grasp dataset.")
    parser.add_argument("--functional-region-root", type=Path, default=defaults["functional_region_root"], help="Root directory of the manual functional-region annotations.")
    parser.add_argument("--pybullet-python", default=defaults["pybullet_python"], help="Python interpreter used by the PyBullet compatibility workers.")
    parser.add_argument("--observation-cache-dir", type=Path, default=defaults["observation_cache_dir"], help="Content-addressed observation cache directory.")
    parser.add_argument("--gpus", type=int, default=defaults["gpus"], help="Number of local training GPUs.")
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"], help="Directory for checkpoints, JSONL metrics, and TensorBoard logs.")
    # resume 恢复优化器/调度器等完整状态；initialize 仅加载模型权重，二者不可同时使用。
    parser.add_argument("--resume", type=Path, default=defaults["resume"], help="Checkpoint used to resume the complete training state.")
    parser.add_argument("--initialize", type=Path, default=defaults["initialize"], help="Checkpoint used only to initialize model weights.")
    parser.add_argument(
        "--data-fraction", type=float, default=None,
        help="Override training.data_fraction with a deterministic fraction in (0, 1].",
    )
    args = parser.parse_args(argv)
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
    arguments.extend(path_overrides)
    arguments.append(_quoted_override("output_dir", output))
    named_training_overrides = {
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "validation_num_workers": "validation_num_workers",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "max_optimizer_steps": "max_optimizer_steps",
        "validation_interval": "validation_interval",
        "data_fraction": "data_fraction",
    }
    for argument_name, config_name in named_training_overrides.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            arguments.append(f"training.{config_name}={value:g}")
    return arguments


def main() -> None:
    args = _parse_args()
    # 在创建 DDP 子进程前先核对显卡数量，避免部分 worker 启动后才失败。
    if args.gpus > 1:
        import torch

        available = torch.cuda.device_count()
        if available < args.gpus:
            raise RuntimeError(
                f"Requested {args.gpus} GPUs, but PyTorch detects only {available}. "
                "Multi-GPU training was not started."
            )
    dataset, acronym, functional_region, pybullet_python, observation_cache = _resolve_paths(args)
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
        _quoted_override("cache.directory", observation_cache),
    )
    training_args = _training_arguments(args, path_overrides)
    print("TCD-PRG formal training", flush=True)
    print(f"  platform={sys.platform} gpus={args.gpus}", flush=True)
    print(f"  dataset={dataset}", flush=True)
    print(f"  acronym={acronym}", flush=True)
    print(f"  functional_regions={functional_region}", flush=True)
    print(f"  pybullet_python={pybullet_python}", flush=True)
    print(f"  config={args.config.resolve()}", flush=True)
    os.chdir(PROJECT)
    # 单卡直接进入训练器；Windows 多卡使用 spawn，Linux 使用 torchrun。
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
