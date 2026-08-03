from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

from scripts.launch_ddp_windows import _worker
from train import DEFAULT_OVERRIDES, PROJECT, _parse_args, _training_arguments


def test_launcher_path_arguments_have_local_config_defaults(tmp_path) -> None:
    paths = tmp_path / "local_paths.yaml"
    paths.write_text(
        "dataset_root: D:/datasets/scenes\n"
        "acronym_root: D:/datasets/acronym\n"
        "functional_region_root: D:/datasets/regions\n"
        "pybullet_python: D:/envs/gapg/python.exe\n",
        encoding="utf-8",
    )

    args = _parse_args(["--paths-config", str(paths)])

    assert args.dataset_root == Path("D:/datasets/scenes")
    assert args.acronym_root == Path("D:/datasets/acronym")
    assert args.functional_region_root == Path("D:/datasets/regions")
    assert args.pybullet_python == "D:/envs/gapg/python.exe"
    assert args.gpus == 1
    assert args.output_dir.parent == PROJECT / "outputs"
    assert args.output_dir.name.startswith("ptv3_full_")
    assert args.resume is None
    assert args.initialize is None


def test_formal_launcher_defaults_and_user_override_order(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    arguments = _training_arguments(Namespace(
        config=config,
        output_dir=tmp_path / "output",
        resume=None,
        initialize=None,
        overrides=["training.batch_size=2"],
    ))
    assert "backbone.backend=point_transformer_v3" in DEFAULT_OVERRIDES
    assert "backbone.enable_flash_attention=false" in arguments
    assert "dataset.scene_points=16384" in arguments
    assert "training.gradient_accumulation_steps=1" in arguments
    assert arguments[-1] == "training.batch_size=2"
    assert not any("dry" in argument.lower() for argument in arguments)


def test_windows_worker_passes_ddp_state_explicitly(monkeypatch) -> None:
    called = []
    monkeypatch.setattr("tcd_prg.scripts.train.main", lambda: called.append(tuple(sys.argv)))
    before = {
        name: os.environ.get(name)
        for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "TCD_DDP_INIT_METHOD")
    }
    _worker(1, 2, "file:///temporary/store", ["--config", "config.yaml"])
    arguments = called[0]
    assert arguments[arguments.index("--world-size") + 1] == "2"
    assert arguments[arguments.index("--rank") + 1] == "1"
    assert arguments[arguments.index("--local-rank") + 1] == "1"
    assert arguments[arguments.index("--ddp-init-method") + 1] == "file:///temporary/store"
    assert before == {
        name: os.environ.get(name)
        for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "TCD_DDP_INIT_METHOD")
    }


def test_launchers_resolve_repository_without_editable_install() -> None:
    assert str(PROJECT) in sys.path
