from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import train
from scripts.launch_ddp_windows import _worker
from train import PROJECT, _parse_args, _training_arguments


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
    assert args.output_dir.name.startswith("formal_")
    assert args.resume is None
    assert args.pretrain_checkpoint is None
    assert args.batch_size is None
    assert args.num_workers is None
    assert args.validation_num_workers is None
    assert args.gradient_accumulation_steps is None
    assert args.max_optimizer_steps is None
    assert args.validation_interval is None
    assert args.data_fraction is None


def test_formal_launcher_defaults_and_user_override_order(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    arguments = _training_arguments(Namespace(
        config=config,
        output_dir=tmp_path / "output",
        resume=None,
        data_fraction=None,
    ))
    assert not any(argument.startswith("backbone.") for argument in arguments)
    assert not any(argument.startswith("training.") for argument in arguments)
    assert not any(argument.startswith("cache.") for argument in arguments)
    assert arguments[-1].startswith("output_dir=")
    assert not any("dry" in argument.lower() for argument in arguments)


def test_launcher_data_fraction_flag_is_forwarded(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    args = _parse_args([
        "--stage", "perception", "--config", str(config), "--data-fraction", "0.25",
    ])
    arguments = _training_arguments(args)
    assert "training.data_fraction=0.25" in arguments


def test_launcher_forwards_pretrain_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    args = _parse_args([
        "--stage", "perception", "--pretrain-checkpoint", str(checkpoint),
    ])
    arguments = _training_arguments(args)
    assert arguments[arguments.index("--pretrain-checkpoint") + 1] == str(
        checkpoint.resolve()
    )


def test_launcher_only_forwards_explicit_named_training_overrides(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    args = _parse_args([
        "--stage", "perception", "--config", str(config),
        "--batch-size", "3", "--max-optimizer-steps", "17",
    ])
    arguments = _training_arguments(args)
    assert "training.batch_size=3" in arguments
    assert "training.max_optimizer_steps=17" in arguments
    assert not any(argument.startswith("training.num_workers=") for argument in arguments)


def test_launcher_forwards_validation_worker_override(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    args = _parse_args([
        "--stage", "perception", "--config", str(config),
        "--validation-num-workers", "1",
    ])
    arguments = _training_arguments(args)
    assert "training.validation_num_workers=1" in arguments


def test_all_stage_pipeline_rejects_missing_stageb_data_before_launch(tmp_path) -> None:
    dataset = tmp_path / "empty-dataset"
    acronym = tmp_path / "empty-acronym"
    regions = tmp_path / "empty-regions"
    dataset.mkdir()
    acronym.mkdir()
    regions.mkdir()
    args = _parse_args([
        "--stage", "all",
        "--dataset-root", str(dataset),
        "--acronym-root", str(acronym),
        "--functional-region-root", str(regions),
        "--stageb-binary-root", str(tmp_path / "missing-stageb"),
        "--output-dir", str(tmp_path / "output"),
    ])
    try:
        train._run_all_stages(args)
    except FileNotFoundError as error:
        assert "requires the ACRONYM object grasp database" in str(error)
    else:
        raise AssertionError("all-stage launch must fail before Stage A when Stage B is absent")


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
