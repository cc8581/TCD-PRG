from __future__ import annotations

from pathlib import Path

import torch
import yaml

import train
from tcd_prg.config import TCDPRGConfig
from tcd_prg.constants import ActionType
from tcd_prg.models import StandalonePushModel, resolve_staged_checkpoint_root
from tcd_prg.scripts.infer_state import requires_robot_certification
from tcd_prg.scripts.train import build_optimizer_parameter_groups


def _yaml_leaf_paths(value, prefix="") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}
    result: set[str] = set()
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        result.update(_yaml_leaf_paths(child, child_prefix))
    return result


def test_common_and_formal_stage_configs_have_no_duplicate_leaf_parameters() -> None:
    common = _yaml_leaf_paths(
        yaml.safe_load((train.PROJECT / "configs/config.yaml").read_text(encoding="utf-8"))
    )
    for path in train.STAGE_CONFIGS.values():
        stage = _yaml_leaf_paths(yaml.safe_load(path.read_text(encoding="utf-8"))) - {"defaults"}
        assert not common.intersection(stage), path


def test_stage_entrypoints_share_base_config_and_select_stage() -> None:
    for stage in ("perception", "grasp", "push"):
        args = train._parse_args(["--stage", stage])
        command = train._pipeline_command(args, stage, Path("outputs/test-stage"))
        assert command[command.index("--config") + 1].endswith(
            f"configs\\stage\\{stage}.yaml"
        )
        assert command[command.index("--stage") + 1] == stage
        assert f"training.stage={stage}" not in command
        assert args.stage == stage


def test_single_stage_explicit_config_is_respected() -> None:
    args = train._parse_args(["--stage", "grasp", "--config", "configs/overfit/grasp.yaml"])
    assert train._config_for_stage(args, "grasp").as_posix().endswith(
        "configs/overfit/grasp.yaml"
    )


def test_push_evaluator_command_uses_proposal_checkpoint_and_push_semantics(tmp_path) -> None:
    args = train._parse_args(["--stage", "all"])
    proposal = tmp_path / "push" / "push_last.pt"
    output = tmp_path / "push_evaluator" / "push_evaluator_best.pt"
    command = train._push_evaluator_command(args, proposal, output)
    assert command[1].endswith("train_push_evaluator.py")
    assert command[command.index("--proposal-checkpoint") + 1] == str(proposal.resolve())
    assert command[command.index("--output") + 1] == str(output.resolve())
    assert command[command.index("--config") + 1].endswith(
        "configs\\stage\\push_evaluator.yaml"
    )


def test_all_stage_launcher_runs_push_evaluator_after_stage_c(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    manifest = dataset / "task_training_labels" / "acronym_binary_grasps" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    args = train._parse_args(
        ["--stage", "all", "--output-dir", str(tmp_path / "out")]
    )
    monkeypatch.setattr(
        train,
        "_resolve_paths",
        lambda unused: (dataset, tmp_path / "acronym", tmp_path / "regions", "python", tmp_path / "cache"),
    )
    calls: list[list[str]] = []

    def fake_run(command, **unused):
        calls.append(command)
        if "--stage" in command:
            stage = command[command.index("--stage") + 1]
            filename = f"{stage}_last.pt" if stage == "push" else f"{stage}_best.pt"
            checkpoint = args.output_dir.resolve() / stage / filename
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
        elif "--output" in command:
            checkpoint = Path(command[command.index("--output") + 1])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    train._run_all_stages(args)
    assert len(calls) == 4
    assert calls[-1][1].endswith("train_push_evaluator.py")
    assert calls[-1][calls[-1].index("--proposal-checkpoint") + 1].endswith(
        "push\\push_last.pt"
    )
    manifest = yaml.safe_load((args.output_dir / "checkpoints.json").read_text())
    assert manifest["checkpoints"] == {
        "perception": "perception/perception_best.pt",
        "grasp": "grasp/grasp_best.pt",
        "push": "push/push_last.pt",
        "push_evaluator": "push_evaluator/push_evaluator_best.pt",
    }


def test_all_stage_launcher_rejects_single_stage_resume() -> None:
    args = train._parse_args(
        ["--stage", "all", "--resume", "outputs/old/last.pt"]
    )
    try:
        train._run_all_stages(args)
    except ValueError as error:
        assert "all-stage pipeline" in str(error)
    else:
        raise AssertionError("all-stage resume should be rejected")


def test_staged_checkpoint_root_resolves_portable_relative_layout(tmp_path) -> None:
    expected = {}
    for stage in ("perception", "grasp", "push", "push_evaluator"):
        path = tmp_path / stage / f"{stage}_best.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"checkpoint")
        expected[stage] = path.resolve()
    resolved = resolve_staged_checkpoint_root(tmp_path)
    assert resolved == expected


def test_stagec_optimizer_step_does_not_require_encoder() -> None:
    config = TCDPRGConfig()
    config.training.stage = "push"
    model = StandalonePushModel(config.model)
    groups = build_optimizer_parameter_groups(model, config)
    assert [group["name"] for group in groups] == ["new_modules"]
    optimizer = torch.optim.AdamW(groups)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_push_bypasses_grasp_robot_certifier() -> None:
    assert not requires_robot_certification({"action_type": int(ActionType.PUSH)})
    assert requires_robot_certification({"action_type": int(ActionType.TASK_GRASP)})
    assert requires_robot_certification({"action_type": int(ActionType.PICK_REMOVE)})
