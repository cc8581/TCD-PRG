from __future__ import annotations

from pathlib import Path

import train


def test_stage_entrypoints_share_base_config_and_select_stage() -> None:
    for stage in ("perception", "grasp", "push"):
        args = train._parse_args(["--stage", stage, "--config", "configs/config.yaml"])
        command = train._pipeline_command(args, stage, Path("outputs/test-stage"))
        assert command[command.index("--config") + 1].endswith("configs\\config.yaml")
        assert command[command.index("--stage") + 1] == stage
        assert f"training.stage={stage}" not in command
        assert args.stage == stage


def test_all_stage_launcher_rejects_single_stage_resume() -> None:
    args = train._parse_args(
        ["--stage", "all", "--resume", "outputs/old/last.pt", "--config", "configs/config.yaml"]
    )
    try:
        train._run_all_stages(args)
    except ValueError as error:
        assert "all-stage pipeline" in str(error)
    else:
        raise AssertionError("all-stage resume should be rejected")
