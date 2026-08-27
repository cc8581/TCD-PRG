from __future__ import annotations

from pathlib import Path
import torch

import train
from tcd_prg.config import TCDPRGConfig
from tcd_prg.models import StandalonePushModel
from tcd_prg.scripts.train import build_optimizer_parameter_groups


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


def test_stagec_optimizer_step_does_not_require_encoder() -> None:
    config = TCDPRGConfig(); config.training.stage = "push"
    model = StandalonePushModel(config.model)
    groups = build_optimizer_parameter_groups(model, config)
    assert [group["name"] for group in groups] == ["new_modules"]
    optimizer = torch.optim.AdamW(groups)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward(); optimizer.step()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
