from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def fake_graspnet(monkeypatch):
    """Deterministic CPU contract double for full-model unit tests."""
    from tcd_prg.models.graspnet import FrozenGraspNetProposalGenerator

    def forward(
        self,
        xyz,
        point_mask,
        *,
        importance=None,
        instance_probability=None,
        proposal_count=None,
        input_points=None,
    ):
        del importance, input_points
        batch, points, _ = xyz.shape
        count = int(proposal_count or self.proposal_count)
        base = torch.arange(count, device=xyz.device) % points
        index = base[None].expand(batch, -1)
        rows = torch.arange(batch, device=xyz.device)[:, None]
        valid = point_mask[rows, index].bool()
        translation = xyz[rows, index]
        rotation = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
        rotation = rotation.reshape(1, 1, 3, 3).expand(batch, count, -1, -1)
        score = torch.linspace(0.8, 0.2, count, device=xyz.device, dtype=xyz.dtype)
        score = score[None].expand(batch, -1)
        if instance_probability is None:
            object_logits = xyz.new_zeros((batch, count, 1))
        else:
            object_logits = instance_probability[rows, :, index].clamp_min(1e-6).log()
        return {
            "translation_world": translation,
            "rotation_matrix": rotation,
            "width_m": xyz.new_full((batch, count), 0.04),
            "depth_m": xyz.new_full((batch, count), 0.02),
            "quality_logit": torch.logit(score.clamp(1e-5, 1 - 1e-5)),
            "graspnet_score": score,
            "attention_point_index": index,
            "object_logits": object_logits,
            "valid": valid,
        }

    monkeypatch.setattr(FrozenGraspNetProposalGenerator, "forward", forward)


@pytest.fixture
def dataset_root() -> Path:
    import yaml

    local_paths = Path(__file__).parents[1] / "configs" / "local_paths.yaml"
    values = (
        yaml.safe_load(local_paths.read_text(encoding="utf-8"))
        if local_paths.is_file()
        else {}
    )
    value = values.get("dataset_root") if isinstance(values, dict) else None
    if not value or not Path(value).exists():
        pytest.skip("Configure dataset_root in configs/local_paths.yaml")
    return Path(value)


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    b, n, o = 1, 24, 3
    xyz = torch.randn(b, n, 3)
    instance = torch.arange(n)[None] % o
    relation = torch.zeros(b, o, o, 5)
    relation[:, 0, 1, 2] = 1
    return {
        "xyz": xyz,
        "rgb": torch.rand(b, n, 3),
        "source_view": torch.full((b, n), 2, dtype=torch.long),
        "graspnet_xyz_world": xyz.clone(),
        "graspnet_point_mask": torch.ones(b, n, dtype=torch.bool),
        "camera2_eye_world": torch.zeros(b, 3),
        "camera2_target_world": torch.tensor([[0.0, 0.0, 1.0]]),
        "camera2_up_world": torch.tensor([[0.0, -1.0, 0.0]]),
        "camera2_valid": torch.ones(b, dtype=torch.bool),
        "instance_id": instance,
        "point_mask": torch.ones(b, n, dtype=torch.bool),
        "target_mask": instance == 1,
        "target_object": torch.tensor([1]),
        "object_pose": torch.cat(
            (
                torch.randn(b, o, 3),
                torch.tensor([0, 0, 0, 1.0]).repeat(b, o, 1),
            ),
            -1,
        ),
        "object_mask": torch.ones(b, o, dtype=torch.bool),
        "object_active": torch.ones(b, o, dtype=torch.bool),
        "task_category_id": torch.tensor([1]),
        "task_region_id": torch.tensor([2]),
        "relation_graph": relation,
        "remaining_steps": torch.tensor([5]),
    }
