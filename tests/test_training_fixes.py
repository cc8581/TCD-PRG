"""Regression tests for the training-fix patch."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tcd_prg.config import BackboneConfig
from tcd_prg.datasets.torch_dataset import _deterministic_fraction_indices
from tcd_prg.pretrained import load_pretrained_backbone, prepare_pretrained_checkpoint


def test_data_fraction_is_deterministic_and_exact() -> None:
    first = _deterministic_fraction_indices(100, 0.5, 2026)
    second = _deterministic_fraction_indices(100, 0.5, 2026)
    different = _deterministic_fraction_indices(100, 0.5, 2027)
    assert len(first) == 50
    assert torch.equal(torch.as_tensor(first), torch.as_tensor(second))
    assert not torch.equal(torch.as_tensor(first), torch.as_tensor(different))


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        trunk = nn.Module()
        trunk.enc = nn.Linear(4, 4)
        trunk.dec = nn.Linear(4, 4)
        scene = nn.Module()
        scene.backbone = trunk
        encoder = nn.Module()
        encoder.scene_backbone = scene
        self.encoder = encoder


def test_sonata_loader_shape_matches_encoder_only(tmp_path) -> None:
    model = _FakeModel()
    checkpoint = tmp_path / "sonata.pth"
    torch.save({
        "state_dict": {
            "enc.weight": torch.full_like(model.encoder.scene_backbone.backbone.enc.weight, 2),
            "enc.bias": torch.full_like(model.encoder.scene_backbone.backbone.enc.bias, 3),
            "embedding.weight": torch.zeros(9, 9),
        }
    }, checkpoint)
    config = BackboneConfig(
        pretrained_format="sonata",
        pretrained_min_parameter_fraction=0.4,
    )
    report = load_pretrained_backbone(model, checkpoint, config)
    assert report["matched_tensors"] == 2
    assert report["freeze_prefixes"] == [
        "encoder.scene_backbone.backbone.enc.bias",
        "encoder.scene_backbone.backbone.enc.weight",
    ]
    assert torch.all(model.encoder.scene_backbone.backbone.enc.weight == 2)
    assert torch.all(model.encoder.scene_backbone.backbone.enc.bias == 3)


def test_checksum_failure_does_not_delete_explicit_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "user-owned.pth"
    checkpoint.write_bytes(b"not-the-configured-checkpoint")
    config = BackboneConfig(
        pretrained_checkpoint=str(checkpoint),
        pretrained_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        prepare_pretrained_checkpoint(config, allow_download=False)
    assert checkpoint.read_bytes() == b"not-the-configured-checkpoint"
