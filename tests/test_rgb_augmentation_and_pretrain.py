from __future__ import annotations

import copy

import pytest
import torch

from tcd_prg.config import (
    AugmentationConfig,
    AugmentationMethodConfig,
    ColorJitterConfig,
    ObjectRecolorConfig,
)
from tcd_prg.datasets.rgb_augmentation import PointCloudRGBAugmentation
from tcd_prg.scripts.train import load_pretrain_checkpoint, validate_checkpoint_gate


def _batch() -> dict[str, torch.Tensor]:
    return {
        "xyz": torch.arange(24, dtype=torch.float32).reshape(1, 8, 3),
        "rgb": torch.linspace(0.05, 0.95, 24).reshape(1, 8, 3),
        "point_mask": torch.tensor([[True, True, True, True, True, True, False, False]]),
        "instance_id": torch.tensor([[0, 0, 1, 1, 2, 2, -1, -1]]),
        "target_mask": torch.tensor([[True, True, False, False, False, False, False, False]]),
    }


def test_rgb_augmentation_changes_only_rgb_and_preserves_padding() -> None:
    config = AugmentationConfig(
        zero_rgb=AugmentationMethodConfig(probability=0.0),
        grayscale=AugmentationMethodConfig(probability=0.0),
        color_jitter=ColorJitterConfig(probability=1.0),
        object_recolor=ObjectRecolorConfig(probability=1.0),
        material_jitter=AugmentationMethodConfig(probability=0.0),
        lighting_jitter=AugmentationMethodConfig(probability=0.0),
        sensor_noise=AugmentationMethodConfig(probability=0.0),
        channel_dropout=AugmentationMethodConfig(probability=0.0),
        point_dropout=AugmentationMethodConfig(probability=0.0),
    )
    batch = _batch()
    original = {name: value.clone() for name, value in batch.items()}
    torch.manual_seed(7)
    PointCloudRGBAugmentation(config)(batch)
    assert not torch.equal(batch["rgb"][:, :6], original["rgb"][:, :6])
    assert torch.equal(batch["rgb"][:, 6:], torch.zeros_like(batch["rgb"][:, 6:]))
    for name in ("xyz", "point_mask", "instance_id", "target_mask"):
        assert torch.equal(batch[name], original[name])


def test_rgb_zero_mode_keeps_three_channels() -> None:
    config = AugmentationConfig(
        zero_rgb=AugmentationMethodConfig(probability=1.0),
        grayscale=AugmentationMethodConfig(probability=0.0),
        color_jitter=ColorJitterConfig(probability=0.0),
        object_recolor=ObjectRecolorConfig(probability=0.0),
        material_jitter=AugmentationMethodConfig(probability=0.0),
        lighting_jitter=AugmentationMethodConfig(probability=0.0),
        sensor_noise=AugmentationMethodConfig(probability=0.0),
        channel_dropout=AugmentationMethodConfig(probability=0.0),
        point_dropout=AugmentationMethodConfig(probability=0.0),
    )
    batch = _batch()
    PointCloudRGBAugmentation(config)(batch)
    assert batch["rgb"].shape == (1, 8, 3)
    assert torch.count_nonzero(batch["rgb"]) == 0


def test_zero_probabilities_leave_rgb_unchanged() -> None:
    config = AugmentationConfig(
        zero_rgb=AugmentationMethodConfig(probability=0.0),
        grayscale=AugmentationMethodConfig(probability=0.0),
        color_jitter=ColorJitterConfig(probability=0.0),
        object_recolor=ObjectRecolorConfig(probability=0.0),
        material_jitter=AugmentationMethodConfig(probability=0.0),
        lighting_jitter=AugmentationMethodConfig(probability=0.0),
        sensor_noise=AugmentationMethodConfig(probability=0.0),
        channel_dropout=AugmentationMethodConfig(probability=0.0),
        point_dropout=AugmentationMethodConfig(probability=0.0),
    )
    batch = _batch()
    original = batch["rgb"].clone()
    PointCloudRGBAugmentation(config)(batch)
    assert torch.equal(batch["rgb"][:, :6], original[:, :6])
    assert torch.count_nonzero(batch["rgb"][:, 6:]) == 0


def test_pretrain_load_is_strict_and_does_not_create_trainer_state() -> None:
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    target = copy.deepcopy(source)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(2.0)
    payload = {
        "schema_version": 12,
        "training_stage": "perception",
        "model": source.state_dict(),
        "optimizer": {"sentinel": "must not be consumed"},
        "trainer_state": {"optimizer_steps": 1234},
    }
    validate_checkpoint_gate("perception", resume_payload=None, pretrain_payload=payload)
    load_pretrain_checkpoint(target, payload, "perception")
    pairs = zip(source.state_dict().values(), target.state_dict().values(), strict=True)
    for source_value, target_value in pairs:
        assert torch.equal(source_value, target_value)


def test_pretrain_rejects_stage_and_structure_mismatch() -> None:
    payload = {"schema_version": 12, "training_stage": "grasp", "model": {}}
    with pytest.raises(RuntimeError, match="requires a 'perception' checkpoint"):
        validate_checkpoint_gate("perception", resume_payload=None, pretrain_payload=payload)
    with pytest.raises(RuntimeError, match="parameter mismatch"):
        load_pretrain_checkpoint(torch.nn.Linear(2, 2), payload, "grasp")
