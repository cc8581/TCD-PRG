from __future__ import annotations

import copy
import json

import pytest
import torch

from tcd_prg.config import (
    AugmentationConfig,
    AugmentationMethodConfig,
    ColorJitterConfig,
    ObjectRecolorConfig,
)
from tcd_prg.datasets.augmentation_debug import claim_debug_batch, save_debug_batch
from tcd_prg.datasets.pointcloud_augmentation import PointCloudGeometryAugmentation
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


def _geometry_batch() -> dict[str, torch.Tensor]:
    point_count = 128
    valid_count = 120
    generator = torch.Generator().manual_seed(19)
    xyz = torch.rand((1, point_count, 3), generator=generator)
    point_mask = torch.zeros((1, point_count), dtype=torch.bool)
    point_mask[:, :valid_count] = True
    source_view = torch.full((1, point_count), -1, dtype=torch.long)
    source_view[:, :40] = 0
    source_view[:, 40:80] = 1
    source_view[:, 80:valid_count] = 2
    target_mask = torch.zeros((1, point_count), dtype=torch.bool)
    target_mask[:, :12] = True
    return {
        "xyz": xyz,
        "rgb": torch.rand((1, point_count, 3), generator=generator),
        "point_mask": point_mask,
        "source_view": source_view,
        "instance_id": torch.where(
            point_mask, torch.arange(point_count)[None] // 20, torch.full((1, point_count), -1)
        ),
        "target_mask": target_mask,
        "region_target": target_mask.clone(),
        "region_valid": target_mask.clone(),
        "grid_coord": torch.zeros((1, point_count, 3), dtype=torch.int32),
    }


def _disable_all_augmentation(config: AugmentationConfig) -> None:
    for name in (
        "zero_rgb", "grayscale", "color_jitter", "object_recolor",
        "material_jitter", "lighting_jitter", "sensor_noise",
        "channel_dropout", "point_dropout", "depth_noise", "hole_dropout",
        "outlier_injection", "view_dropout", "density_variation",
        "extrinsic_jitter", "occlusion",
    ):
        getattr(config, name).probability = 0.0


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


def test_zero_probability_geometry_augmentation_is_identity() -> None:
    config = AugmentationConfig()
    _disable_all_augmentation(config)
    batch = _geometry_batch()
    original = {name: value.clone() for name, value in batch.items()}
    PointCloudGeometryAugmentation(config)(batch)
    assert set(batch) == set(original)
    for name, value in original.items():
        assert torch.equal(batch[name], value)


def test_depth_and_extrinsic_augmentation_rebuilds_grid_coordinates() -> None:
    config = AugmentationConfig()
    _disable_all_augmentation(config)
    config.depth_noise.probability = 1.0
    config.depth_noise.std_m = (0.002, 0.002)
    config.extrinsic_jitter.probability = 1.0
    config.extrinsic_jitter.translation_std_m = (0.001, 0.001)
    config.extrinsic_jitter.rotation_degrees = (0.5, 0.5)
    batch = _geometry_batch()
    xyz_before = batch["xyz"].clone()
    mask_before = batch["point_mask"].clone()
    torch.manual_seed(23)
    PointCloudGeometryAugmentation(config)(batch)
    assert "grid_coord" not in batch
    assert not torch.equal(batch["xyz"][mask_before], xyz_before[mask_before])
    assert torch.equal(batch["point_mask"], mask_before)


def test_visibility_augmentations_remove_points_but_keep_visible_target() -> None:
    config = AugmentationConfig()
    _disable_all_augmentation(config)
    config.view_dropout.probability = 1.0
    config.view_dropout.max_views = 1
    config.density_variation.probability = 1.0
    config.density_variation.keep_ratio = (0.7, 0.7)
    config.occlusion.probability = 1.0
    config.occlusion.fraction = (0.1, 0.1)
    batch = _geometry_batch()
    mask_before = batch["point_mask"].clone()
    before_count = int(batch["point_mask"].sum())
    torch.manual_seed(29)
    PointCloudGeometryAugmentation(config)(batch)
    after = batch["point_mask"]
    assert PointCloudGeometryAugmentation.MIN_REMAINING_POINTS <= int(after.sum()) < before_count
    assert bool((after & batch["target_mask"]).any())
    removed = mask_before & ~after
    assert torch.count_nonzero(batch["xyz"][removed]) == 0
    assert torch.count_nonzero(batch["rgb"][removed]) == 0
    assert torch.all(batch["instance_id"][removed] == -1)
    assert "grid_coord" not in batch


def test_hole_dropout_removes_local_geometry_with_probability_one() -> None:
    config = AugmentationConfig()
    _disable_all_augmentation(config)
    config.hole_dropout.probability = 1.0
    config.hole_dropout.count = (1, 1)
    config.hole_dropout.radius_m = (10.0, 10.0)
    batch = _geometry_batch()
    torch.manual_seed(30)
    PointCloudGeometryAugmentation(config)(batch)
    assert (
        int(batch["point_mask"].sum())
        == PointCloudGeometryAugmentation.MIN_REMAINING_POINTS
    )
    assert bool((batch["point_mask"] & batch["target_mask"]).any())


def test_outlier_injection_moves_points_and_clears_point_labels() -> None:
    config = AugmentationConfig()
    _disable_all_augmentation(config)
    config.outlier_injection.probability = 1.0
    config.outlier_injection.fraction = (0.1, 0.1)
    config.outlier_injection.displacement_m = (0.02, 0.02)
    batch = _geometry_batch()
    xyz_before = batch["xyz"].clone()
    torch.manual_seed(31)
    PointCloudGeometryAugmentation(config)(batch)
    moved = torch.linalg.vector_norm(batch["xyz"] - xyz_before, dim=-1) > 0
    assert int(moved.sum()) == 12
    assert torch.all(batch["instance_id"][moved] == -1)
    assert not batch["target_mask"][moved].any()
    assert "grid_coord" not in batch


def test_augmentation_debug_is_globally_bounded_and_uses_output_dir(tmp_path) -> None:
    first = claim_debug_batch(tmp_path, 2)
    second = claim_debug_batch(tmp_path, 2)
    assert first == tmp_path / "augmentation_debug" / "batch_0000"
    assert second == tmp_path / "augmentation_debug" / "batch_0001"
    assert claim_debug_batch(tmp_path, 2) is None
    batch = _batch()
    before = batch["rgb"].clone()
    xyz_before = batch["xyz"].clone()
    point_mask_before = batch["point_mask"].clone()
    batch["rgb"][:, :6] = 0
    batch["xyz"][:, :6, 2] += 0.001
    batch["point_mask"][:, 5] = False
    save_debug_batch(first, batch, before, xyz_before, point_mask_before)
    assert (first / "batch_before_after.npz").is_file()
    assert (first / "report.json").is_file()
    assert (first / "preview.svg").is_file()
    report = json.loads((first / "report.json").read_text(encoding="utf-8"))
    assert report["xyz_changed"] is True
    assert report["removed_points"] == 1


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
