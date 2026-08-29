"""Composable online RGB, geometry, and depth augmentation for point clouds."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tcd_prg.config import AugmentationConfig

from .rgb_augmentation import PointCloudRGBAugmentation


def _chance(probability: float, *, device: torch.device) -> bool:
    return probability > 0 and bool(torch.rand((), device=device) < probability)


def _uniform(low: float, high: float, *, device: torch.device) -> Tensor:
    if low == high:
        return torch.tensor(low, device=device)
    return torch.empty((), device=device).uniform_(low, high)


def _rotation_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind()
    zero = axis.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


class PointCloudGeometryAugmentation:
    """Apply independent training-only sensor-domain transforms to ``xyz``/visibility."""

    MIN_REMAINING_POINTS = 32

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        xyz = batch.get("xyz")
        point_mask = batch.get("point_mask")
        if not isinstance(xyz, Tensor) or not isinstance(point_mask, Tensor):
            return batch
        changed = False
        for row in range(len(xyz)):
            if not point_mask[row].any():
                continue
            changed |= self._sample(batch, row)
        if changed:
            # Collation's precomputed voxel coordinates no longer describe the
            # augmented geometry. Let PTv3 rebuild them from xyz and point_mask.
            batch.pop("grid_coord", None)
        return batch

    def _sample(self, batch: dict[str, Any], row: int) -> bool:
        xyz = batch["xyz"]
        point_mask = batch["point_mask"]
        assert isinstance(xyz, Tensor) and isinstance(point_mask, Tensor)
        device = xyz.device
        changed = False

        depth = self.config.depth_noise
        if _chance(depth.probability, device=device):
            valid = point_mask[row].bool()
            std = _uniform(*depth.std_m, device=device)
            # The saved clouds are in world coordinates and do not retain all
            # camera rays. Axial Z noise is therefore the auditable proxy for
            # depth/range uncertainty; lateral calibration error is handled below.
            xyz[row, valid, 2] += torch.randn(int(valid.sum()), device=device) * std
            changed = True

        extrinsic = self.config.extrinsic_jitter
        if _chance(extrinsic.probability, device=device):
            valid = point_mask[row].bool()
            source_view = batch.get("source_view")
            if isinstance(source_view, Tensor):
                view_ids = torch.unique(source_view[row, valid])
                groups = [
                    valid & (source_view[row] == view_id)
                    for view_id in view_ids
                    if int(view_id) >= 0
                ]
            else:
                groups = [valid]
            if not groups:
                groups = [valid]
            for selected in groups:
                if not selected.any():
                    continue
                axis = torch.randn(3, device=device, dtype=xyz.dtype)
                degrees = _uniform(*extrinsic.rotation_degrees, device=device)
                angle = degrees * (math.pi / 180.0)
                rotation = _rotation_matrix(axis, angle)
                std = _uniform(*extrinsic.translation_std_m, device=device)
                translation = torch.randn(3, device=device, dtype=xyz.dtype) * std
                points = xyz[row, selected]
                center = points.mean(0, keepdim=True)
                xyz[row, selected] = (points - center) @ rotation.T + center + translation
            changed = True

        holes = self.config.hole_dropout
        if _chance(holes.probability, device=device):
            valid_indices = torch.nonzero(point_mask[row], as_tuple=False).flatten()
            if len(valid_indices):
                count = int(torch.randint(holes.count[0], holes.count[1] + 1, (), device=device))
                center_indices = torch.randint(
                    0, len(valid_indices), (count,), device=device
                )
                centers = xyz[row, valid_indices[center_indices]]
                radii = torch.empty(count, device=device).uniform_(*holes.radius_m)
                distances = torch.cdist(xyz[row], centers)
                drop = (distances <= radii[None]).any(1) & point_mask[row].bool()
                changed |= self._drop_points(batch, row, drop)

        view_dropout = self.config.view_dropout
        source_view = batch.get("source_view")
        if (
            isinstance(source_view, Tensor)
            and _chance(view_dropout.probability, device=device)
        ):
            valid = point_mask[row].bool()
            views = torch.unique(source_view[row, valid])
            views = views[views >= 0]
            if len(views) > 1:
                maximum = min(int(view_dropout.max_views), len(views) - 1)
                count = int(torch.randint(1, maximum + 1, (), device=device))
                chosen = views[torch.randperm(len(views), device=device)[:count]]
                drop = valid & (source_view[row, :, None] == chosen[None]).any(1)
                changed |= self._drop_points(batch, row, drop)

        density = self.config.density_variation
        if _chance(density.probability, device=device):
            valid = point_mask[row].bool()
            keep_ratio = _uniform(*density.keep_ratio, device=device)
            drop = valid & (torch.rand(valid.shape, device=device) >= keep_ratio)
            changed |= self._drop_points(batch, row, drop)

        occlusion = self.config.occlusion
        if _chance(occlusion.probability, device=device):
            valid_indices = torch.nonzero(point_mask[row], as_tuple=False).flatten()
            if len(valid_indices) > self.MIN_REMAINING_POINTS:
                direction = torch.randn(3, device=device, dtype=xyz.dtype)
                direction /= direction.norm().clamp_min(1e-8)
                projection = xyz[row, valid_indices] @ direction
                order = torch.argsort(projection)
                fraction = float(_uniform(*occlusion.fraction, device=device))
                count = max(
                    1,
                    min(
                        len(order) - self.MIN_REMAINING_POINTS,
                        int(len(order) * fraction),
                    ),
                )
                start = int(torch.randint(0, len(order) - count + 1, (), device=device))
                drop = torch.zeros_like(point_mask[row], dtype=torch.bool)
                drop[valid_indices[order[start : start + count]]] = True
                changed |= self._drop_points(batch, row, drop)

        outliers = self.config.outlier_injection
        if _chance(outliers.probability, device=device):
            valid_indices = torch.nonzero(point_mask[row], as_tuple=False).flatten()
            if len(valid_indices):
                fraction = float(_uniform(*outliers.fraction, device=device))
                count = max(1, min(len(valid_indices), int(round(len(valid_indices) * fraction))))
                selected = valid_indices[torch.randperm(len(valid_indices), device=device)[:count]]
                direction = torch.randn((count, 3), device=device, dtype=xyz.dtype)
                direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                distance = torch.empty((count, 1), device=device).uniform_(*outliers.displacement_m)
                xyz[row, selected] += direction * distance
                self._clear_labels(batch, row, selected)
                changed = True
        return changed

    def _drop_points(self, batch: dict[str, Any], row: int, drop: Tensor) -> bool:
        point_mask = batch["point_mask"]
        assert isinstance(point_mask, Tensor)
        valid = point_mask[row].bool()
        drop = drop.bool() & valid
        if not drop.any():
            return False
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        maximum_drop = max(0, len(valid_indices) - self.MIN_REMAINING_POINTS)
        drop_indices = torch.nonzero(drop, as_tuple=False).flatten()
        if len(drop_indices) > maximum_drop:
            drop_indices = drop_indices[
                torch.randperm(len(drop_indices), device=drop.device)[:maximum_drop]
            ]
            drop = torch.zeros_like(drop)
            drop[drop_indices] = True
        if not drop.any():
            return False
        target_mask = batch.get("target_mask")
        if isinstance(target_mask, Tensor):
            target_valid = valid & target_mask[row].bool()
            if target_valid.any() and not (target_valid & ~drop).any():
                restore = torch.nonzero(target_valid, as_tuple=False).flatten()[0]
                drop[restore] = False
        removed = torch.nonzero(drop, as_tuple=False).flatten()
        point_mask[row, removed] = False
        self._clear_labels(batch, row, removed)
        xyz = batch.get("xyz")
        rgb = batch.get("rgb")
        if isinstance(xyz, Tensor):
            xyz[row, removed] = 0
        if isinstance(rgb, Tensor):
            rgb[row, removed] = 0
        return True

    @staticmethod
    def _clear_labels(batch: dict[str, Any], row: int, indices: Tensor) -> None:
        fills: dict[str, int | bool] = {
            "source_view": -1,
            "instance_id": -1,
            "target_mask": False,
            "region_target": False,
            "region_valid": False,
        }
        for name, fill in fills.items():
            value = batch.get(name)
            if (
                isinstance(value, Tensor)
                and value.ndim >= 2
                and value.shape[:2] == batch["point_mask"].shape
            ):
                value[row, indices] = fill


class PointCloudAugmentation:
    """Training pipeline shared by every collator that emits a scene point cloud."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.rgb = PointCloudRGBAugmentation(config)
        self.geometry = PointCloudGeometryAugmentation(config)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.rgb(batch)
        self.geometry(batch)
        return batch
