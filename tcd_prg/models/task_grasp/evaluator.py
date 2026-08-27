"""Independent task-conditioned binary evaluation of concrete AG-160-95 poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


def rotation_6d(matrix: Tensor) -> Tensor:
    """Continuous pose representation formed by the first two rotation columns."""
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def world_to_grasp(
    points_world: Tensor, translation_world: Tensor, rotation_world: Tensor
) -> Tensor:
    """Transform row-vector world points to the canonical AG/TCD grasp frame."""
    return torch.einsum(
        "bkni,bkij->bknj", points_world[:, None] - translation_world[:, :, None], rotation_world
    )


def masked_farthest_point_sample(xyz: Tensor, mask: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Deterministic geometry-based FPS with padding kept explicitly invalid."""
    batch, _, _ = xyz.shape
    indices = torch.zeros((batch, count), dtype=torch.long, device=xyz.device)
    valid_count = mask.sum(-1)
    selected_mask = torch.arange(count, device=xyz.device)[None] < valid_count[:, None]
    mass = valid_count.clamp_min(1).to(xyz.dtype)[:, None]
    centroid = (xyz * mask[..., None]).sum(1) / mass
    initial = ((xyz - centroid[:, None]) ** 2).sum(-1).masked_fill(~mask, -1.0)
    current = initial.argmax(-1)
    minimum = torch.full(mask.shape, torch.inf, dtype=xyz.dtype, device=xyz.device)
    rows = torch.arange(batch, device=xyz.device)
    for step in range(count):
        indices[:, step] = current
        center = xyz[rows, current]
        distance = ((xyz - center[:, None]) ** 2).sum(-1)
        minimum = torch.minimum(minimum, distance).masked_fill(~mask, -1.0)
        minimum[rows, current] = -1.0
        current = minimum.argmax(-1)
    return indices, selected_mask


def deterministic_voxel_representatives(
    xyz: Tensor,
    feature: Tensor,
    mask: Tensor,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    voxel_size_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Choose the observed point nearest each fixed voxel center, independent of order."""
    lower_tensor = xyz.new_tensor(lower)
    upper_tensor = xyz.new_tensor(upper)
    grid_shape = torch.ceil((upper_tensor - lower_tensor) / voxel_size_m).long()
    voxel_count = int(grid_shape.prod())
    coordinate = torch.floor((xyz - lower_tensor) / voxel_size_m).long()
    coordinate = torch.minimum(
        torch.maximum(coordinate, torch.zeros_like(coordinate)), grid_shape - 1
    )
    voxel = coordinate[..., 0] + grid_shape[0] * (
        coordinate[..., 1] + grid_shape[1] * coordinate[..., 2]
    )
    center = lower_tensor + (coordinate.to(xyz.dtype) + 0.5) * voxel_size_m
    center_distance = ((xyz - center) ** 2).sum(-1).masked_fill(~mask, torch.inf)
    minimum = torch.full(
        (xyz.shape[0], voxel_count),
        torch.inf,
        dtype=center_distance.dtype,
        device=xyz.device,
    )
    minimum.scatter_reduce_(1, voxel, center_distance, reduce="amin", include_self=True)
    winning = mask & (center_distance <= minimum.gather(1, voxel))
    total = xyz.new_zeros((xyz.shape[0], voxel_count, 3))
    total.scatter_add_(1, voxel[..., None].expand(-1, -1, 3), xyz * winning[..., None])
    count = xyz.new_zeros((xyz.shape[0], voxel_count))
    count.scatter_add_(1, voxel, winning.to(xyz.dtype))
    representative = total / count.clamp_min(1)[..., None]
    feature_total = feature.new_zeros((feature.shape[0], voxel_count, feature.shape[-1]))
    feature_total.scatter_add_(
        1, voxel[..., None].expand(-1, -1, feature.shape[-1]), feature * winning[..., None]
    )
    representative_feature = feature_total / count.clamp_min(1)[..., None]
    return representative, representative_feature, count > 0


class PointNetSetAbstraction(nn.Module):
    """Small deterministic PointNet++ set-abstraction layer implemented in PyTorch."""

    def __init__(self, input_dim: int, output_dim: int, centers: int, radius_m: float) -> None:
        super().__init__()
        self.centers = int(centers)
        self.radius_m = float(radius_m)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, xyz: Tensor, feature: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        index, center_mask = masked_farthest_point_sample(xyz, mask, self.centers)
        center = xyz.gather(1, index[..., None].expand(-1, -1, 3))
        distance = torch.cdist(center.float(), xyz.float()).to(xyz.dtype)
        neighbor = (distance <= self.radius_m) & mask[:, None] & center_mask[:, :, None]
        relative = xyz[:, None] - center[:, :, None]
        values = self.mlp(
            torch.cat((relative, feature[:, None].expand(-1, self.centers, -1, -1)), -1)
        )
        values = values.masked_fill(~neighbor[..., None], torch.finfo(values.dtype).min)
        pooled = values.max(2).values
        pooled = torch.where(center_mask[..., None], pooled, torch.zeros_like(pooled))
        return center, pooled, center_mask


class TaskGraspEvaluator(nn.Module):
    """Three-stream binary evaluator; candidates never attend to one another."""

    def __init__(
        self,
        dim: int,
        gripper_geometry_path: str | Path,
        num_categories: int = 64,
        num_regions: int = 64,
        scene_points: int = 256,
        gripper_points: int = 128,
        voxel_size_m: float = 0.017,
    ) -> None:
        super().__init__()
        payload = np.load(Path(gripper_geometry_path))
        points = torch.from_numpy(payload["points_tcp"].astype(np.float32))
        part = torch.from_numpy(payload["part_id"].astype(np.int64))
        if points.shape != (gripper_points, 3) or part.shape != (gripper_points,):
            raise ValueError(
                f"AG geometry must contain exactly {gripper_points} points and part IDs"
            )
        self.register_buffer("gripper_points_tcp", points, persistent=True)
        self.register_buffer("gripper_part_id", part, persistent=True)
        self.scene_points = int(scene_points)
        self.voxel_size_m = float(voxel_size_m)
        self.sa1 = PointNetSetAbstraction(9, 96, centers=64, radius_m=0.045)
        self.sa2 = PointNetSetAbstraction(96, 128, centers=16, radius_m=0.090)
        self.pose = nn.Sequential(nn.Linear(10, dim), nn.LayerNorm(dim), nn.GELU())
        self.category_embedding = nn.Embedding(num_categories, dim)
        self.region_embedding = nn.Embedding(num_regions, dim)
        self.task_projection = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.semantic = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(128 + 2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _local_cloud(
        self,
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        target_probability: Tensor,
        region_probability: Tensor,
        translation: Tensor,
        rotation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local = world_to_grasp(xyz, translation, rotation)
        # Contains the complete label-side approach corridor plus 5 mm margin.
        # x is closing, y is height, z is approach.
        inside = (
            (local[..., 0].abs() <= 0.090)
            & (local[..., 1].abs() <= 0.045)
            & (local[..., 2] >= -0.230)
            & (local[..., 2] <= 0.025)
            & point_mask[:, None]
        )
        b, k, n, _ = local.shape
        flat_local = torch.nan_to_num(local.reshape(b * k, n, 3))
        flat_inside = inside.reshape(b * k, n)
        scene_identity = torch.zeros((*xyz.shape[:2], 4), dtype=xyz.dtype, device=xyz.device)
        scene_identity[..., 0] = 1
        scene_feature = torch.cat(
            (rgb, target_probability[..., None], region_probability[..., None], scene_identity), -1
        )
        flat_scene_feature = scene_feature[:, None].expand(-1, k, -1, -1).reshape(b * k, n, 9)
        preselected_local, preselected_feature, preselected_mask = deterministic_voxel_representatives(
            flat_local,
            flat_scene_feature,
            flat_inside,
            (-0.090, -0.045, -0.230),
            (0.090, 0.045, 0.025),
            self.voxel_size_m,
        )
        local_index, selected_mask = masked_farthest_point_sample(
            preselected_local, preselected_mask, self.scene_points
        )
        selected = preselected_local.gather(1, local_index[..., None].expand(-1, -1, 3))
        selected_feature = preselected_feature.gather(
            1, local_index[..., None].expand(-1, -1, preselected_feature.shape[-1])
        )
        selected = selected.reshape(b, k, self.scene_points, 3)
        selected_feature = selected_feature.reshape(b, k, self.scene_points, 9)
        selected_mask = selected_mask.reshape(b, k, self.scene_points)
        gripper = self.gripper_points_tcp.to(local)[None, None].expand(b, k, -1, -1)
        cloud = torch.cat((selected, gripper), 2)
        gripper_feature = torch.zeros((b, k, gripper.shape[2], 9), dtype=local.dtype, device=local.device)
        gripper_feature[..., 5:] = torch.nn.functional.one_hot(
            self.gripper_part_id.to(local.device), num_classes=4
        ).to(local.dtype)[None, None]
        feature = torch.cat((selected_feature, gripper_feature), 2)
        mask = torch.cat(
            (
                selected_mask,
                torch.ones((b, k, gripper.shape[2]), dtype=torch.bool, device=local.device),
            ),
            2,
        )
        return cloud, feature, mask

    def forward(
        self,
        proposals: dict[str, Tensor],
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        target_probability: Tensor,
        region_probability: Tensor,
        task_category_id: Tensor,
        task_region_id: Tensor,
    ) -> dict[str, Tensor]:
        translation = proposals["translation_world"]
        rotation = proposals["rotation_matrix"]
        valid = proposals.get(
            "valid", torch.ones(translation.shape[:2], dtype=torch.bool, device=translation.device)
        ).bool()
        cloud, local_feature, local_mask = self._local_cloud(
            xyz, rgb, point_mask.bool(), target_probability, region_probability,
            translation, rotation
        )
        b, k, p, _ = cloud.shape
        flat_xyz = cloud.reshape(b * k, p, 3)
        flat_feature = local_feature.reshape(b * k, p, 9)
        flat_mask = local_mask.reshape(b * k, p)
        center, feature, mask = self.sa1(flat_xyz, flat_feature, flat_mask)
        _, feature, mask = self.sa2(center, feature, mask)
        geometry = (
            feature.masked_fill(~mask[..., None], torch.finfo(feature.dtype).min).max(1).values
        )
        geometry = torch.where(valid.reshape(-1, 1), geometry, torch.zeros_like(geometry)).reshape(
            b, k, -1
        )

        scene_semantic = self.semantic(local_feature[:, :, : self.scene_points]).masked_fill(
            ~local_mask[:, :, : self.scene_points, None], 0.0
        )
        scene_semantic = scene_semantic.sum(2) / local_mask[:, :, : self.scene_points].sum(
            2, keepdim=True
        ).clamp_min(1)
        task_semantic = self.task_projection(torch.cat((
            self.category_embedding(task_category_id), self.region_embedding(task_region_id)
        ), -1))
        target_weight = target_probability * point_mask.to(target_probability.dtype)
        target_center = torch.einsum("bn,bnd->bd", target_weight, xyz) / target_weight.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        width = proposals["width_m"].unsqueeze(-1)
        pose_input = torch.nan_to_num(
            torch.cat((translation - target_center[:, None], rotation_6d(rotation), width), -1)
        )
        pose = self.pose(pose_input)
        context = torch.cat((geometry, scene_semantic + task_semantic[:, None], pose), -1)
        logit = self.head(torch.nan_to_num(context)).squeeze(-1).masked_fill(~valid, -30.0)
        return {
            **proposals,
            "task_valid_logit": logit,
            "task_valid_probability": torch.sigmoid(logit),
            "valid": valid,
        }
