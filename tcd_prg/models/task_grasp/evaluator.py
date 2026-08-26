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
        count = xyz.shape[1]
        index = torch.linspace(0, max(0, count - 1), self.centers, device=xyz.device).long()
        center = xyz[:, index]
        center_mask = mask[:, index]
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
        scene_points: int = 256,
        gripper_points: int = 128,
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
        self.sa1 = PointNetSetAbstraction(4, 96, centers=64, radius_m=0.045)
        self.sa2 = PointNetSetAbstraction(96, 128, centers=16, radius_m=0.090)
        self.pose = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.GELU())
        self.semantic = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(128 + 3 * dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    @staticmethod
    def _weighted_pool(feature: Tensor, weight: Tensor) -> Tensor:
        mass = weight.sum(-1, keepdim=True)
        return torch.einsum("bn,bnd->bd", weight, feature) / mass.clamp_min(1e-6)

    def _local_cloud(
        self, xyz: Tensor, point_mask: Tensor, translation: Tensor, rotation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        local = world_to_grasp(xyz, translation, rotation)
        # Covers the real open AG CAD from TCP z=-0.19 to the fingertip plus a
        # small forward margin. x is closing, y is height, z is approach.
        inside = (
            (local[..., 0].abs() <= 0.090)
            & (local[..., 1].abs() <= 0.045)
            & (local[..., 2] >= -0.205)
            & (local[..., 2] <= 0.025)
            & point_mask[:, None]
        )
        b, k, n, _ = local.shape
        selected = local.new_zeros((b, k, self.scene_points, 3))
        selected_mask = torch.zeros(
            (b, k, self.scene_points), dtype=torch.bool, device=local.device
        )
        for row in range(b):
            for candidate in range(k):
                index = torch.nonzero(inside[row, candidate], as_tuple=False).flatten()
                if not len(index):
                    continue
                take = torch.linspace(
                    0, len(index) - 1, self.scene_points, device=local.device
                ).long()
                selected[row, candidate] = local[row, candidate, index[take]]
                selected_mask[row, candidate] = True
        gripper = self.gripper_points_tcp.to(local)[None, None].expand(b, k, -1, -1)
        cloud = torch.cat((selected, gripper), 2)
        part = torch.zeros((b, k, cloud.shape[2]), dtype=torch.long, device=local.device)
        part[:, :, self.scene_points :] = self.gripper_part_id.to(local.device)[None, None]
        feature = torch.nn.functional.one_hot(part, num_classes=4).to(local.dtype)
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
        point_features: Tensor,
        xyz: Tensor,
        point_mask: Tensor,
        region_probability: Tensor,
        target_probability: Tensor,
        task_token: Tensor,
        target_token: Tensor,
    ) -> dict[str, Tensor]:
        translation = proposals["translation_world"]
        rotation = proposals["rotation_matrix"]
        valid = proposals.get(
            "valid", torch.ones(translation.shape[:2], dtype=torch.bool, device=translation.device)
        ).bool()
        cloud, local_feature, local_mask = self._local_cloud(
            xyz, point_mask.bool(), translation, rotation
        )
        b, k, p, _ = cloud.shape
        flat_xyz = cloud.reshape(b * k, p, 3)
        flat_feature = local_feature.reshape(b * k, p, 4)
        flat_mask = local_mask.reshape(b * k, p)
        center, feature, mask = self.sa1(flat_xyz, flat_feature, flat_mask)
        _, feature, mask = self.sa2(center, feature, mask)
        geometry = (
            feature.masked_fill(~mask[..., None], torch.finfo(feature.dtype).min).max(1).values
        )
        geometry = torch.where(valid.reshape(-1, 1), geometry, torch.zeros_like(geometry)).reshape(
            b, k, -1
        )

        target_weight = target_probability * point_mask.to(target_probability.dtype)
        region_weight = target_weight * region_probability
        target_pool = self._weighted_pool(point_features, target_weight)
        region_pool = self._weighted_pool(point_features, region_weight)
        semantic = self.semantic(
            torch.cat((target_pool, region_pool, task_token, target_token), -1)
        )
        target_center = torch.einsum("bn,bnd->bd", target_weight, xyz) / target_weight.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)
        pose_input = torch.nan_to_num(
            torch.cat((translation - target_center[:, None], rotation_6d(rotation)), -1)
        )
        pose = self.pose(pose_input)
        context = torch.cat((geometry, semantic[:, None].expand(-1, k, -1), pose), -1)
        logit = self.head(torch.nan_to_num(context)).squeeze(-1).masked_fill(~valid, -30.0)
        return {
            **proposals,
            "task_valid_logit": logit,
            "task_valid_probability": torch.sigmoid(logit),
            "valid": valid,
        }
