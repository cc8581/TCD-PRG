"""Learned conversion from GraspNet geometry to executable AG-160-95 width."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .scorer import _rotation_6d


class AGWidthAdapter(nn.Module):
    """Predict AG contact opening without treating GraspNet width as identity."""

    def __init__(
        self, dim: int, max_width_m: float = 0.095, local_radius_m: float = 0.08
    ) -> None:
        super().__init__()
        self.max_width_m = float(max_width_m)
        self.local_radius_m = float(local_radius_m)
        self.geometry = nn.Sequential(
            nn.Linear(12, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.local = nn.Sequential(nn.Linear(dim + 2, dim), nn.LayerNorm(dim))
        self.fused = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(
        self,
        proposals: dict[str, Tensor],
        point_features: Tensor,
        xyz_world: Tensor,
        point_mask: Tensor,
        target_probability: Tensor,
    ) -> dict[str, Tensor]:
        translation = proposals["translation_world"]
        valid = proposals["valid"].bool()
        graspnet_width = proposals.get("graspnet_width_m", proposals["width_m"])
        depth = proposals.get("depth_m", torch.zeros_like(graspnet_width))
        quality = proposals["quality_logit"]
        target_weight = target_probability * point_mask.to(target_probability.dtype)
        target_center = torch.einsum("bn,bnd->bd", target_weight, xyz_world) / (
            target_weight.sum(-1, keepdim=True).clamp_min(1e-6)
        )
        geometry_input = torch.cat(
            (
                translation - target_center[:, None],
                _rotation_6d(proposals["rotation_matrix"]),
                graspnet_width.unsqueeze(-1),
                depth.unsqueeze(-1),
                quality.unsqueeze(-1),
            ),
            -1,
        )
        geometry = self.geometry(torch.nan_to_num(geometry_input))

        distance = torch.cdist(
            torch.nan_to_num(translation, nan=0.0).float(), xyz_world.float()
        ).to(point_features.dtype)
        local = (
            (distance <= self.local_radius_m)
            & point_mask[:, None]
            & valid[:, :, None]
        )
        weights = local.to(point_features.dtype) * target_weight[:, None].to(
            point_features.dtype
        )
        mass = weights.sum(-1, keepdim=True)
        pooled = torch.einsum("bkn,bnd->bkd", weights, point_features) / mass.clamp_min(
            1e-6
        )
        local_count = local.sum(-1, keepdim=True).to(point_features.dtype)
        local_target_ratio = mass / local_count.clamp_min(1.0)
        local_feature = self.local(
            torch.cat((pooled, mass.clamp_max(1.0), local_target_ratio), -1)
        )
        width_logit = self.fused(torch.cat((geometry, local_feature), -1)).squeeze(-1)
        ag_width = self.max_width_m * torch.sigmoid(width_logit)
        ag_width = ag_width.masked_fill(~valid, 0.0)
        return {
            "ag_width_m": ag_width,
            "width_m": ag_width,
            "graspnet_width_m": graspnet_width,
            "ag_width_logit": width_logit,
        }
