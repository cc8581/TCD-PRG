"""Task-region residual scorer over frozen GraspNet proposals."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotation_6d(matrix: Tensor) -> Tensor:
    # First two rotation columns; continuous 6D representation.
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


class TaskGraspScorer(nn.Module):
    """Re-rank physically plausible GraspNet proposals for a requested task.

    GraspNet provides the physical grasp prior. This scorer learns only a residual
    task preference, initialized to zero so training starts from the pretrained
    GraspNet ranking rather than destroying it.
    """

    def __init__(
        self,
        dim: int,
        layers: int = 2,
        heads: int = 8,
        local_radius_m: float = 0.08,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("feature_dim must be divisible by task_grasp_scorer_heads")
        self.local_radius_m = float(local_radius_m)
        self.residual_scale = float(residual_scale)

        self.geometry = nn.Sequential(
            nn.Linear(12, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.local_projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.region_projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.context_projection = nn.Sequential(
            nn.Linear(5 * dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(dim)
        )
        self.residual_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _weighted_pool(
        feature: Tensor, weight: Tensor, fallback: Tensor
    ) -> Tensor:
        # feature [B,N,D], weight [B,K,N], fallback [B,K,D]
        mass = weight.sum(-1, keepdim=True)
        pooled = torch.einsum("bkn,bnd->bkd", weight, feature) / mass.clamp_min(1e-6)
        return torch.where((mass > 1e-6).expand_as(pooled), pooled, fallback)

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
        width = proposals.get("graspnet_width_m", proposals["width_m"])
        depth = proposals.get("depth_m", torch.zeros_like(width))
        valid = proposals.get("valid", torch.ones_like(width, dtype=torch.bool)).bool()
        gn_logit = proposals["quality_logit"]

        target_mass = target_probability.sum(-1, keepdim=True).clamp_min(1e-6)
        target_center = torch.einsum(
            "bn,bnd->bd", target_probability, xyz
        ) / target_mass
        relative_translation = translation - target_center[:, None]
        geometry_input = torch.cat(
            (
                relative_translation,
                _rotation_6d(rotation),
                width.unsqueeze(-1),
                depth.unsqueeze(-1),
                gn_logit.unsqueeze(-1),
            ),
            -1,
        )
        geometry = self.geometry(torch.nan_to_num(geometry_input))

        distance = torch.cdist(
            torch.nan_to_num(translation, nan=0.0).float(), xyz.float()
        ).to(point_features.dtype)
        local = (
            (distance <= self.local_radius_m)
            & point_mask[:, None].bool()
            & valid[:, :, None]
        )
        # Keep object context: task scorer should not explain a proposal using a
        # neighboring object's points in clutter.
        target_weight = target_probability[:, None].to(point_features.dtype)
        local_weight = local.to(point_features.dtype) * target_weight
        region_weight = local_weight * region_probability[:, None].to(point_features.dtype)

        local_feature = self._weighted_pool(
            point_features, local_weight, geometry
        )
        region_feature = self._weighted_pool(
            point_features, region_weight, local_feature
        )

        task = task_token[:, None].expand_as(geometry)
        target = target_token[:, None].expand_as(geometry)
        candidate = self.context_projection(
            torch.cat(
                (
                    geometry,
                    self.local_projection(local_feature),
                    self.region_projection(region_feature),
                    task,
                    target,
                ),
                -1,
            )
        )

        padding = ~valid
        all_invalid = padding.all(-1)
        safe_padding = padding
        safe_candidate = candidate
        if all_invalid.any():
            safe_padding = padding.clone()
            safe_candidate = candidate.clone()
            safe_padding[all_invalid, 0] = False
            safe_candidate[all_invalid, 0] = 0.0
        refined = self.candidate_transformer(
            safe_candidate, src_key_padding_mask=safe_padding
        )
        residual = self.residual_head(refined).squeeze(-1)
        residual = residual.masked_fill(~valid, 0.0)
        task_logit = gn_logit + self.residual_scale * residual
        task_logit = task_logit.masked_fill(~valid, -30.0)

        return {
            **proposals,
            "quality_logit": task_logit,
            "graspnet_quality_logit": gn_logit,
            "task_residual_logit": residual,
            "task_probability": torch.sigmoid(task_logit),
            "local_support": local_weight.sum(-1),
            "region_support": region_weight.sum(-1),
            "valid": valid,
        }
