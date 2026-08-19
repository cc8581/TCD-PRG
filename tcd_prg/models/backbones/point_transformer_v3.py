"""Adapter for the official Pointcept Point Transformer V3 implementation.

Only fused XYZRGB and point validity enter the backbone. Instance membership is
predicted later by InstanceQueryHead.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import spconv.pytorch as spconv
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from tcd_prg.paths import project_path
from ..common import MaskedAttentionPool
from .task_point_transformer import SceneGeometryOutput


def _load_official_model(source_root: str | Path) -> type[nn.Module]:
    root = Path(source_root)
    if not root.is_absolute():
        root = project_path(root)
    model_file = root / "model.py"
    if not model_file.is_file():
        raise RuntimeError(
            f"Official PTv3 source is missing at {model_file}. Run "
            "`git submodule update --init third_party/PointTransformerV3`."
        )
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        module = importlib.import_module(f"{root.name}.model")
    except ImportError as error:
        raise RuntimeError(
            "Official PTv3 dependencies are unavailable. Install addict, einops, timm, "
            "torch-scatter and the spconv wheel matching the PyTorch CUDA build."
        ) from error
    return module.PointTransformerV3


class PointTransformerV3SceneGeometryBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        source_root: str,
        grid_size_m: float,
        enable_flash_attention: bool,
        patch_size: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        official = _load_official_model(source_root)
        patches = (patch_size,) * 5
        self.grid_size_m = float(grid_size_m)
        self.activation_checkpointing = activation_checkpointing
        original_submanifold_conv = spconv.SubMConv3d

        def stable_submanifold_conv(*args, **kwargs):
            kwargs.setdefault("algo", spconv.ConvAlgo.Native)
            return original_submanifold_conv(*args, **kwargs)

        spconv.SubMConv3d = stable_submanifold_conv
        try:
            self.backbone = official(
                in_channels=6,
                enc_patch_size=patches,
                dec_patch_size=patches[:-1],
                enable_flash=enable_flash_attention,
                enable_rpe=False,
                upcast_attention=not enable_flash_attention,
                upcast_softmax=not enable_flash_attention,
            )
        finally:
            spconv.SubMConv3d = original_submanifold_conv
        self.output_projection = nn.Sequential(nn.Linear(64, dim), nn.LayerNorm(dim))
        self.pool_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.global_pool = MaskedAttentionPool(dim, dim)

    def _voxelize(
        self, xyz: Tensor, rgb: Tensor, point_mask: Tensor,
        grid_coord: Tensor | None = None,
    ) -> tuple[dict[str, Any], Tensor, Tensor]:
        batch_size, point_count = point_mask.shape
        flat_valid = point_mask.flatten()
        if not flat_valid.any():
            raise ValueError("A scene batch contains no valid points")
        flat_xyz = xyz.reshape(-1, 3)[flat_valid]
        flat_rgb = rgb.reshape(-1, 3)[flat_valid]
        flat_batch = (
            torch.arange(batch_size, device=xyz.device)[:, None]
            .expand(-1, point_count).reshape(-1)[flat_valid]
        )
        if grid_coord is not None:
            if grid_coord.shape != xyz.shape:
                raise ValueError("grid_coord must have the same [B,N,3] shape as xyz")
            flat_grid = grid_coord.reshape(-1, 3)[flat_valid].to(torch.int32)
            inverse = torch.arange(flat_xyz.shape[0], device=xyz.device)
            return {
                "coord": flat_xyz,
                "grid_coord": flat_grid,
                "feat": torch.cat((flat_xyz, flat_rgb), -1),
                "batch": flat_batch.long(),
            }, inverse, flat_valid

        grid_rows = []
        for row in range(batch_size):
            selected = flat_batch == row
            if not selected.any():
                raise ValueError(f"Scene batch row {row} contains no valid points")
            lower = flat_xyz[selected].amin(0)
            grid_rows.append(
                torch.div(
                    flat_xyz[selected] - lower, self.grid_size_m,
                    rounding_mode="floor"
                ).long()
            )
        grid = torch.cat(grid_rows, 0)
        voxel_key = torch.cat((flat_batch[:, None], grid), -1)
        unique_key, inverse = torch.unique(
            voxel_key, dim=0, sorted=True, return_inverse=True
        )
        voxel_count = unique_key.shape[0]
        counts = torch.bincount(inverse, minlength=voxel_count)
        order = torch.argsort(inverse, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        if self.training:
            offsets = torch.randint(
                0, int(counts.max().item()), (voxel_count,), device=xyz.device
            ) % counts
        else:
            offsets = torch.zeros(voxel_count, dtype=torch.long, device=xyz.device)
        representative = order[starts + offsets]
        return {
            "coord": flat_xyz[representative],
            "grid_coord": unique_key[:, 1:].to(torch.int32),
            "feat": torch.cat(
                (flat_xyz[representative], flat_rgb[representative]), -1
            ),
            "batch": unique_key[:, 0].long(),
        }, inverse, flat_valid

    def forward(
        self,
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        grid_coord: Tensor | None = None,
    ) -> SceneGeometryOutput:
        data, inverse, flat_valid = self._voxelize(
            xyz, rgb, point_mask, grid_coord=grid_coord
        )
        if self.activation_checkpointing and self.training:
            def encode(
                coord: Tensor, grid: Tensor, feat: Tensor, batch: Tensor
            ) -> Tensor:
                return self.backbone({
                    "coord": coord, "grid_coord": grid,
                    "feat": feat, "batch": batch,
                }).feat

            encoded_features = checkpoint(
                encode, data["coord"], data["grid_coord"], data["feat"], data["batch"],
                use_reentrant=False,
            )
        else:
            encoded_features = self.backbone(data).feat

        voxel_features = self.output_projection(encoded_features)
        point_features = xyz.new_zeros(
            (*point_mask.shape, voxel_features.shape[-1])
        )
        point_features.view(-1, voxel_features.shape[-1])[flat_valid] = (
            voxel_features[inverse]
        )
        query = self.pool_query[None].expand(xyz.shape[0], -1)
        global_token = self.global_pool(point_features, point_mask, query)
        return SceneGeometryOutput(point_features, global_token)
