"""Adapter for the official Pointcept Point Transformer V3 implementation.

Only fused XYZRGB and point validity enter the backbone. Instance membership is
predicted later by InstanceQueryHead.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import spconv.pytorch as spconv
import torch
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


class _SonataEmbeddingStem(nn.Module):
    """Sonata's linear point stem (the older vendored PTv3 used sparse conv)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(9, 48)
        self.norm = nn.LayerNorm(48)
        self.act = nn.GELU()

    def forward(self, point):
        point.feat = self.act(self.norm(self.linear(point.feat)))
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class PointTransformerV3SceneGeometryBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        source_root: str,
        grid_size_m: float,
        enable_flash_attention: bool,
        patch_size: int,
        activation_checkpointing: bool,
        profile: str = "default",
    ) -> None:
        super().__init__()
        official = _load_official_model(source_root)
        if profile not in {"default", "sonata"}:
            raise ValueError("PTv3 profile must be default or sonata")
        self.profile = profile
        patches = (patch_size,) * 5
        self.grid_size_m = float(grid_size_m)
        self.activation_checkpointing = activation_checkpointing
        original_submanifold_conv = spconv.SubMConv3d

        def stable_submanifold_conv(*args, **kwargs):
            kwargs.setdefault("algo", spconv.ConvAlgo.Native)
            return original_submanifold_conv(*args, **kwargs)

        spconv.SubMConv3d = stable_submanifold_conv
        try:
            arguments = dict(
                in_channels=6,
                enc_patch_size=patches,
                dec_patch_size=patches[:-1],
                enable_flash=enable_flash_attention,
                enable_rpe=False,
                upcast_attention=not enable_flash_attention,
                upcast_softmax=not enable_flash_attention,
            )
            if profile == "sonata":
                arguments.update(
                    in_channels=9,
                    order=("z", "z-trans", "hilbert", "hilbert-trans"),
                    stride=(2, 2, 2, 2),
                    enc_depths=(3, 3, 3, 12, 3),
                    enc_channels=(48, 96, 192, 384, 512),
                    enc_num_head=(3, 6, 12, 24, 32),
                    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                    drop_path=0.3,
                    cls_mode=True,
                )
                arguments.pop("dec_patch_size")
            self.backbone = official(**arguments)
            if profile == "sonata":
                # Sonata's released checkpoint comes from the newer encoder:
                # a Linear+LayerNorm stem and LayerNorm after serialized pooling.
                # The vendored standalone PTv3 predates those two changes.
                self.backbone.embedding.stem = _SonataEmbeddingStem()
                for stage, channels in enumerate((96, 192, 384, 512), start=1):
                    down = getattr(self.backbone.enc, f"enc{stage}").down
                    down.norm._modules["0"] = nn.LayerNorm(channels)
        finally:
            spconv.SubMConv3d = original_submanifold_conv
        output_channels = 512 if profile == "sonata" else 64
        self.output_projection = nn.Sequential(
            nn.Linear(output_channels, dim), nn.LayerNorm(dim)
        )
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
        normals = self._estimate_normals(xyz, point_mask) if self.profile == "sonata" else None
        flat_xyz = xyz.reshape(-1, 3)[flat_valid]
        flat_rgb = rgb.reshape(-1, 3)[flat_valid]
        flat_normal = normals.reshape(-1, 3)[flat_valid] if normals is not None else None
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
                "feat": self._features(flat_xyz, flat_rgb, flat_normal, flat_batch),
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
            "feat": self._features(
                flat_xyz[representative], flat_rgb[representative],
                flat_normal[representative] if flat_normal is not None else None,
                unique_key[:, 0].long(),
            ),
            "batch": unique_key[:, 0].long(),
        }, inverse, flat_valid

    def _features(
        self, xyz: Tensor, rgb: Tensor, normal: Tensor | None, batch: Tensor
    ) -> Tensor:
        if self.profile != "sonata":
            return torch.cat((xyz, rgb), -1)
        # Sonata is trained with CenterShift(apply_z=False), NormalizeColor and
        # feat_keys=(coord,color,normal).
        centered = xyz.clone()
        for row in torch.unique(batch):
            selected = batch == row
            centered[selected, :2] -= xyz[selected, :2].mean(0)
        color = rgb * 2.0 - 1.0 if rgb.numel() and rgb.amin() >= 0 else rgb
        return torch.cat((centered, color, normal), -1)

    @staticmethod
    def _estimate_normals(xyz: Tensor, point_mask: Tensor, neighbors: int = 16) -> Tensor:
        """Deterministic local-PCA normals for fused point clouds."""
        result = torch.zeros_like(xyz)
        with torch.no_grad():
            for row in range(xyz.shape[0]):
                valid = point_mask[row]
                points = xyz[row, valid]
                if len(points) < 3:
                    continue
                k = min(neighbors + 1, len(points))
                indices = torch.cdist(points, points).topk(k, largest=False).indices[:, 1:]
                offsets = points[indices] - points[:, None]
                covariance = offsets.transpose(1, 2) @ offsets / max(1, k - 1)
                normal = torch.linalg.eigh(covariance).eigenvectors[..., 0]
                # PCA signs are arbitrary. Canonicalize by the dominant axis so
                # repeated runs and point permutations produce identical inputs.
                dominant = normal.abs().argmax(-1, keepdim=True)
                sign = normal.gather(-1, dominant).sign().clamp(min=-1, max=1)
                sign[sign == 0] = 1
                result[row, valid] = normal * sign
        return result

    @staticmethod
    def _restore_encoder_resolution(point):
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point

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
        if self.activation_checkpointing and self.training and self.profile != "sonata":
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
            encoded = self.backbone(data)
            if self.profile == "sonata":
                encoded = self._restore_encoder_resolution(encoded)
            encoded_features = encoded.feat

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
