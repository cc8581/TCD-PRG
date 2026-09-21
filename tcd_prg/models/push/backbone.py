"""PUSH-specific adapters over interchangeable point-cloud backbones."""

from __future__ import annotations

from torch import nn

from tcd_prg.config import BackboneConfig
from tcd_prg.pretrained import load_pretrained_backbone, prepare_pretrained_checkpoint

from ..backbones.point_transformer_v3 import PointTransformerV3SceneGeometryBackbone
from .pointnet2 import PushPointNet2

PUSH_BACKBONES = frozenset({"point_transformer_v3", "pointnet2"})


class PushPointTransformerV3(nn.Module):
    """Expose PTv3 through the dense ``xyz, rgb -> point features`` PUSH seam."""

    backend_name = "point_transformer_v3"

    def __init__(
        self,
        dim: int,
        config: BackboneConfig,
        *,
        activation_checkpointing: bool = False,
        scene_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_backbone = scene_backbone or PointTransformerV3SceneGeometryBackbone(
            dim=dim,
            source_root=config.source_root,
            grid_size_m=config.grid_size_m,
            enable_flash_attention=config.enable_flash_attention,
            patch_size=config.patch_size,
            activation_checkpointing=activation_checkpointing,
            profile="sonata",
        )

    def load_pretrained(self):
        checkpoint = prepare_pretrained_checkpoint(self.config, allow_download=True)
        if checkpoint is None:
            return {"backend": self.backend_name, "initialization": "random"}

        class _Encoder(nn.Module):
            def __init__(self, scene_backbone):
                super().__init__()
                self.scene_backbone = scene_backbone

        class _Target(nn.Module):
            def __init__(self, scene_backbone):
                super().__init__()
                self.encoder = _Encoder(scene_backbone)

        report = load_pretrained_backbone(_Target(self.scene_backbone), checkpoint, self.config)
        return {"backend": self.backend_name, **report}

    def forward(self, xyz, rgb):
        point_mask = xyz.new_ones(xyz.shape[:2], dtype=bool)
        return self.scene_backbone(xyz, rgb, point_mask).point_features


def build_push_backbone(
    backend: str,
    dim: int,
    config: BackboneConfig,
    *,
    activation_checkpointing: bool = False,
) -> nn.Module:
    if backend == "point_transformer_v3":
        return PushPointTransformerV3(
            dim,
            config,
            activation_checkpointing=activation_checkpointing,
        )
    if backend == "pointnet2":
        result = PushPointNet2(dim)
        result.backend_name = "pointnet2"
        return result
    raise ValueError(f"Unsupported PUSH backbone: {backend}")
