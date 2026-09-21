import pytest
import torch
from torch import nn

from tcd_prg.config import BackboneConfig, load_config
from tcd_prg.models.push.backbone import PushPointTransformerV3


def test_push_stage_defaults_to_ptv3_and_allows_pointnet2_override() -> None:
    config = load_config("configs/stage/push_evaluator.yaml")
    assert config.training.push_backbone == "point_transformer_v3"

    fallback = load_config(
        "configs/stage/push_evaluator.yaml",
        ["training.push_backbone=pointnet2"],
    )
    assert fallback.training.push_backbone == "pointnet2"


def test_push_stage_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="training.push_backbone"):
        load_config(
            "configs/stage/push_evaluator.yaml",
            ["training.push_backbone=unknown"],
        )


class _FakeSceneBackbone(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(6, dim)

    def forward(self, xyz, rgb, point_mask):
        from tcd_prg.models.backbones.task_point_transformer import SceneGeometryOutput

        points = self.projection(torch.cat((xyz, rgb), -1))
        points = points * point_mask.unsqueeze(-1)
        return SceneGeometryOutput(points, points.mean(1))


def test_ptv3_push_adapter_preserves_dense_point_feature_contract() -> None:
    adapter = PushPointTransformerV3(
        16,
        BackboneConfig(),
        scene_backbone=_FakeSceneBackbone(16),
    )
    xyz = torch.randn(2, 11, 3)
    rgb = torch.randn_like(xyz)
    output = adapter(xyz, rgb)
    assert output.shape == (2, 11, 16)
    output.sum().backward()
    assert adapter.scene_backbone.projection.weight.grad is not None
