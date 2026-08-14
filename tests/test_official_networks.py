from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tcd_prg.config import ModelConfig
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.backbones import point_transformer_v3 as ptv3_adapter


class _FakeOfficialPTv3(nn.Module):
    """Cheap contract double; official CUDA execution is covered by smoke profiling."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        assert kwargs["in_channels"] == 6
        self.projection = nn.Linear(6, 64)

    def forward(self, data):
        return SimpleNamespace(feat=self.projection(data["feat"]))


def test_ptv3_adapter_voxelizes_and_restores_dense_point_alignment(monkeypatch) -> None:
    monkeypatch.setattr(
        ptv3_adapter, "_load_official_model", lambda source_root: _FakeOfficialPTv3
    )
    backbone = ptv3_adapter.PointTransformerV3SceneGeometryBackbone(
        dim=16,
        source_root="unused",
        grid_size_m=0.01,
        enable_flash_attention=False,
        patch_size=32,
        activation_checkpointing=False,
    )
    backbone.eval()
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    output = backbone(
        xyz,
        torch.rand(1, 3, 3),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert output.point_features.shape == (1, 3, 16)
    # The first two samples share one voxel and therefore one PTv3 feature.
    assert torch.equal(output.point_features[:, 0], output.point_features[:, 1])


def test_ptv3_adapter_packs_variable_scene_lengths_after_grid_sample(monkeypatch) -> None:
    monkeypatch.setattr(
        ptv3_adapter, "_load_official_model", lambda source_root: _FakeOfficialPTv3
    )
    backbone = ptv3_adapter.PointTransformerV3SceneGeometryBackbone(
        dim=16,
        source_root="unused",
        grid_size_m=0.01,
        enable_flash_attention=False,
        patch_size=32,
        activation_checkpointing=False,
    ).eval()
    xyz = torch.tensor([
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
    ])
    rgb = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    data, inverse, flat_valid = backbone._voxelize(xyz, rgb, mask)
    assert data["coord"].shape == (3, 3)
    assert data["batch"].tolist() == [0, 0, 1]
    assert inverse.tolist() == [0, 0, 1, 2]
    assert flat_valid.sum().item() == 4
    # Evaluation follows official GridSample semantics deterministically: the
    # first point in an occupied voxel is used as its representative.
    assert torch.equal(data["coord"][0], xyz[0, 0])


def test_task_and_global_grasp_use_one_shared_decoder() -> None:
    model = TCDPRGModel(
        ModelConfig(
            feature_dim=16,
            task_dim=8,
            grasp_decoder_heads=8,
            verifier_transformer_heads=8,
        )
    )
    decoder_keys = [key for key in model.state_dict() if "grasp_decoder.decoder" in key]
    assert decoder_keys
    assert not any("task_grasp.decoder" in key for key in model.state_dict())
    assert not any("global_grasp.decoder" in key for key in model.state_dict())
