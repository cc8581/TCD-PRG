from __future__ import annotations

import torch

from tcd_prg.models.backbones.task_point_transformer import (
    TaskFreeSceneGeometryBackbone,
    _farthest_point_indices,
    _local_knn,
)


def test_fps_is_invariant_to_input_order() -> None:
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.7, 0.0, 0.0],
                         [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    first, first_valid = _farthest_point_indices(xyz, mask, 3)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    shuffled = xyz[:, permutation]
    second, second_valid = _farthest_point_indices(shuffled, mask, 3)
    first_points = xyz[0, first[0, first_valid[0]]]
    second_points = shuffled[0, second[0, second_valid[0]]]
    assert torch.equal(first_points, second_points)


def test_local_knn_finds_nearby_points_without_global_cdist(monkeypatch) -> None:
    reference = torch.arange(16, dtype=torch.float32)[None, :, None].expand(-1, -1, 3)
    query = torch.tensor([[[0.1, 0.1, 0.1], [14.9, 14.9, 14.9]]])
    mask = torch.ones(1, 16, dtype=torch.bool)

    def reject_cdist(*args, **kwargs):
        raise AssertionError("global torch.cdist must not be used")

    monkeypatch.setattr(torch, "cdist", reject_cdist)
    index = _local_knn(query, reference, mask, 1)
    assert index.squeeze(-1).tolist() == [[0, 15]]


def test_local_knn_preserves_useful_neighborhood_recall() -> None:
    torch.manual_seed(19)
    xyz = torch.rand(1, 512, 3)
    mask = torch.ones(1, 512, dtype=torch.bool)
    approximate = _local_knn(xyz, xyz, mask, 16)
    exact = torch.cdist(xyz, xyz).topk(16, largest=False).indices
    overlap = sum(
        len(set(predicted.tolist()) & set(target.tolist()))
        for predicted, target in zip(approximate[0], exact[0], strict=True)
    )
    assert overlap / (512 * 16) >= 0.80


def test_backbone_forward_does_not_materialize_quadratic_distance_matrix(monkeypatch) -> None:
    backbone = TaskFreeSceneGeometryBackbone(
        dim=16, blocks=1, heads=4, neighbors=4,
        attention_points=8, activation_checkpointing=False,
    ).eval()
    xyz = torch.randn(2, 24, 3)
    rgb = torch.rand(2, 24, 3)
    point_mask = torch.ones(2, 24, dtype=torch.bool)

    def reject_cdist(*args, **kwargs):
        raise AssertionError("global torch.cdist must not be used")

    monkeypatch.setattr(torch, "cdist", reject_cdist)
    with torch.no_grad():
        output = backbone(xyz, rgb, point_mask)
    assert output.point_features.shape == (2, 24, 16)
