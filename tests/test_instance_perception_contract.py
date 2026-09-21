from __future__ import annotations

import torch

from tcd_prg.config import ModelConfig
from tcd_prg.losses.instance import InstanceSetLoss, build_instance_targets
from tcd_prg.models.instance_segmentation import InstanceMaskDecoder
from tcd_prg.models.instance_segmentation.decoder import InstanceQueryOutput


def test_instance_head_uses_sensor_features_only():
    config = ModelConfig(
        feature_dim=32,
        task_dim=16,
        num_categories=20,
        num_task_regions=20,
        instance_queries=8,
        instance_decoder_heads=4,
        instance_decoder_layers=1,
    )
    head = InstanceMaskDecoder(
        32, config.instance_queries, config.num_categories,
        layers=config.instance_decoder_layers,
        heads=config.instance_decoder_heads,
        objectness_threshold=config.instance_objectness_threshold,
    )
    output = head(
        torch.randn(2, 24, 32),
        torch.randn(2, 24, 3),
        torch.ones(2, 24, dtype=torch.bool),
    )
    assert output.mask_logits.shape == (2, 8, 24)
    assert output.object_tokens.shape == (2, 8, 32)
    assert output.category_logits.shape == (2, 8, 20)
    assert output.object_mask.any(-1).all()
    assert output.point_offsets.shape == (2, 24, 3)
    assert not any("offset" in name for name in head.state_dict())


def test_instance_gt_is_loss_side_only(tiny_batch):
    tiny_batch = dict(tiny_batch)
    tiny_batch["object_category_id"] = torch.tensor([[0, 1, 2]], dtype=torch.long)
    config = ModelConfig(
        feature_dim=32,
        task_dim=16,
        num_categories=20,
        num_task_regions=20,
        instance_queries=8,
        instance_decoder_heads=4,
        instance_decoder_layers=1,
    )
    head = InstanceMaskDecoder(
        32, 8, 20, layers=1, heads=4, objectness_threshold=0.5
    )
    pred = head(
        torch.randn(1, 24, 32),
        tiny_batch["xyz"],
        tiny_batch["point_mask"],
    )
    targets = build_instance_targets(tiny_batch, 8)
    values, match = InstanceSetLoss(matching_points=24)(pred, targets)
    assert torch.isfinite(values["loss"])
    assert match.gt_to_query.shape[1] == tiny_batch["object_mask"].shape[1]


def test_adjacent_foreign_points_are_harder_than_distant_foreign_points():
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
                         [0.011, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    batch = {
        "xyz": xyz,
        "instance_id": torch.tensor([[0, 0, 1, 1]]),
        "point_mask": torch.ones(1, 4, dtype=torch.bool),
        "object_category_id": torch.tensor([[0, 1]]),
        "object_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_object": torch.tensor([0]),
        "target_mask": torch.tensor([[True, True, False, False]]),
    }
    targets = build_instance_targets(batch, 2)
    loss_fn = InstanceSetLoss(matching_points=4)

    def prediction(foreign_index):
        logits = torch.tensor([[[5.0, 5.0, -5.0, -5.0],
                                [-5.0, -5.0, 5.0, 5.0]]])
        logits[0, 0, foreign_index] = 5.0
        return InstanceQueryOutput(
            object_tokens=torch.zeros(1, 2, 4),
            mask_logits=logits,
            mask_probability=logits.sigmoid(),
            objectness_logits=torch.full((1, 2), 5.0),
            category_logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
            object_mask=torch.ones(1, 2, dtype=torch.bool),
            centers_world=torch.zeros(1, 2, 3),
        )

    near, _ = loss_fn(prediction(2), targets)
    far, _ = loss_fn(prediction(3), targets)
    assert near["loss"] > far["loss"]
    assert near.keys() == far.keys()


def test_per_point_offsets_supervise_own_instance_centers():
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0],
                         [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]]])
    batch = {
        "xyz": xyz,
        "instance_id": torch.tensor([[0, 0, 1, 1]]),
        "point_mask": torch.ones(1, 4, dtype=torch.bool),
        "object_category_id": torch.tensor([[0, 1]]),
        "object_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_object": torch.tensor([0]),
        "target_mask": torch.tensor([[True, True, False, False]]),
    }
    targets = build_instance_targets(batch, 2)
    logits = torch.tensor([[[5.0, 5.0, -5.0, -5.0],
                            [-5.0, -5.0, 5.0, 5.0]]])
    output = InstanceQueryOutput(
        object_tokens=torch.zeros(1, 2, 4),
        mask_logits=logits,
        mask_probability=logits.sigmoid(),
        objectness_logits=torch.full((1, 2), 5.0),
        category_logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
        object_mask=torch.ones(1, 2, dtype=torch.bool),
        centers_world=torch.zeros(1, 2, 3),
        point_offsets=torch.tensor([[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0],
                                     [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]]),
    )
    loss_fn = InstanceSetLoss(matching_points=4)
    correct, _ = loss_fn(output, targets)
    output.point_offsets = -output.point_offsets
    reversed_direction, _ = loss_fn(output, targets)
    assert reversed_direction["loss"] > correct["loss"]
