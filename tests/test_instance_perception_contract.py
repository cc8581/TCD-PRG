from __future__ import annotations

import torch

from tcd_prg.config import ModelConfig
from tcd_prg.models.instance_segmentation import InstanceMaskDecoder
from tcd_prg.losses.instance import InstanceSetLoss, build_instance_targets


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
