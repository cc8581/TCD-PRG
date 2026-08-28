"""Resource-isolated Stage-C model containing only the PUSH branch."""

from __future__ import annotations
from typing import Any, Mapping
from torch import Tensor, nn
from tcd_prg.config import ModelConfig
from .push import PushEffectivenessEvaluator, PushHead
from .push_condition import PushCondition


class StandalonePushModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.push = PushHead(
            config.feature_dim,
            config.num_direction_bins,
            config.push_direction_feature_dim,
            config.push_direction_transformer_layers,
            config.push_direction_transformer_heads,
            config.push_direction_contact_topk,
            config.push_object_topk,
            config.num_categories,
            config.num_task_regions,
        )
        self.push_evaluator_ready = False
        self.push_evaluator = PushEffectivenessEvaluator(
            config.feature_dim, config.push_direction_feature_dim
        )

    @staticmethod
    def _sensor(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        source = batch.get("model_inputs", batch)
        return {key: source[key] for key in ("xyz", "rgb", "point_mask")}

    def forward(self, batch: Mapping[str, Any], *, forward_mode: str = "push") -> dict[str, Any]:
        if forward_mode != "push":
            raise ValueError("StandalonePushModel supports only forward_mode='push'")
        sensor = self._sensor(batch)
        condition = batch.get("push_condition")
        if not isinstance(condition, PushCondition):
            raise TypeError("Stage-C forward requires a PushCondition")
        hints = batch.get("training_hints") or {}
        push = self.push(sensor, condition, hints.get("push_direction_point_mask"))
        return {
            "push_condition": condition,
            "sensor": sensor,
            "task": {
                "task_category_id": condition.task_category_id,
                "task_region_id": condition.task_region_id,
            },
            "encoded": None,
            "instance": None,
            "region": None,
            "task_grasp": None,
            "global_grasp": None,
            "push": push,
        }
