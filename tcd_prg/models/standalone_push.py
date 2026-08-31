"""Resource-isolated Stage-C model containing only the PUSH branch."""

from __future__ import annotations
from typing import Any, Mapping
from torch import Tensor, nn
import hashlib

from pathlib import Path

import torch

from tcd_prg.config import BackboneConfig, ModelConfig
from .backbones import PointTransformerV3SceneGeometryBackbone
from .backbones.task_point_transformer import TaskFreeSceneGeometryBackbone
from .push import PushEffectivenessEvaluator, RulePushGenerator
from .push_condition import PushCondition


class StandalonePushModel(nn.Module):
    def __init__(self, config: ModelConfig, backbone: BackboneConfig | None = None) -> None:
        super().__init__()
        backbone = backbone or BackboneConfig(backend="legacy")
        if backbone.backend == "point_transformer_v3":
            self.geometry_encoder = PointTransformerV3SceneGeometryBackbone(
                dim=config.feature_dim,
                source_root=backbone.source_root,
                grid_size_m=backbone.grid_size_m,
                enable_flash_attention=backbone.enable_flash_attention,
                patch_size=backbone.patch_size,
                activation_checkpointing=False,
            )
        else:
            self.geometry_encoder = TaskFreeSceneGeometryBackbone(
                dim=config.feature_dim,
                attention_points=backbone.attention_points,
                activation_checkpointing=False,
            )
        self.geometry_encoder.requires_grad_(False).eval()
        self.perception_geometry_fingerprint: str | None = None
        self.push = RulePushGenerator(config)
        self.push_evaluator_ready = False
        self.push_evaluator = PushEffectivenessEvaluator(
            config.feature_dim, config.num_categories, config.num_task_regions
        )

    @staticmethod
    def _sensor(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        source = batch.get("model_inputs", batch)
        sensor = {
            key: source[key]
            for key in ("xyz", "rgb", "point_mask", "grid_coord", "geometry_feature")
            if key in source
        }
        if "geometry_feature" in batch:
            sensor["geometry_feature"] = batch["geometry_feature"]
        return sensor

    def train(self, mode: bool = True):
        super().train(mode)
        self.geometry_encoder.eval()
        return self

    def load_perception_geometry(self, checkpoint: str | Path) -> str:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("training_stage") != "perception" or int(payload.get("schema_version", -1)) != 12:
            raise RuntimeError("Stage-C geometry requires a schema-12 perception checkpoint")
        source = payload.get("ema") or payload["model"]
        prefix = "encoder.scene_backbone."
        supplied = {name[len(prefix):]: value for name, value in source.items() if name.startswith(prefix)}
        expected = self.geometry_encoder.state_dict()
        if set(supplied) != set(expected):
            raise RuntimeError("Perception checkpoint geometry encoder does not match Stage-C")
        self.geometry_encoder.load_state_dict(supplied, strict=True)
        digest = hashlib.sha256()
        for name in sorted(supplied):
            tensor = supplied[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        self.perception_geometry_fingerprint = digest.hexdigest()
        return self.perception_geometry_fingerprint

    def encode_scene(self, batch):
        sensor = self._sensor(batch)
        if "geometry_feature" not in sensor:
            if self.perception_geometry_fingerprint is None:
                raise RuntimeError("Load perception_checkpoint before PUSH evaluation")
            with torch.no_grad():
                sensor["geometry_feature"] = self.geometry_encoder(
                    sensor["xyz"], sensor["rgb"], sensor["point_mask"],
                    grid_coord=sensor.get("grid_coord"),
                ).point_features.detach()
        return sensor

    def score_actions(self, batch, condition, actions):
        return self.push_evaluator(self.encode_scene(batch), condition, actions)

    def forward(self, batch, *, forward_mode="push"):
        if forward_mode != "push":
            raise ValueError("StandalonePushModel supports only push")
        if self.training:
            raise RuntimeError("Use score_actions with logged actions for training; rules are inference-only")
        sensor = self.encode_scene(batch)
        condition = batch["push_condition"]
        actions = self.push(sensor, condition)
        logits = self.push_evaluator(sensor, condition, actions)
        return {"sensor": sensor, "push_condition": condition,
                "push": {"actions": actions, "effective_logit": logits}}
