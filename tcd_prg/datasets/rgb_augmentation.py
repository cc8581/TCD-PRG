"""Composable online augmentation for point-cloud RGB features only."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.config import RGBAugmentationConfig


def _uniform(low: float, high: float, *, device: torch.device) -> Tensor:
    if low == high:
        return torch.tensor(low, device=device)
    return torch.empty((), device=device).uniform_(low, high)


def _chance(probability: float, *, device: torch.device) -> bool:
    return probability > 0 and bool(torch.rand((), device=device) < probability)


def _hue_rotate(rgb: Tensor, delta: float) -> Tensor:
    if not delta or rgb.numel() == 0:
        return rgb
    angle = float(delta) * 2.0 * math.pi
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = rgb.new_tensor(
        [
            [0.299 + 0.701 * cosine + 0.168 * sine,
             0.587 - 0.587 * cosine + 0.330 * sine,
             0.114 - 0.114 * cosine - 0.497 * sine],
            [0.299 - 0.299 * cosine - 0.328 * sine,
             0.587 + 0.413 * cosine + 0.035 * sine,
             0.114 - 0.114 * cosine + 0.292 * sine],
            [0.299 - 0.300 * cosine + 1.250 * sine,
             0.587 - 0.588 * cosine - 1.050 * sine,
             0.114 + 0.886 * cosine - 0.203 * sine],
        ]
    )
    return rgb @ matrix.T


class PointCloudRGBAugmentation:
    """Apply stochastic RGB transforms while preserving every non-RGB field."""

    def __init__(self, config: RGBAugmentationConfig) -> None:
        self.config = config

    def __call__(self, batch: dict[str, object]) -> dict[str, object]:
        rgb = batch.get("rgb")
        point_mask = batch.get("point_mask")
        if (
            not self.config.enabled
            or not isinstance(rgb, Tensor)
            or not isinstance(point_mask, Tensor)
        ):
            return batch
        instance_id = batch.get("instance_id")
        instance_id = instance_id if isinstance(instance_id, Tensor) else None
        if instance_id is not None and instance_id.shape != point_mask.shape:
            instance_id = None
        output = rgb.clone()
        for row in range(len(output)):
            valid = point_mask[row].bool()
            if valid.any():
                ids = instance_id[row] if instance_id is not None else None
                output[row, valid] = self._sample(
                    output[row, valid], None if ids is None else ids[valid]
                )
        output[~point_mask.bool()] = 0
        batch["rgb"] = output
        return batch

    def _sample(self, rgb: Tensor, instance_id: Tensor | None) -> Tensor:
        config, device = self.config, rgb.device
        value = rgb.float().clone().clamp(0, 1)
        base_draw = float(torch.rand((), device=device))
        zero_probability = config.zero_probability if config.zero_enabled else 0.0
        grayscale_probability = (
            config.grayscale_probability if config.grayscale_enabled else 0.0
        )
        if base_draw < zero_probability:
            return torch.zeros_like(value)
        if base_draw < zero_probability + grayscale_probability:
            gray = (value * value.new_tensor((0.299, 0.587, 0.114))).sum(-1, keepdim=True)
            value = gray.expand_as(value).clone()

        if config.color_jitter_enabled and _chance(config.color_jitter_probability, device=device):
            brightness = _uniform(1 - config.brightness, 1 + config.brightness, device=device)
            contrast = _uniform(1 - config.contrast, 1 + config.contrast, device=device)
            saturation = _uniform(1 - config.saturation, 1 + config.saturation, device=device)
            value = value * brightness
            mean = value.mean(dim=0, keepdim=True)
            value = (value - mean) * contrast + mean
            gray = (value * value.new_tensor((0.299, 0.587, 0.114))).sum(-1, keepdim=True)
            value = gray + (value - gray) * saturation
            value = _hue_rotate(value, float(_uniform(-config.hue, config.hue, device=device)))
            value = value.clamp(0, 1).pow(_uniform(*config.gamma, device=device))

        if (
            config.object_recolor_enabled
            and instance_id is not None
            and _chance(config.object_recolor_probability, device=device)
        ):
            for object_id in torch.unique(instance_id):
                if int(object_id) < 0:
                    continue
                selected = instance_id == object_id
                color = torch.rand(3, device=device)
                strength = _uniform(*config.object_recolor_strength, device=device)
                value[selected] = value[selected] * (1 - strength) + color * strength

        if config.material_jitter_enabled and _chance(
            config.material_jitter_probability, device=device
        ):
            std = _uniform(*config.material_noise_std, device=device)
            channel_scale = torch.empty(3, device=device).normal_(1.0, float(std))
            value = value * channel_scale + torch.randn_like(value) * std
        if config.lighting_jitter_enabled and _chance(
            config.lighting_jitter_probability, device=device
        ):
            value = value * _uniform(*config.lighting_gain, device=device)
            value = value + _uniform(*config.lighting_bias, device=device)
        if config.noise_enabled and _chance(config.noise_probability, device=device):
            value = value + torch.randn_like(value) * _uniform(*config.noise_std, device=device)
        if config.channel_dropout_enabled and _chance(
            config.channel_dropout_probability, device=device
        ):
            value[:, int(torch.randint(0, 3, (), device=device))] = 0
        if config.point_dropout_enabled and _chance(
            config.point_dropout_probability, device=device
        ):
            fraction = _uniform(*config.point_dropout_fraction, device=device)
            value[torch.rand(len(value), device=device) < fraction] = 0
        return value.clamp_(0, 1).to(rgb.dtype)
