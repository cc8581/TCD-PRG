"""Exponential moving average weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        for ema, current in zip(self.model.parameters(), source.parameters(), strict=True):
            ema.lerp_(current.detach(), 1.0 - self.decay)
        for ema, current in zip(self.model.buffers(), source.buffers(), strict=True):
            ema.copy_(current)

