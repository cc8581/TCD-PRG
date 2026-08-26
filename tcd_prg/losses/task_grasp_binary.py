"""Strict binary Stage-B objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TaskGraspBinaryLoss(nn.Module):
    def forward(
        self, prediction: dict[str, Tensor], label: Tensor, valid: Tensor
    ) -> dict[str, Tensor]:
        valid = valid.bool() & prediction["valid"].bool()
        logit = prediction["task_valid_logit"]
        if not bool(valid.any()):
            raise RuntimeError("Stage-B batch contains no valid binary candidates")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit[valid], label.float()[valid]
        )
        predicted = logit[valid] >= 0
        truth = label.bool()[valid]
        correct = (predicted == truth).float().mean()
        tp = (predicted & truth).float().sum()
        fp = (predicted & ~truth).float().sum()
        fn = (~predicted & truth).float().sum()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        return {
            "loss": loss,
            "task_grasp_binary_bce": loss.detach(),
            "task_grasp_binary_accuracy": correct.detach(),
            "task_grasp_binary_f1": f1.detach(),
            "task_grasp_supervised_candidates": valid.float().sum().detach(),
            "task_grasp_positive_fraction": truth.float().mean().detach(),
        }
