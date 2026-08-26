"""Strict binary Stage-B objective."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from tcd_prg.evaluators.metrics import binary_auroc, binary_average_precision


def stageb_split_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute candidate-level metrics once over the complete validation split."""
    score = np.asarray(score, np.float64)
    target = np.asarray(target, bool)
    if not len(score) or score.shape != target.shape:
        raise ValueError("Stage-B validation scores and targets must be aligned and non-empty")
    thresholds = np.unique(np.r_[0.0, score, 1.0])
    best = (-1.0, 0.5, 0.0, 0.0)
    for threshold in thresholds:
        predicted = score >= threshold
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best[0]:
            best = (f1, float(threshold), precision, recall)
    return {
        "task_grasp_validation_f1": best[0],
        "task_grasp_validation_threshold": best[1],
        "task_grasp_validation_precision": best[2],
        "task_grasp_validation_recall": best[3],
        "task_grasp_validation_auroc": binary_auroc(score, target),
        "task_grasp_validation_auprc": binary_average_precision(score, target),
    }


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
