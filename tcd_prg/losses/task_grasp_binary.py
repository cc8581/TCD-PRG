"""Strict binary Stage-B objective."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from tcd_prg.evaluators.metrics import binary_auroc, binary_average_precision


def stageb_split_metrics(
    score: np.ndarray,
    target: np.ndarray,
    deployment_score: np.ndarray | None = None,
    deployment_target: np.ndarray | None = None,
) -> dict[str, float]:
    """Report raw evaluator quality and calibrate threshold after deployment selection."""
    score = np.asarray(score, np.float64)
    target = np.asarray(target, bool)
    if not len(score) or score.shape != target.shape:
        raise ValueError("Stage-B validation scores and targets must be aligned and non-empty")
    calibration_score = score if deployment_score is None else np.asarray(deployment_score, np.float64)
    calibration_target = target if deployment_target is None else np.asarray(deployment_target, bool)
    if not len(calibration_score) or calibration_score.shape != calibration_target.shape:
        raise ValueError("Stage-B deployment calibration scores and targets must align")
    has_deployable_positive = bool(calibration_target.any())
    if has_deployable_positive:
        thresholds = np.unique(calibration_score)
        best = (-1.0, 0.5, 0.0, 0.0)
        for threshold in thresholds:
            predicted = calibration_score >= threshold
            tp = int((predicted & calibration_target).sum())
            fp = int((predicted & ~calibration_target).sum())
            fn = int((~predicted & calibration_target).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if f1 > best[0]:
                best = (f1, float(threshold), precision, recall)
    else:
        # No selected candidate is a positive calibration example. Reject all
        # for this report, and let Trainer retain the last calibrated threshold.
        reject_all = float(np.nextafter(calibration_score.max(), np.inf))
        best = (0.0, reject_all, 0.0, 0.0)
    result = {
        "task_grasp_validation_f1": best[0],
        "task_grasp_validation_threshold": best[1],
        "task_grasp_validation_precision": best[2],
        "task_grasp_validation_recall": best[3],
        "task_grasp_validation_auroc": binary_auroc(score, target),
        "task_grasp_validation_auprc": binary_average_precision(score, target),
    }
    if deployment_score is not None:
        result.update({
            "task_grasp_raw_f1": _best_f1(score, target)[0],
            "task_grasp_deployment_selected_candidates": float(len(calibration_score)),
            "task_grasp_deployment_has_positive": float(has_deployable_positive),
        })
    return result


def _best_f1(score: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    best = (-1.0, 0.5)
    for threshold in np.unique(score):
        predicted = score >= threshold
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best


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
