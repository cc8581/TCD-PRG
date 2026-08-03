"""Numerically stable metric kernels shared by training and offline evaluation."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _filtered_binary(
    score: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    mask = np.isfinite(score)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    return score[mask], target[mask]


def binary_confusion(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None
) -> tuple[int, int, int, int]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    mask = np.ones(target.shape, bool) if valid is None else np.asarray(valid, bool)
    return (
        int(np.count_nonzero(prediction & target & mask)),
        int(np.count_nonzero(prediction & ~target & mask)),
        int(np.count_nonzero(~prediction & target & mask)),
        int(np.count_nonzero(~prediction & ~target & mask)),
    )


def confusion_metrics(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float]:
    del tn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    dice = f1
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "iou": float(iou), "dice": float(dice),
    }


def binary_f1(prediction: np.ndarray, target: np.ndarray,
              valid: np.ndarray | None = None) -> float:
    return confusion_metrics(*binary_confusion(prediction, target, valid))["f1"]


def binary_auroc(score: np.ndarray, target: np.ndarray,
                 valid: np.ndarray | None = None) -> float:
    """scikit-learn ROC AUC after applying the shared validity mask."""

    score, target = _filtered_binary(score, target, valid)
    positives, negatives = int(target.sum()), int((~target).sum())
    if not positives or not negatives:
        return float("nan")
    return float(roc_auc_score(target, score))


def binary_average_precision(score: np.ndarray, target: np.ndarray,
                             valid: np.ndarray | None = None) -> float:
    """scikit-learn non-interpolated average precision."""

    score, target = _filtered_binary(score, target, valid)
    positives = int(target.sum())
    if not positives:
        return float("nan")
    return float(average_precision_score(target, score))


def brier_score(probability: np.ndarray, target: np.ndarray,
                valid: np.ndarray | None = None) -> float:
    probability, target = _filtered_binary(probability, target, valid)
    return float(brier_score_loss(target, probability)) if len(probability) else float("nan")


def expected_calibration_error(
    probability: np.ndarray, target: np.ndarray,
    valid: np.ndarray | None = None, bins: int = 15,
) -> float:
    probability, target = _filtered_binary(probability, target, valid)
    if not len(probability):
        return float("nan")
    probability = np.clip(probability, 0.0, 1.0)
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            error += selected.mean() * abs(
                probability[selected].mean() - target[selected].mean()
            )
    return float(error)


def macro_f1(prediction: np.ndarray, target: np.ndarray,
             valid: np.ndarray, classes: int) -> float:
    values = [
        binary_f1(prediction == label, target == label, valid)
        for label in range(classes)
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def intersection_over_union(prediction: np.ndarray, target: np.ndarray,
                            valid: np.ndarray) -> float:
    return confusion_metrics(*binary_confusion(prediction, target, valid))["iou"]


def dice_score(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    return confusion_metrics(*binary_confusion(prediction, target, valid))["dice"]


def ndcg(scores: np.ndarray, relevance: np.ndarray,
         valid: np.ndarray, k: int | None = None) -> float:
    scores, relevance, valid = np.asarray(scores), np.asarray(relevance, float), np.asarray(valid, bool)
    scores, relevance = scores[valid], relevance[valid]
    if not len(scores):
        return float("nan")
    k = min(k or len(scores), len(scores))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    actual = np.argsort(-scores, kind="mergesort")[:k]
    ideal = np.argsort(-relevance, kind="mergesort")[:k]
    dcg = np.sum((np.power(2.0, relevance[actual]) - 1.0) * discount)
    idcg = np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discount)
    return float(dcg / idcg) if idcg > 0 else float("nan")


def recall_at_k(scores: np.ndarray, target: np.ndarray,
                valid: np.ndarray, k: int) -> float:
    scores, target, valid = np.asarray(scores), np.asarray(target, bool), np.asarray(valid, bool)
    positives = target & valid
    if not positives.any():
        return float("nan")
    indices = np.flatnonzero(valid)
    ranked = indices[np.argsort(-scores[indices], kind="mergesort")[:k]]
    return float(np.count_nonzero(positives[ranked]) / np.count_nonzero(positives))


def direction_angle_error_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    prediction /= max(np.linalg.norm(prediction), 1e-12)
    target /= max(np.linalg.norm(target), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(prediction, target), -1.0, 1.0))))
