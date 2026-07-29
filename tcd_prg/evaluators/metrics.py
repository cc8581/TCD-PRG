"""Dependency-light metric kernels used by every baseline and ablation."""

from __future__ import annotations

import math

import numpy as np


def binary_f1(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    mask = np.ones(target.shape, bool) if valid is None else np.asarray(valid, bool)
    tp = np.count_nonzero(prediction & target & mask)
    fp = np.count_nonzero(prediction & ~target & mask)
    fn = np.count_nonzero(~prediction & target & mask)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else float("nan")


def binary_auroc(score: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> float:
    score, target = np.asarray(score, float), np.asarray(target, bool)
    mask = np.isfinite(score)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    score, target = score[mask], target[mask]
    positives, negatives = np.count_nonzero(target), np.count_nonzero(~target)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = target[order]
    tp = np.r_[0, np.cumsum(y)] / positives
    fp = np.r_[0, np.cumsum(~y)] / negatives
    return float(np.trapz(tp, fp))


def macro_f1(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        value = binary_f1(prediction == label, target == label, valid)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def intersection_over_union(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    prediction, target, valid = np.asarray(prediction, bool), np.asarray(target, bool), np.asarray(valid, bool)
    intersection = np.count_nonzero(prediction & target & valid)
    union = np.count_nonzero((prediction | target) & valid)
    return float(intersection / union) if union else float("nan")


def dice_score(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    prediction, target, valid = np.asarray(prediction, bool), np.asarray(target, bool), np.asarray(valid, bool)
    intersection = np.count_nonzero(prediction & target & valid)
    denominator = np.count_nonzero(prediction & valid) + np.count_nonzero(target & valid)
    return float(2 * intersection / denominator) if denominator else float("nan")


def ndcg(scores: np.ndarray, relevance: np.ndarray, valid: np.ndarray, k: int | None = None) -> float:
    scores, relevance, valid = np.asarray(scores), np.asarray(relevance, float), np.asarray(valid, bool)
    scores, relevance = scores[valid], relevance[valid]
    if not len(scores):
        return float("nan")
    k = min(k or len(scores), len(scores))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    actual = np.argsort(-scores)[:k]
    ideal = np.argsort(-relevance)[:k]
    dcg = np.sum((np.power(2.0, relevance[actual]) - 1.0) * discount)
    idcg = np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discount)
    return float(dcg / idcg) if idcg > 0 else float("nan")


def direction_angle_error_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    prediction /= max(np.linalg.norm(prediction), 1e-12)
    target /= max(np.linalg.norm(target), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(prediction, target), -1.0, 1.0))))
