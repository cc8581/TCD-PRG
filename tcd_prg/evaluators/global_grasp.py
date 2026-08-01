"""Fair two-track evaluation for task-free global grasp prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tcd_prg.baselines.base import GlobalGraspPrediction
from tcd_prg.datasets.types import GlobalGraspLabels
from tcd_prg.geometry.numpy_se3 import quaternion_xyzw_to_matrix_numpy


@dataclass(frozen=True, slots=True)
class GlobalGraspMatchConfig:
    translation_m: float = 0.01
    rotation_deg: float = 15.0
    width_m: float = 0.005
    require_same_instance: bool = True
    apply_nms: bool = True


def _parallel_jaw_rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """SO(3) distance with 180-degree closing-axis gripper symmetry."""

    distances = []
    # Avoid NumPy BLAS here: native Windows PyTorch + NumPy OpenMP runtimes can
    # abort the process instead of raising an exception.
    for signs in ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)):
        trace_relative = 0.0
        for column in range(3):
            trace_relative += signs[column] * sum(
                float(first[row, column] * second[row, column]) for row in range(3)
            )
        cosine = np.clip((trace_relative - 1.0) * 0.5, -1.0, 1.0)
        distances.append(float(np.degrees(np.arccos(cosine))))
    return min(distances)


class GlobalGraspEvaluator:
    """Evaluate the unified scene-executable grasp quality before/after exact certification."""

    def __init__(self, config: GlobalGraspMatchConfig | None = None) -> None:
        self.config = config or GlobalGraspMatchConfig()

    def _nms(
        self, predictions: list[GlobalGraspPrediction], *, certified: bool
    ) -> list[GlobalGraspPrediction]:
        del certified
        ranked = sorted(predictions, key=lambda item: item.scene_score, reverse=True)
        if not self.config.apply_nms:
            return ranked
        selected: list[GlobalGraspPrediction] = []
        for candidate in ranked:
            duplicate = False
            candidate_rotation = quaternion_xyzw_to_matrix_numpy(candidate.grasp_pose_world[3:])
            for prior in selected:
                if candidate.object_index < 0 or candidate.object_index != prior.object_index:
                    continue
                delta = candidate.grasp_pose_world[:3] - prior.grasp_pose_world[:3]
                translation = float(np.sqrt(np.sum(delta * delta)))
                rotation = _parallel_jaw_rotation_distance_deg(
                    candidate_rotation, quaternion_xyzw_to_matrix_numpy(prior.grasp_pose_world[3:])
                )
                if (
                    translation <= self.config.translation_m
                    and rotation <= self.config.rotation_deg
                    and abs(candidate.width_m - prior.width_m) <= self.config.width_m
                ):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
        return selected

    def _matches(
        self, prediction: GlobalGraspPrediction, labels: GlobalGraspLabels, positive: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.flatnonzero(positive)
        if self.config.require_same_instance:
            indices = indices[labels.object_index[indices] == prediction.object_index]
        if not len(indices):
            empty = np.empty(0, np.float32)
            return indices, empty, empty
        delta = labels.grasp_pose_world[indices, :3] - prediction.grasp_pose_world[:3]
        translation = np.sqrt(np.sum(delta * delta, axis=-1))
        predicted_rotation = quaternion_xyzw_to_matrix_numpy(prediction.grasp_pose_world[3:])
        rotation = np.asarray([
            _parallel_jaw_rotation_distance_deg(
                predicted_rotation, quaternion_xyzw_to_matrix_numpy(labels.grasp_pose_world[index, 3:])
            )
            for index in indices
        ], np.float32)
        width = np.abs(labels.width_m[indices] - prediction.width_m)
        valid = (
            (translation <= self.config.translation_m)
            & (rotation <= self.config.rotation_deg)
            & (width <= self.config.width_m)
        )
        return indices[valid], translation[valid], rotation[valid]

    def evaluate(
        self, predictions: list[GlobalGraspPrediction], labels: GlobalGraspLabels,
        *, certified: bool, topk: tuple[int, ...] = (1, 5, 10, 50),
    ) -> dict[str, float]:
        positive = labels.valid_mask & (labels.scene_executable == 1)
        if certified:
            considered_predictions = [item for item in predictions if item.certified]
            prefix = "certified"
        else:
            considered_predictions = predictions
            prefix = "scene"
        ranked = self._nms(considered_predictions, certified=certified)
        matched_truth: set[int] = set()
        true_positive = []
        translation_errors = []
        rotation_errors = []
        width_errors = []
        matched_objects: set[int] = set()
        for prediction in ranked:
            matches, translation, rotation = self._matches(prediction, labels, positive)
            unmatched = [index for index in matches.tolist() if index not in matched_truth]
            success = bool(unmatched)
            true_positive.append(success)
            if success:
                chosen = unmatched[0]
                matched_truth.add(chosen)
                local = int(np.flatnonzero(matches == chosen)[0])
                translation_errors.append(float(translation[local]))
                rotation_errors.append(float(rotation[local]))
                width_errors.append(abs(float(labels.width_m[chosen]) - prediction.width_m))
                matched_objects.add(int(labels.object_index[chosen]))
        tp = np.asarray(true_positive, np.float32)
        precision = np.cumsum(tp) / np.arange(1, len(tp) + 1) if len(tp) else np.empty(0)
        recall = np.cumsum(tp) / max(1, int(positive.sum())) if len(tp) else np.empty(0)
        ap = float(np.sum(precision * tp) / max(1, int(positive.sum()))) if len(tp) else 0.0
        gt_objects = set(int(value) for value in np.unique(labels.object_index[positive]))
        valid_ranked = [prediction for prediction in ranked if prediction.object_index >= 0]
        nearest_translation: list[float] = []
        nearest_rotation: list[float] = []
        approach_bins: set[int] = set()
        for index, prediction in enumerate(valid_ranked):
            rotation_a = quaternion_xyzw_to_matrix_numpy(prediction.grasp_pose_world[3:])
            approach = rotation_a[:, 2]
            axis = int(np.argmax(np.abs(approach)))
            approach_bins.add(axis * 2 + int(approach[axis] >= 0))
            distances = []
            angles = []
            for other_index, other in enumerate(valid_ranked):
                if index == other_index or prediction.object_index != other.object_index:
                    continue
                delta = prediction.grasp_pose_world[:3] - other.grasp_pose_world[:3]
                distances.append(float(np.sqrt(np.sum(delta * delta))))
                angles.append(_parallel_jaw_rotation_distance_deg(
                    rotation_a, quaternion_xyzw_to_matrix_numpy(other.grasp_pose_world[3:])
                ))
            if distances:
                nearest_translation.append(min(distances))
                nearest_rotation.append(min(angles))
        result = {
            f"{prefix}_ap": ap,
            f"{prefix}_object_coverage": float(len(matched_objects & gt_objects) / max(1, len(gt_objects))),
            f"{prefix}_translation_error_m": float(np.mean(translation_errors)) if translation_errors else float("nan"),
            f"{prefix}_rotation_error_deg": float(np.mean(rotation_errors)) if rotation_errors else float("nan"),
            f"{prefix}_width_error_m": float(np.mean(width_errors)) if width_errors else float("nan"),
            f"{prefix}_unique_match_ratio": float(len(matched_truth) / max(1, len(ranked))),
            f"{prefix}_nms_candidate_count": float(len(ranked)),
            f"{prefix}_mean_nearest_translation_m": float(np.mean(nearest_translation)) if nearest_translation else float("nan"),
            f"{prefix}_mean_nearest_rotation_deg": float(np.mean(nearest_rotation)) if nearest_rotation else float("nan"),
            f"{prefix}_approach_direction_coverage": float(len(approach_bins) / 6.0),
            f"{prefix}_grasp_clusters_per_object": float(len(valid_ranked) / max(1, len({p.object_index for p in valid_ranked}))),
        }
        for k in topk:
            result[f"{prefix}_recall@{k}"] = float(recall[min(k, len(recall)) - 1]) if len(recall) else 0.0
            result[f"{prefix}_precision@{k}"] = float(tp[:k].mean()) if len(tp[:k]) else 0.0
        return result
