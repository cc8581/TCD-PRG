"""Object-model keyed functional-region and canonical grasp registries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tcd_prg.geometry.numpy_se3 import quaternion_xyzw_to_matrix_numpy


@dataclass(frozen=True, slots=True)
class GraspLibrary:
    source_index: np.ndarray
    task_label: np.ndarray
    transform_object: np.ndarray
    contact_points_object: np.ndarray
    contact_span_m: np.ndarray
    confidence: np.ndarray
    depth_m: np.ndarray

    @property
    def ag_width_m(self) -> np.ndarray:
        """Expose the existing contact span under the proposal sampler's width name."""

        return self.contact_span_m

    @property
    def quality(self) -> np.ndarray:
        """Use the source grasp confidence for deterministic diversity sampling."""

        return self.confidence

    def rows_for_source(self, source_indices: np.ndarray) -> np.ndarray:
        mapping = {int(value): row for row, value in enumerate(self.source_index)}
        try:
            return np.asarray([mapping[int(value)] for value in source_indices], dtype=np.int64)
        except KeyError as error:
            raise KeyError(f"Unknown grasp source index {error.args[0]}") from error


class GraspLibraryRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @lru_cache(maxsize=512)
    def load(self, relative_match_file: str) -> GraspLibrary:
        path = self.root / relative_match_file
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            required = {
                "source_grasp_index",
                "task_label",
                "canonical_transform_object",
                "contact_points_object",
                "contact_span",
                "confidence",
            }
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"{path} missing {sorted(missing)}")
            canonical = data["canonical_transform_object"].astype(np.float32)
            source = data["source_transform_object"].astype(np.float32)
            translation_delta = canonical[:, :3, 3] - source[:, :3, 3]
            depth_m = np.sum(
                np.transpose(source[:, :3, :3], (0, 2, 1)) * translation_delta[:, None, :], axis=2
            )[:, 2]
            return GraspLibrary(
                source_index=data["source_grasp_index"].astype(np.int64),
                task_label=data["task_label"].astype(np.int64),
                transform_object=canonical,
                contact_points_object=data["contact_points_object"].astype(np.float32),
                contact_span_m=data["contact_span"].astype(np.float32),
                confidence=data["confidence"].astype(np.float32),
                depth_m=depth_m.astype(np.float32),
            )


@dataclass(frozen=True, slots=True)
class FunctionalRegion:
    xyz_object: np.ndarray
    labels: np.ndarray
    region_keys: tuple[str, ...]


class FunctionalRegionRegistry:
    def __init__(self, root: str | Path, association_tolerance_m: float = 0.006) -> None:
        self.root = Path(root)
        self.association_tolerance_m = association_tolerance_m

    @lru_cache(maxsize=512)
    def load(self, category_key: str, model_id: str) -> FunctionalRegion:
        path = self.root / category_key / f"{model_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            keys = tuple(str(x) for x in data["region_keys"])
            return FunctionalRegion(data["point_xyz"].astype(np.float32), data["point_labels"].astype(np.int64), keys)

    @lru_cache(maxsize=512)
    def _spatial_index(
        self, category_key: str, model_id: str, scale_key: float
    ) -> tuple[np.ndarray, cKDTree]:
        """Build one immutable nearest-neighbor index per scaled CAD object."""

        points = self.load(category_key, model_id).xyz_object * scale_key
        return points, cKDTree(points)

    def visible_labels(
        self,
        xyz_world: np.ndarray,
        object_pose_xyzw: np.ndarray,
        category_key: str,
        model_id: str,
        object_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Associate visible points to full CAD labels; unmatched points are invalid."""

        region = self.load(category_key, model_id)
        rotation = quaternion_xyzw_to_matrix_numpy(object_pose_xyzw[3:]).astype(np.float32)
        centered = xyz_world - object_pose_xyzw[:3]
        xyz_object = np.empty_like(centered, dtype=np.float32)
        for column in range(3):
            xyz_object[:, column] = np.sum(centered * rotation[:, column], axis=1)
        if not np.isfinite(object_scale) or object_scale <= 0:
            raise ValueError(f"Invalid object scale {object_scale}")
        scale_key = round(float(object_scale), 12)
        _, index = self._spatial_index(category_key, model_id, scale_key)
        distance, nearest = index.query(
            xyz_object,
            k=1,
            distance_upper_bound=self.association_tolerance_m,
            workers=1,
        )
        valid = np.isfinite(distance)
        labels = np.full(len(xyz_world), -1, dtype=np.int64)
        labels[valid] = region.labels[nearest[valid]]
        return labels, valid
