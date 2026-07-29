"""Conversion of original GAPG arrays into the common observation contract."""

from __future__ import annotations

import numpy as np

from .types import CameraParameters, SceneObservation


class GAPGObservationAdapter:
    """Build a common observation without importing or modifying GAPG source."""

    @staticmethod
    def from_fused_points(
        xyz: np.ndarray,
        rgb: np.ndarray,
        instance_id: np.ndarray,
        target_object: int,
        camera_parameters: tuple[CameraParameters, ...],
        *,
        scene_id: int = 0,
        state_id: int = 0,
        object_category_id: np.ndarray | None = None,
    ) -> SceneObservation:
        instance = np.asarray(instance_id, dtype=np.int32)
        object_count = int(instance[instance >= 0].max(initial=-1)) + 1
        categories = (np.full(object_count, -1, np.int64) if object_category_id is None
                      else np.asarray(object_category_id, np.int64))
        observation = SceneObservation(
            scene_id=scene_id,
            state_id=state_id,
            task_index=0,
            xyz=np.asarray(xyz, np.float32),
            rgb=np.asarray(rgb, np.float32),
            instance_id=instance,
            target_mask=instance == target_object,
            target_object=int(target_object),
            task_region_id=0,
            object_uuid=tuple(f"scene_{scene_id:04d}/object_{index:02d}"
                              for index in range(object_count)),
            object_pose=np.full((object_count, 7), np.nan, np.float32),
            object_category_id=categories,
            object_present=np.asarray([np.any(instance == index)
                                       for index in range(object_count)], bool),
            object_active=np.asarray([np.any(instance == index)
                                      for index in range(object_count)], bool),
            camera_parameters=camera_parameters,
            point_valid=np.isfinite(xyz).all(axis=1),
            metadata={"source": "gapg_three_pro_s_wrapper"},
        )
        observation.validate()
        return observation
