"""Explicit frame-safe geometry utilities with lazy PyTorch imports."""

from typing import Any

from .numpy_se3 import (
    compose_pose_with_transform,
    matrix_to_quaternion_xyzw_numpy,
    quaternion_xyzw_to_matrix_numpy,
)

__all__ = [
    "SE3",
    "ExactAG16095GeometryProvider",
    "compose_pose_with_transform",
    "matrix_to_quaternion_xyzw",
    "matrix_to_quaternion_xyzw_numpy",
    "quaternion_xyzw_to_matrix",
    "quaternion_xyzw_to_matrix_numpy",
]


def __getattr__(name: str) -> Any:
    """Load tensor geometry only when a model-side caller requests it."""

    if name == "ExactAG16095GeometryProvider":
        from .gripper_provider import ExactAG16095GeometryProvider

        return ExactAG16095GeometryProvider
    if name in {"SE3", "matrix_to_quaternion_xyzw", "quaternion_xyzw_to_matrix"}:
        from .se3 import SE3, matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix

        return {
            "SE3": SE3,
            "matrix_to_quaternion_xyzw": matrix_to_quaternion_xyzw,
            "quaternion_xyzw_to_matrix": quaternion_xyzw_to_matrix,
        }[name]
    raise AttributeError(name)
