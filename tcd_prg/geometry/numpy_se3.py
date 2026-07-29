"""Small NumPy SE(3) kernels used by data-loader worker processes.

These routines deliberately avoid importing SciPy in workers that also import
PyTorch.  On native Windows, loading two OpenMP runtimes from SciPy and PyTorch
can abort the interpreter instead of raising a Python exception.
"""

from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert one ``[x,y,z,w]`` quaternion to a proper rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion [4], got {q.shape}")
    norm = float(np.sqrt(np.sum(q * q)))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion must be finite and non-zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw_numpy(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to a deterministic xyzw quaternion."""

    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected rotation [3,3], got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("Rotation contains non-finite values")
    determinant = float(
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ValueError(f"Rotation determinant must be +1, got {determinant}")
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    else:
        diagonal = (float(m[0, 0]), float(m[1, 1]), float(m[2, 2]))
        index = int(max(range(3), key=diagonal.__getitem__))
        if index == 0:
            scale = np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            x, y, z, w = 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[2, 1] - m[1, 2]) / scale
        elif index == 1:
            scale = np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            x, y, z, w = (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale, (m[0, 2] - m[2, 0]) / scale
        else:
            scale = np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            x, y, z, w = (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale, (m[1, 0] - m[0, 1]) / scale
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= np.sqrt(np.sum(q * q))
    if q[3] < 0:
        q = -q
    return q


def compose_pose_with_transform(
    pose_world_object_xyzw: np.ndarray,
    transform_object_child: np.ndarray,
) -> np.ndarray:
    """Return ``T_world_child`` as ``[xyz,xyzw]`` from a pose and 4x4 transform."""

    pose = np.asarray(pose_world_object_xyzw, dtype=np.float64)
    child = np.asarray(transform_object_child, dtype=np.float64)
    if pose.shape != (7,) or child.shape != (4, 4):
        raise ValueError(f"Expected pose [7] and transform [4,4], got {pose.shape}, {child.shape}")
    rotation_world_object = quaternion_xyzw_to_matrix_numpy(pose[3:])
    rotation_world_child = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            rotation_world_child[row, column] = sum(
                float(rotation_world_object[row, inner] * child[inner, column])
                for inner in range(3)
            )
    translation_world_child = np.asarray(
        [
            sum(float(rotation_world_object[row, inner] * child[inner, 3]) for inner in range(3))
            + float(pose[row])
            for row in range(3)
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        (translation_world_child, matrix_to_quaternion_xyzw_numpy(rotation_world_child))
    ).astype(np.float32)
