"""SE(3) operations with named source/destination frames and xyzw quaternions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert the continuous Zhou et al. 6D representation to SO(3)."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must end in 6 values")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    x_axis = torch.nn.functional.normalize(first, dim=-1, eps=eps)
    second = second - (x_axis * second).sum(-1, keepdim=True) * x_axis
    y_axis = torch.nn.functional.normalize(second, dim=-1, eps=eps)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def parallel_jaw_rotation_distance(first: Tensor, second: Tensor) -> Tensor:
    """SO(3) geodesic distance with 180-degree jaw-swap symmetry."""

    relative = first.transpose(-1, -2) @ second
    jaw_swap = torch.diag(torch.tensor(
        [-1.0, -1.0, 1.0], dtype=first.dtype, device=first.device
    ))
    swapped = first.transpose(-1, -2) @ (second @ jaw_swap)

    def angle(matrix: Tensor) -> Tensor:
        trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
        return torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))

    return torch.minimum(angle(relative), angle(swapped))


def normalize_quaternion_xyzw(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize xyzw quaternions and choose a deterministic sign (w >= 0)."""

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., 3:4] < 0, -q, q)


def quaternion_xyzw_to_matrix(q: Tensor) -> Tensor:
    """Convert ``[...,4]`` xyzw quaternion to proper ``[...,3,3]`` rotation."""

    x, y, z, w = normalize_quaternion_xyzw(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(matrix: Tensor) -> Tensor:
    """Numerically stable proper rotation matrix to xyzw quaternion conversion."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(matrix.shape)}")
    if torch.any(torch.det(matrix) < 0.0):
        raise ValueError("Reflection matrix cannot be converted to a rotation quaternion")
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                    1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                ),
                dim=-1,
            ),
            0.0,
        )
    )
    x, y, z, w = q_abs.unbind(-1)
    candidates = torch.stack(
        (
            torch.stack((x * x, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]), -1),
            torch.stack((m[..., 0, 1] + m[..., 1, 0], y * y, m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]), -1),
            torch.stack((m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], z * z, m[..., 1, 0] - m[..., 0, 1]), -1),
            torch.stack((m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], w * w), -1),
        ),
        dim=-2,
    )
    denom = (2.0 * q_abs).clamp_min(0.1).unsqueeze(-1)
    candidates = candidates / denom
    best = q_abs.argmax(dim=-1)
    selected = torch.gather(candidates, -2, best[..., None, None].expand(best.shape + (1, 4))).squeeze(-2)
    return normalize_quaternion_xyzw(selected)


@dataclass(frozen=True, slots=True)
class SE3:
    """A transform named ``T_destination_source``.

    ``rotation`` is ``[...,3,3]`` and ``translation`` is ``[...,3]`` in metres.
    """

    rotation: Tensor
    translation: Tensor
    destination: str
    source: str

    def __post_init__(self) -> None:
        if self.rotation.shape[-2:] != (3, 3):
            raise ValueError("rotation must end in [3,3]")
        if self.translation.shape[-1:] != (3,):
            raise ValueError("translation must end in [3]")
        det = torch.det(self.rotation.detach())
        if torch.any((det - 1.0).abs() > 1e-3):
            raise ValueError("rotation must be proper with determinant +1")

    @classmethod
    def from_pose_xyzw(cls, pose: Tensor, destination: str, source: str) -> "SE3":
        if pose.shape[-1] != 7:
            raise ValueError("Pose must be [x,y,z,qx,qy,qz,qw]")
        return cls(quaternion_xyzw_to_matrix(pose[..., 3:]), pose[..., :3], destination, source)

    def as_matrix(self) -> Tensor:
        eye = torch.eye(4, dtype=self.rotation.dtype, device=self.rotation.device)
        out = eye.expand(self.rotation.shape[:-2] + (4, 4)).clone()
        out[..., :3, :3] = self.rotation
        out[..., :3, 3] = self.translation
        return out

    def as_pose_xyzw(self) -> Tensor:
        return torch.cat((self.translation, matrix_to_quaternion_xyzw(self.rotation)), dim=-1)

    def inverse(self) -> "SE3":
        r = self.rotation.transpose(-1, -2)
        t = -(r @ self.translation.unsqueeze(-1)).squeeze(-1)
        return SE3(r, t, self.source, self.destination)

    def compose(self, other: "SE3") -> "SE3":
        if self.source != other.destination:
            raise ValueError(f"Frame mismatch: {self.source} != {other.destination}")
        r = self.rotation @ other.rotation
        t = (self.rotation @ other.translation.unsqueeze(-1)).squeeze(-1) + self.translation
        return SE3(r, t, self.destination, other.source)

    def transform_points(self, points: Tensor, frame: str) -> Tensor:
        if frame != self.source:
            raise ValueError(f"Points are in {frame}, transform expects {self.source}")
        return (self.rotation @ points.unsqueeze(-1)).squeeze(-1) + self.translation
