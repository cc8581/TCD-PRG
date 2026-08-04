import sys

import numpy as np
import pytest
import torch

from tcd_prg.geometry.gripper_provider import ExactAG16095GeometryProvider
from tcd_prg.geometry.se3 import SE3, matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix


def test_quaternion_xyzw_round_trip() -> None:
    q = torch.tensor([[0.1, -0.2, 0.3, 0.9]])
    q = q / q.norm(dim=-1, keepdim=True)
    recovered = matrix_to_quaternion_xyzw(quaternion_xyzw_to_matrix(q))
    assert torch.allclose(recovered, q, atol=1e-5)


def test_coordinate_transform_direction_and_grasp_pose() -> None:
    pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    world_from_object = SE3.from_pose_xyzw(pose, "world", "object")
    point = world_from_object.transform_points(torch.tensor([0.1, 0.0, 0.0]), "object")
    assert torch.allclose(point, torch.tensor([1.1, 2.0, 3.0]))
    assert torch.allclose(world_from_object.inverse().transform_points(point, "world"), torch.tensor([0.1, 0, 0.0]))


def test_reflection_rejected() -> None:
    with pytest.raises(ValueError):
        SE3(-torch.eye(3), torch.zeros(3), "world", "grasp")


def test_gripper_widths_use_finite_millimeter_bins(tmp_path) -> None:
    worker = tmp_path / "worker.py"
    urdf = tmp_path / "gripper.urdf"
    worker.write_text("", encoding="utf-8")
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    provider = ExactAG16095GeometryProvider(
        sys.executable, worker, urdf, tmp_path / "cache",
        point_count=8, width_quantization_m=0.001, allow_generate=False,
    )
    quantized = provider.quantize_widths(
        np.asarray([-1.0, 0.00049, 0.00051, 0.0946, 1.0])
    )
    assert np.allclose(quantized, [0.0, 0.0, 0.001, 0.095, 0.095])
    bins = provider.uniform_width_bins()
    assert len(bins) == 96
    assert bins[0] == pytest.approx(0.0)
    assert bins[-1] == pytest.approx(0.095)
