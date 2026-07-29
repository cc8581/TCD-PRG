import pytest
import torch

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

