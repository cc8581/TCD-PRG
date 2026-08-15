import torch

from tcd_prg.config import BackboneConfig, GraspNetConfig, ModelConfig
from tcd_prg.geometry.camera import (
    camera_to_world_points,
    graspnet_to_tcd_rotation,
    look_at_rotation_world_camera,
    world_to_camera_points,
)
from tcd_prg.models import TCDPRGModel


def _model_config() -> ModelConfig:
    return ModelConfig(
        feature_dim=32,
        task_dim=32,
        instance_queries=4,
        instance_decoder_layers=1,
        instance_decoder_heads=4,
        task_grasp_scorer_layers=1,
        task_grasp_scorer_heads=4,
        verifier_transformer_layers=1,
        verifier_transformer_heads=4,
        push_direction_feature_dim=32,
        push_direction_transformer_heads=4,
    )


def test_camera_world_se3_round_trip():
    eye = torch.tensor([[0.4, -0.3, 0.8]])
    target = torch.tensor([[0.0, 0.0, 0.2]])
    up = torch.tensor([[0.0, 0.0, 1.0]])
    rotation = look_at_rotation_world_camera(eye, target, up)
    world = torch.randn(1, 17, 3)
    camera = world_to_camera_points(world, rotation, eye)
    restored = camera_to_world_points(camera, rotation, eye)
    assert torch.allclose(restored, world, atol=1e-6)
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3).reshape(1, 3, 3),
        atol=1e-6,
    )


def test_graspnet_to_tcd_axis_mapping():
    rotation = torch.eye(3).reshape(1, 1, 3, 3)
    mapped = graspnet_to_tcd_rotation(rotation)
    assert torch.equal(mapped[..., :, 0], rotation[..., :, 1])
    assert torch.equal(mapped[..., :, 1], rotation[..., :, 2])
    assert torch.equal(mapped[..., :, 2], rotation[..., :, 0])
    assert torch.allclose(torch.det(mapped), torch.ones(1, 1))


def test_chunked_camera_to_ptv3_nearest_index_matches_dense_distance():
    torch.manual_seed(11)
    query = torch.randn(2, 23, 3)
    reference = torch.randn(2, 19, 3)
    query_mask = torch.ones(2, 23, dtype=torch.bool)
    reference_mask = torch.ones(2, 19, dtype=torch.bool)
    query_mask[1, 20:] = False
    reference_mask[0, 17:] = False
    actual, distance, valid = TCDPRGModel._nearest_reference_index(
        query, query_mask, reference, reference_mask, chunk_size=5
    )
    for row in range(2):
        queries = torch.nonzero(query_mask[row], as_tuple=False).flatten()
        references = torch.nonzero(reference_mask[row], as_tuple=False).flatten()
        expected = references[
            torch.cdist(query[row, queries], reference[row, references]).argmin(-1)
        ]
        assert torch.equal(actual[row, queries], expected)
        assert torch.all(valid[row, queries])
        assert torch.all(torch.isfinite(distance[row, queries]))


def test_camera2_strict_crop_never_falls_back_to_scene(fake_graspnet, tiny_batch):
    model = TCDPRGModel(
        _model_config(),
        backbone_config=BackboneConfig(backend="legacy"),
        graspnet_config=GraspNetConfig(
            target_min_crop_points=4,
            target_proposals=4,
            global_proposals=4,
            target_input_points=8,
            scene_input_points=8,
        ),
    ).eval()
    sensor = model._sensor(tiny_batch)
    points = sensor["xyz"].shape[1]
    instance_probability = torch.ones(1, 4, points) / 4
    insufficient = torch.zeros(1, points)
    insufficient[:, :3] = 1.0
    output = model._forward_camera_graspnet(
        sensor,
        target_probability=insufficient,
        instance_probability=instance_probability,
        strict_target_crop=True,
        proposal_count=4,
        input_points=8,
    )
    assert not bool(output["target_grasp_valid"].any())
    assert not bool(output["valid"].any())
    assert int(output["target_crop_mask"].sum()) == 0
