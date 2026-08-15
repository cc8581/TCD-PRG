from types import SimpleNamespace

import torch
from test_graspnet_io_protocol import _model_config

from tcd_prg.config import BackboneConfig, GraspNetConfig
from tcd_prg.models import TCDPRGModel


def test_transfer_distance_gate():
    query = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    query_mask = torch.tensor([[True, True]])
    reference = torch.tensor([[[0.001, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    reference_mask = torch.tensor([[True, True]])
    index, distance, valid = TCDPRGModel._nearest_reference_index(
        query,
        query_mask,
        reference,
        reference_mask,
        max_distance_m=0.01,
    )
    assert index.tolist() == [[0, 1]]
    assert valid.tolist() == [[True, False]]
    assert distance[0, 0] < 0.01


def test_encoder_exposes_selected_instance_mask_not_soft_query_mixture(fake_graspnet, tiny_batch):
    model = TCDPRGModel(
        _model_config(),
        backbone_config=BackboneConfig(backend="legacy"),
        graspnet_config=GraspNetConfig(
            target_min_crop_points=1,
            target_proposals=4,
            global_proposals=4,
            target_input_points=8,
            scene_input_points=8,
        ),
    ).eval()
    encoded, _, _ = model._encode_scene(tiny_batch)
    rows = torch.arange(encoded.target_query_index.shape[0])
    expected = encoded.instance.mask_probability[rows, encoded.target_query_index] * tiny_batch[
        "point_mask"
    ].to(encoded.target_probability.dtype)
    assert torch.allclose(encoded.target_instance_probability, expected)


def test_target_identity_gate_accepts_reid_without_prompt(fake_graspnet):
    model = TCDPRGModel(
        _model_config(),
        backbone_config=BackboneConfig(backend="legacy"),
        graspnet_config=GraspNetConfig(),
    )
    encoded = SimpleNamespace(
        target_prompt_used=torch.tensor([False, True, False]),
        target_reid_used=torch.tensor([True, True, False]),
        target_prompt_support=torch.tensor([0.0, 0.0, 1.0]),
        target_selection_margin=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert model._target_identity_gate(encoded).tolist() == [True, False, False]
