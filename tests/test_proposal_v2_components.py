import torch
import pytest

from tcd_prg.models.instance_segmentation import InstanceMaskDecoder
from tcd_prg.models.task_grasp import TaskGraspScorer
from tcd_prg.losses.task_grasp_score import TaskGraspScoringLoss


def test_instance_decoder_shapes_and_auxiliary_outputs():
    decoder = InstanceMaskDecoder(64, 8, 5, layers=3, heads=8)
    feature = torch.randn(2, 40, 64)
    xyz = torch.randn(2, 40, 3)
    mask = torch.ones(2, 40, dtype=torch.bool)
    out = decoder(feature, xyz, mask)
    assert out.mask_logits.shape == (2, 8, 40)
    assert out.object_tokens.shape == (2, 8, 64)
    assert len(out.aux_outputs) == 2


def test_task_grasp_score_is_independent_from_graspnet_score():
    scorer = TaskGraspScorer(64, layers=2, heads=8)
    b, k, n = 2, 6, 32
    proposals = {
        "translation_world": torch.randn(b, k, 3) * 0.02,
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3).repeat(b, k, 1, 1),
        "width_m": torch.full((b, k), 0.04),
        "depth_m": torch.full((b, k), 0.02),
        "quality_logit": torch.randn(b, k),
        "valid": torch.ones(b, k, dtype=torch.bool),
        "attention_point_index": torch.zeros(b, k, dtype=torch.long),
        "object_logits": torch.zeros(b, k, 2),
    }
    output = scorer(
        proposals, torch.randn(b, n, 64), torch.randn(b, n, 3) * 0.03,
        torch.ones(b, n, dtype=torch.bool), torch.rand(b, n), torch.rand(b, n),
        torch.randn(b, 64), torch.randn(b, 64),
    )
    assert output["quality_logit"].shape == proposals["quality_logit"].shape
    assert torch.isfinite(output["quality_logit"]).all()
    assert not torch.allclose(output["quality_logit"], proposals["quality_logit"])


def test_task_grasp_score_loss_has_proposal_metrics():
    loss_fn = TaskGraspScoringLoss(translation_m=0.03, rotation_deg=20, width_m=0.02)
    b, k, m = 1, 4, 2
    eye = torch.eye(3)
    pred = {
        "translation_world": torch.tensor([[[0.,0.,0.],[.1,0,0],[.2,0,0],[.3,0,0]]]),
        "rotation_matrix": eye.reshape(1,1,3,3).repeat(b,k,1,1),
        "width_m": torch.full((b,k), .04),
        "quality_logit": torch.tensor([[2.,1.,0.,-1.]], requires_grad=True),
        "graspnet_quality_logit": torch.tensor([[2.,1.,0.,-1.]]),
        "valid": torch.ones(b,k,dtype=torch.bool),
    }
    labels = {
        "translation_world": torch.tensor([[[0.,0.,0.],[0.,0.,0.]]]),
        "rotation_matrix": eye.reshape(1,1,3,3).repeat(b,m,1,1),
        "width_m": torch.full((b,m), .04),
        "target_valid": torch.tensor([[True,False]]),
        "wrong_region_translation_world": torch.tensor([[[.1,0,0.],[0.,0.,0.]]]),
        "wrong_region_rotation_matrix": eye.reshape(1,1,3,3).repeat(b,m,1,1),
        "wrong_region_width_m": torch.full((b,m), .04),
        "wrong_region_valid": torch.tensor([[True,False]]),
        "label_set_complete": torch.tensor([False]),
    }
    out = loss_fn(pred, labels)
    assert out["loss"].requires_grad
    assert float(out["task_proposal_recall_at_16"]) == 1.0
    assert float(out["task_grasp_effective_fraction"]) == 1.0
    assert 0.0 <= float(out["task_grasp_unknown_fraction"]) <= 1.0
    assert float(out["task_grasp_top1_positive"]) == 1.0
    assert float(out["task_grasp_top1_known_positive"]) == 1.0
    assert float(out["task_grasp_top1_unknown"]) == 0.0
    assert float(out["task_grasp_top1_known_negative"]) == 0.0


def test_task_grasp_matching_respects_parallel_jaw_symmetry():
    loss_fn = TaskGraspScoringLoss()
    eye = torch.eye(3)
    jaw_swap = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    prediction = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": jaw_swap.reshape(1, 1, 3, 3),
        "width_m": torch.full((1, 1), 0.04),
        "quality_logit": torch.zeros(1, 1, requires_grad=True),
        "valid": torch.ones(1, 1, dtype=torch.bool),
    }
    labels = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": eye.reshape(1, 1, 3, 3),
        "width_m": torch.full((1, 1), 0.04),
        "target_valid": torch.ones(1, 1, dtype=torch.bool),
        "label_set_complete": torch.zeros(1, dtype=torch.bool),
    }

    output = loss_fn(prediction, labels)

    assert float(output["task_grasp_positive_proposals"]) == 1.0


def test_task_grasp_ranking_uses_top_known_candidates_not_set_mass():
    loss_fn = TaskGraspScoringLoss()
    prediction = {
        "quality_logit": torch.tensor([[1.0, 0.0, 0.9, 0.9]], requires_grad=True),
        "translation_world": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        ),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3),
        "valid": torch.ones(1, 4, dtype=torch.bool),
    }
    labels = {
        "translation_world": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "target_valid": torch.ones(1, 1, dtype=torch.bool),
        "wrong_region_translation_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "wrong_region_rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3),
        "wrong_region_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    output = loss_fn(prediction, labels)
    expected = torch.nn.functional.softplus(torch.tensor(0.9 - 1.0))
    assert output["task_grasp_score_ranking"] == pytest.approx(float(expected))
    assert float(output["task_proposal_recall_at_16"]) == 1.0


def test_ag_width_is_not_part_of_proposal_identity():
    loss_fn = TaskGraspScoringLoss(translation_m=0.02, rotation_deg=20)
    eye = torch.eye(3).reshape(1, 1, 3, 3)
    prediction = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": eye,
        "width_m": torch.tensor([[0.095]]),
        "quality_logit": torch.zeros(1, 1, requires_grad=True),
        "valid": torch.ones(1, 1, dtype=torch.bool),
    }
    labels = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": eye,
        "width_m": torch.tensor([[0.005]]),
        "target_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    output = loss_fn(prediction, labels)
    assert float(output["task_grasp_positive_proposals"]) == 1.0


def test_unknown_only_task_row_has_zero_scorer_gradient():
    logits = torch.tensor([[0.7, -0.3]], requires_grad=True)
    eye = torch.eye(3).reshape(1, 1, 3, 3)
    prediction = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": eye.repeat(1, 2, 1, 1),
        "quality_logit": logits,
        "valid": torch.ones(1, 2, dtype=torch.bool),
    }
    labels = {
        "translation_world": torch.zeros(1, 1, 3),
        "rotation_matrix": eye,
        "target_valid": torch.zeros(1, 1, dtype=torch.bool),
        "wrong_region_translation_world": torch.zeros(1, 1, 3),
        "wrong_region_rotation_matrix": eye,
        "wrong_region_valid": torch.zeros(1, 1, dtype=torch.bool),
    }
    output = TaskGraspScoringLoss()(prediction, labels)
    output["loss"].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
