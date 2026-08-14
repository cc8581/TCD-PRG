import torch

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


def test_task_grasp_residual_starts_from_graspnet_score():
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
    assert torch.allclose(output["quality_logit"], proposals["quality_logit"], atol=1e-6)


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
        "negative_translation_world": torch.tensor([[[.1,0,0.],[0.,0.,0.]]]),
        "negative_rotation_matrix": eye.reshape(1,1,3,3).repeat(b,m,1,1),
        "negative_width_m": torch.full((b,m), .04),
        "negative_valid": torch.tensor([[True,False]]),
        "label_set_complete": torch.tensor([False]),
    }
    out = loss_fn(pred, labels)
    assert out["loss"].requires_grad
    assert float(out["task_proposal_recall_at_16"]) == 1.0
    assert float(out["task_grasp_top1_positive"]) == 1.0


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
    assert float(output["task_proposal_recall_at_16"]) == 1.0
