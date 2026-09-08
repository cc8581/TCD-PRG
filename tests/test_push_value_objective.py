import pytest
import torch

from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models.staged_checkpoint import stage_training_state
from tcd_prg.scripts.train_push_evaluator import push_optimizer_groups


def test_single_head_value_and_ranking_share_the_same_prediction():
    score = torch.zeros(3, requires_grad=True)
    prediction = {"push_value": score}
    key = torch.tensor([
        [0., -1., -1., 0., 0., .5, .0],
        [0., -2., -1., 0., 0., .5, .0],
        [0., -3., -1., 0., 0., .5, .0],
    ])
    losses = PushEffectivenessLoss()(
        prediction,
        value_target=torch.tensor([1., 0., -1.]),
        value_valid=torch.ones(3, dtype=torch.bool),
        rank_key=key,
        rank_valid=torch.ones(3, dtype=torch.bool),
        group_index=torch.zeros(3, dtype=torch.long),
    )
    losses["push_effectiveness"].backward()
    assert losses["push_rank"] > 0
    assert score.grad is not None and score.grad.abs().sum() > 0
    assert set(prediction) == {"push_value"}


def test_batch_without_core_value_supervision_is_zero_not_an_exception():
    score = torch.zeros(2, requires_grad=True)
    losses = PushEffectivenessLoss()(
        {"push_value": score},
        value_target=torch.full((2,), float("nan")),
        value_valid=torch.zeros(2, dtype=torch.bool),
        rank_key=torch.full((2, 7), float("nan")),
        rank_valid=torch.zeros(2, dtype=torch.bool),
        group_index=torch.zeros(2, dtype=torch.long),
    )
    assert losses["push_effectiveness"].item() == 0.0
    losses["push_effectiveness"].backward()
    assert score.grad is not None


def test_push_optimizer_uses_configured_lower_backbone_learning_rate():
    class Evaluator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

    class Model:
        push_evaluator = Evaluator()

    class Config:
        class optimizer:
            learning_rate = 1e-4
            backbone_learning_rate = 2e-5

    groups = push_optimizer_groups(Model(), Config())
    assert [group["name"] for group in groups] == ["push_heads", "pointnet2_backbone"]
    assert [group["lr"] for group in groups] == [pytest.approx(1e-4), pytest.approx(2e-5)]


def test_ab_checkpoint_migration_replaces_only_push_tensors():
    class Stub:
        def state_dict(self):
            return {
                "encoder.weight": torch.tensor([9.]),
                "task_grasp.weight": torch.tensor([8.]),
                "push_evaluator.new": torch.tensor([7.]),
            }

    old = {
        "encoder.weight": torch.tensor([1.]),
        "task_grasp.weight": torch.tensor([2.]),
        "push_evaluator.old": torch.tensor([3.]),
    }
    migrated = stage_training_state(Stub(), old, "perception")
    assert migrated["encoder.weight"].item() == 1
    assert migrated["task_grasp.weight"].item() == 2
    assert "push_evaluator.old" not in migrated
    assert migrated["push_evaluator.new"].item() == 7


def test_evaluator_has_one_learned_output_head():
    from test_independent_push import model, scene
    from tcd_prg.models import push_condition_from_gt
    from tcd_prg.trainers.push_evaluator import logged_push_actions

    batch = scene()
    network = model().eval()
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    with torch.no_grad():
        output = network.score_actions(batch, condition, actions)
    assert output["push_value"].shape == (len(actions.batch_index),)
    assert not hasattr(network.push_evaluator, "q_head")
    assert not hasattr(network.push_evaluator, "safety_head")
    assert not hasattr(network.push_evaluator, "auxiliary_delta_head")
