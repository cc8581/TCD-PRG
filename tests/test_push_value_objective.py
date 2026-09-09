import pytest
import torch

from tcd_prg.losses.push_effectiveness import PushImprovementLoss
from tcd_prg.models.staged_checkpoint import stage_training_state
from tcd_prg.push_improvement import improvement_event
from tcd_prg.scripts.train_push_evaluator import push_optimizer_groups


def test_binary_event_ignores_small_changes_and_rejects_conflicting_evidence():
    before = [0, -2, -1, 0, 0, .50, 0]
    assert not improvement_event(before, [0, -2, -1, 0, 0, .505, 0])[0]
    assert improvement_event(before, [0, -2, -1, 0, 0, .52, 0])[0]
    assert not improvement_event(before, [0, -3, -1, 0, 0, .70, 0])[0]
    assert not improvement_event(before, [1, -3, -1, 0, 0, .70, 0])[0]
    assert not improvement_event(before, [0, -2, -1, 0, 0, .52, -.2])[0]
    assert not improvement_event(before, [0, -2, -1, 0, 0, .48, .2])[0]
    assert improvement_event(before, [1, -2, -1, 0, 0, .52, .2])[0]
    assert not improvement_event([1, -2, -1, 0, 0, .50, 0], [0, -2, 0, 0, 0, .60, .2])[0]


def test_single_head_binary_bce_uses_improvement_logit():
    score = torch.zeros(3, requires_grad=True)
    prediction = {"improvement_logit": score}
    losses = PushImprovementLoss()(
        prediction,
        improvement_target=torch.tensor([1., 0., 0.]),
        improvement_valid=torch.ones(3, dtype=torch.bool),
    )
    losses["push_improvement"].backward()
    assert score.grad is not None and score.grad.abs().sum() > 0
    assert set(prediction) == {"improvement_logit"}


def test_batch_without_binary_supervision_is_zero_not_an_exception():
    score = torch.zeros(2, requires_grad=True)
    losses = PushImprovementLoss()(
        {"improvement_logit": score},
        improvement_target=torch.zeros(2),
        improvement_valid=torch.zeros(2, dtype=torch.bool),
    )
    assert losses["push_improvement"].item() == 0.0
    losses["push_improvement"].backward()
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
    batch = scene(); network = model().eval()
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    with torch.no_grad():
        output = network.score_actions(batch, condition, actions)
    assert output["improvement_logit"].shape == (len(actions.batch_index),)
    assert not hasattr(network.push_evaluator, "q_head")
    assert not hasattr(network.push_evaluator, "safety_head")
    assert not hasattr(network.push_evaluator, "auxiliary_delta_head")
