import torch

from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models.staged_checkpoint import stage_training_state


def test_q_objective_trains_value_ranking_safety_and_auxiliary():
    prediction = {
        "q_value": torch.tensor([[.6, .7], [.4, .5]], requires_grad=True),
        "safety_logit": torch.tensor([1., -1.], requires_grad=True),
        "potential_delta": torch.zeros(2, 5, requires_grad=True),
    }
    losses = PushEffectivenessLoss(delta_scales=(1., 1., 1., 1., 1.))(
        prediction,
        q_target=torch.tensor([[.2, .3], [.8, .9]]),
        q_valid=torch.ones(2, 2, dtype=torch.bool),
        safety_target=torch.tensor([True, False]),
        safety_valid=torch.ones(2, dtype=torch.bool),
        auxiliary_target=torch.ones(2, 5),
        auxiliary_valid=torch.ones(2, dtype=torch.bool),
        group_index=torch.zeros(2, dtype=torch.long),
    )
    losses["push_effectiveness"].backward()
    assert losses["push_rank"] > 0
    assert all(value.grad is not None for value in prediction.values())


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


def test_new_evaluator_returns_monotonic_q_safety_and_effects():
    from test_independent_push import model, scene
    from tcd_prg.models import push_condition_from_gt
    from tcd_prg.trainers.push_evaluator import logged_push_actions

    batch = scene()
    network = model().eval()
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    with torch.no_grad():
        output = network.score_actions(batch, condition, actions)
    assert output["q_value"].shape == (len(actions.batch_index), 5)
    assert bool((output["q_value"][:, 1:] >= output["q_value"][:, :-1]).all())
    assert output["safety_probability"].shape == (len(actions.batch_index),)
    assert output["potential_delta"].shape == (len(actions.batch_index), 5)
