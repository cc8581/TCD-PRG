"""Regression tests for independent logged-action training and rule-only inference."""
import torch
import pytest
from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.models import StandalonePushModel, push_condition_from_gt
from tcd_prg.models.push import PushActions
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.trainers.push_evaluator import logged_push_actions, push_effectiveness_batch_loss
from tcd_prg.models.staged_checkpoint import load_push_evaluator, PUSH_EVALUATOR_PROTOCOL_VERSION, PUSH_ARCHITECTURE
from tcd_prg.planners.push_decoder import decode_push_candidates


def scene():
    # Target, two upper blockers, and one higher but horizontally disjoint object.
    square = torch.tensor([[x, y, z] for x in [-.03, 0., .03]
                           for y in [-.03, 0., .03] for z in [0., .01]])
    xyz = torch.cat([square, square+torch.tensor([.02, 0., .07]),
                     square+torch.tensor([-.02, 0., .12]), square+torch.tensor([1., 0., .2])])[None]
    ids = torch.arange(4).repeat_interleave(len(square))[None]
    return dict(xyz=xyz, rgb=torch.zeros_like(xyz), point_mask=torch.ones_like(ids, dtype=torch.bool),
                instance_id=ids, object_mask=torch.ones(1, 4, dtype=torch.bool),
                target_mask=ids == 0, region_valid=ids == 0, region_target=ids == 0,
                task_category_id=torch.tensor([0]), task_region_id=torch.tensor([0]),
                geometry_feature=torch.randn(1, xyz.shape[1], 16),
                candidate_mask=torch.tensor([[True, True, True]]),
                action_type=torch.full((1, 3), int(ActionType.PUSH)),
                evaluation_status=torch.tensor([[1, 0, int(CandidateStatus.UNKNOWN_UNTESTED)]]),
                action_improves_state=torch.tensor([[True, False, True]]),
                acted_object=torch.tensor([[1, 1, 1]]),
                action_parameters={
                    "push_contact_world": torch.tensor([[[.02, -.03, .075], [.02, -.03, .075], [0., 0., .1]]]),
                    "push_direction_world": torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]])})


def model():
    return StandalonePushModel(ModelConfig(feature_dim=16, instance_queries=4))


def test_training_never_calls_rule_generator_or_rejects_far_contact(monkeypatch):
    batch = scene()
    batch["action_parameters"]["push_contact_world"][0, 0] = torch.tensor([.8, .8, .8])
    m = model().train()
    monkeypatch.setattr(m.push, "forward", lambda *a: pytest.fail("training called rule generation"))
    condition = push_condition_from_gt(batch, 4)
    actions, valid = logged_push_actions(batch, condition)
    assert len(actions.object) == 2
    assert torch.equal(actions.contact_world[0], batch["action_parameters"]["push_contact_world"][0, 0])
    loss, details = push_effectiveness_batch_loss(m, batch, instance_queries=4, loss_function=PushEffectivenessLoss())
    loss.backward()
    assert torch.isfinite(loss)
    assert details["effective_logit"].shape == (2,)
    assert any(p.grad is not None for p in m.push_evaluator.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.push_evaluator.backbone.parameters())
    assert not list(m.push.parameters())


def test_rules_keep_multiple_above_objects_and_do_not_read_gt_actions():
    batch = scene()
    m = model().eval()
    condition = push_condition_from_gt(batch, 4)
    actions = m.push(m._sensor(batch), condition)
    assert set(actions.object.tolist()) == {1, 2}
    actions.validate(1, 4)
    assert (actions.push_distance == .15).all()
    assert (actions.direction_world[:, 2] == 0).all()
    batch["push_condition"] = condition
    out = m(batch)
    assert torch.equal(out["push"]["actions"].contact_world, actions.contact_world)
    direct = m.score_actions(batch, condition, actions)
    assert torch.allclose(direct, out["push"]["effective_logit"])
    pre, final = decode_push_candidates(out["sensor"], condition, out["push"], m.push.config)
    assert len(pre[0]["object"]) <= 32
    assert torch.all(pre[0]["effective_probability"][:-1] >= pre[0]["effective_probability"][1:])


def test_gt_and_rule_action_contract_is_identical():
    batch = scene()
    m = model().eval()
    condition = push_condition_from_gt(batch, 4)
    rules = m.push(m._sensor(batch), condition)
    a = rules.select(torch.tensor([0]))
    batch["candidate_mask"] = torch.tensor([[True]])
    batch["action_type"] = torch.tensor([[int(ActionType.PUSH)]])
    batch["evaluation_status"] = torch.tensor([[1]])
    batch["acted_object"] = a.object[None]
    batch["action_parameters"] = {"push_contact_world": a.contact_world[None],
                                 "push_direction_world": a.direction_world[None]}
    logged, _ = logged_push_actions(batch, condition)
    for field in a.__dataclass_fields__:
        assert torch.allclose(getattr(logged, field), getattr(a, field))
    assert torch.allclose(m.score_actions(batch, condition, logged), m.score_actions(batch, condition, a))


def test_evaluator_can_overfit_opposite_outcomes_for_same_contact():
    torch.manual_seed(17)
    batch, m = scene(), model().train()
    optimizer = torch.optim.Adam(m.push_evaluator.parameters(), lr=.01)
    initial = None
    for _ in range(60):
        optimizer.zero_grad()
        loss, details = push_effectiveness_batch_loss(m, batch, instance_queries=4, loss_function=PushEffectivenessLoss())
        if initial is None:
            initial = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * .1
    assert details["effective_logit"][0] > 2 and details["effective_logit"][1] < -2


def test_checkpoint_is_independent_and_rejects_old_weights(tmp_path):
    m = model()
    path = tmp_path / 'evaluator.pt'
    payload = dict(training_stage='push_evaluator', push_evaluator_protocol_version=PUSH_EVALUATOR_PROTOCOL_VERSION,
                   push_architecture=PUSH_ARCHITECTURE, model=m.push_evaluator.state_dict())
    torch.save(payload,path)
    other=model(); load_push_evaluator(other,path)
    assert other.push_evaluator_ready
    payload['push_evaluator_protocol_version']=3
    torch.save(payload,path)
    with pytest.raises(RuntimeError,match='old evaluator weights'):
        load_push_evaluator(other,path)


def test_empty_actions_and_invalid_direction():
    batch, m = scene(), model()
    condition = push_condition_from_gt(batch, 4)
    empty = PushActions.empty(batch["xyz"])
    assert m.score_actions(batch, condition, empty).shape == (0,)
    invalid = PushActions(torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 3),
                          torch.zeros(1, 3), torch.tensor([.15]))
    with pytest.raises(ValueError, match="unit"):
        m.score_actions(batch, condition, invalid)

def test_ab_checkpoint_active_weights_and_optimizer_moments_survive_push_removal():
    from tcd_prg.models.staged_checkpoint import stage_training_state, stage_training_optimizer_state
    class Old(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)
            self.push = torch.nn.Linear(3, 2).requires_grad_(False)
            self.push_evaluator = torch.nn.Linear(2, 1)
    class New(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)
            self.push_evaluator = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    old, new = Old(), New()
    opt = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad], lr=.01)
    x = torch.ones(2, 3)
    old.encoder(x).square().sum().backward()
    opt.step()
    payload = {"model": old.state_dict(), "optimizer": opt.state_dict(), "training_stage": "perception"}
    new.load_state_dict(stage_training_state(new, payload["model"], "perception"), strict=True)
    resumed = torch.optim.AdamW(new.parameters(), lr=.01)
    resumed.load_state_dict(stage_training_optimizer_state(new, resumed, payload))
    for m, optimizer in [(old, opt), (new, resumed)]:
        optimizer.zero_grad()
        m.encoder(x).square().sum().backward()
        optimizer.step()
    assert torch.equal(old.encoder.weight, new.encoder.weight)
    assert torch.equal(old.encoder.bias, new.encoder.bias)
    malformed = dict(payload["model"])
    malformed.pop("encoder.weight")
    with pytest.raises(RuntimeError):
        new.load_state_dict(stage_training_state(new, malformed, "grasp"), strict=True)
