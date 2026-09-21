from types import SimpleNamespace

import torch

from real_experiment_app.predictor import TCDPRGPredictor
from tcd_prg.models.push.actions import PushActions
from tcd_prg.models.tcd_prg import TCDPRGModel


def test_grasp_forward_does_not_run_push_rules_or_evaluator(monkeypatch):
    model = TCDPRGModel.__new__(TCDPRGModel)
    torch.nn.Module.__init__(model)
    encoded = SimpleNamespace(
        target_instance_probability=torch.ones(1, 1, 1),
        scene_point_features=torch.ones(1, 1, 1),
        instance=None,
    )
    sensor = {"xyz": torch.zeros(1, 1, 3)}
    task = {"task_category_id": torch.zeros(1, dtype=torch.long),
            "task_region_id": torch.zeros(1, dtype=torch.long)}
    monkeypatch.setattr(model, "_encode_scene", lambda _batch: (encoded, sensor, task))
    monkeypatch.setattr(
        model, "_forward_region", lambda *_: {"region_probability": torch.ones(1, 1, 1)}
    )
    monkeypatch.setattr(model, "_target_identity_gate", lambda *_: torch.ones(1, dtype=torch.bool))
    monkeypatch.setattr(model, "forward_task_grasp_from_condition", lambda *_: {})
    monkeypatch.setattr(model, "_forward_global_grasp", lambda *_: {})
    monkeypatch.setattr(model, "_push_condition", lambda *_: object())
    monkeypatch.setattr(
        model, "forward_push_from_condition",
        lambda *_: (_ for _ in ()).throw(AssertionError("PUSH ran during grasp prediction")),
    )
    output = model.forward({}, forward_mode="full_without_push")
    assert output["push"] is None
    assert output["push_condition"] is not None


def test_rules_are_generated_before_the_selected_objects_push_is_scored():
    events = []
    actions = PushActions(
        torch.zeros(2, dtype=torch.long), torch.tensor([2, 4]),
        torch.zeros(2, 3), torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
        torch.full((2,), 0.15),
    )
    class Model:
        def push(self, _sensor, _condition):
            events.append("rules")
            return actions

        def score_push_actions(self, _sensor, _condition, selected):
            events.append("evaluator")
            assert selected.object.tolist() == [2]
            return {"improvement_logit": torch.tensor([0.8])}

    class Policy:
        preparation_actions = 3

        def generate_candidates(self, encoded):
            assert encoded.output["push"]["actions"].object.tolist() == [2]
            assert self.preparation_actions == 0
            return {"encoded": encoded, "candidates": {
                "valid": torch.tensor([[True]]),
                "point_index": torch.tensor([[-1]]),
            }}

        def _action(self, _tensors, _index):
            return {"action_type": 0, "acted_object": 2,
                    "improvement_probability": 0.8}

    condition = SimpleNamespace(validate=lambda _count: None)
    sensor = {"xyz": torch.zeros(1, 2, 3)}
    value = TCDPRGPredictor.__new__(TCDPRGPredictor)
    value.model = Model()
    value.policy = Policy()
    value._encoded = SimpleNamespace(output={"sensor": sensor, "push_condition": condition})
    value._push_actions = None
    rules = value.generate_push_rules(2)
    assert len(rules) == 1 and events == ["rules"]
    scored = value.score_push_rules()
    assert events == ["rules", "evaluator"]
    assert scored[0]["acted_object"] == 2
    assert value.policy.preparation_actions == 3


def test_push_rules_include_every_requested_obstructor():
    actions = PushActions(
        torch.zeros(3, dtype=torch.long), torch.tensor([2, 4, 7]),
        torch.zeros(3, 3), torch.tensor([[1., 0., 0.]] * 3),
        torch.full((3,), .15),
    )
    value = TCDPRGPredictor.__new__(TCDPRGPredictor)
    value.model = SimpleNamespace(push=lambda *_: actions)
    value._encoded = SimpleNamespace(output={
        "sensor": {"xyz": torch.zeros(1, 3, 3)},
        "push_condition": SimpleNamespace(validate=lambda _: None),
    })
    value._push_actions = None
    rules = value.generate_push_rules((2, 4))
    assert [rule["acted_object"] for rule in rules] == [2, 4]
    assert value._push_actions.object.tolist() == [2, 4]


def test_predictor_passes_adjacent_eligibility_only_when_requested():
    actions = PushActions(
        torch.zeros(1, dtype=torch.long), torch.tensor([7]),
        torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.tensor([.15]),
    )
    called = []
    class Model:
        def push(self, _sensor, _condition, *, adjacent_objects):
            called.append(adjacent_objects)
            return actions
    value = TCDPRGPredictor.__new__(TCDPRGPredictor)
    value.model = Model()
    value._encoded = SimpleNamespace(output={
        "sensor": {"xyz": torch.zeros(1, 3, 3)},
        "push_condition": SimpleNamespace(validate=lambda _: None),
    })
    value._push_actions = None
    rules = value.generate_push_rules((7,), adjacent_objects=(7,))
    assert called == [(7,)]
    assert rules[0]["acted_object"] == 7
