from types import SimpleNamespace

import pytest
import torch

from real_experiment_app.predictor import TCDPRGPredictor


def predictor():
    value = TCDPRGPredictor.__new__(TCDPRGPredictor)
    value.device = torch.device("cpu")
    value.policy = SimpleNamespace(
        model=SimpleNamespace(
            task_grasp=torch.nn.Identity(),
            graspnet=torch.nn.Identity(),
            push=torch.nn.Identity(),
            push_evaluator=torch.nn.Identity(),
        )
    )
    return value


def test_profiled_encode_reports_each_required_model_stage():
    value = predictor()

    def encode():
        tensor = torch.ones(1)
        for name in ("task_grasp", "graspnet", "push", "push_evaluator"):
            tensor = getattr(value.policy.model, name)(tensor)
        return tensor

    result, timings = value._profiled_encode(encode)
    assert result.item() == 1
    assert set(timings) == {
        "target_grasp_prediction_s",
        "obstructor_grasp_prediction_s",
        "push_rule_generation_s",
        "push_evaluator_scoring_s",
    }


def test_profiled_encode_removes_hooks_after_failure():
    value = predictor()
    with pytest.raises(RuntimeError, match="failed"):
        value._profiled_encode(lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    for module in vars(value.policy.model).values():
        assert not module._forward_pre_hooks and not module._forward_hooks
