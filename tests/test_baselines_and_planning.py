from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tcd_prg.baselines import GAPGPolicyWrapper, OneShotSequencePolicy
from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType
from tcd_prg.planners import ClosedLoopPlanner, DenseCandidateGenerator


def test_gapg_wrapper_reports_every_missing_external_dependency(tmp_path) -> None:
    wrapper = GAPGPolicyWrapper(
        tmp_path, "grasp.pt", "push.pt", "graspnet.tar", python=tmp_path / "python.exe"
    )
    with pytest.raises(FileNotFoundError, match="GAPG grasp checkpoint"):
        wrapper.paths.validate()


def test_dense_generator_handles_a_scene_with_no_candidate() -> None:
    config = ModelConfig(feature_dim=8, task_dim=4, task_grasp_candidates=2,
                         pick_remove_candidates=2, push_candidates=2)
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.zeros(1, 4, 3), "instance_id": torch.zeros(1, 4, dtype=torch.long),
        "object_mask": torch.zeros(1, 1, dtype=torch.bool),
        "object_active": torch.zeros(1, 1, dtype=torch.bool),
        "target_object": torch.tensor([0]), "target_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    point_head = {
        "contact_logits": torch.zeros(1, 4), "proposal_confidence_logit": torch.zeros(1, 4),
        "task_compatibility_logit": torch.zeros(1, 4), "rotation_logits": torch.zeros(1, 4, 12),
        "approach_direction": torch.ones(1, 4, 3), "width_m": torch.ones(1, 4) * 0.05,
    }
    push = {
        "object_logits": torch.zeros(1, 1), "contact_logits": torch.zeros(1, 4),
        "direction_logits": torch.zeros(1, 4, 16), "direction_residual": torch.zeros(1, 4, 2),
        "approach_logits": torch.zeros(1, 4, 2), "outcome_logits": torch.zeros(1, 4, 7),
        "potential_delta": torch.zeros(1, 4, 5), "risk_logits": torch.zeros(1, 4, 3),
    }
    encoded = SimpleNamespace(object_tokens=torch.zeros(1, 1, 8), task_token=torch.zeros(1, 8))

    class Model:
        @staticmethod
        def candidate_encoder(*args):
            return torch.zeros(1, 1, 8)

    result = generator.generate(Model(), batch, {
        "encoded": encoded, "task_grasp": point_head, "generic_grasp": point_head,
        "push": push, "graph": None,
    })
    assert result["type"].shape == (1, 1)
    assert not result["valid"].any()


def test_one_shot_baseline_never_reencodes_after_initial_plan() -> None:
    class Wrapped:
        def __init__(self): self.encodes = 0
        def reset(self): pass
        def encode_observation(self, observation): self.encodes += 1; return object()
        def generate_candidates(self, encoded):
            router = SimpleNamespace(candidate_logits=torch.tensor([[2.0, 1.0]]))
            return {"candidates": {"valid": torch.ones(1, 2, dtype=torch.bool)}, "router": router}
        @staticmethod
        def _action(tensors, index):
            return {"action_type": int(ActionType.PUSH), "acted_object": index,
                    "candidate_index": index}
        def predict_grasps(self, encoded): return []

    wrapped = Wrapped()
    policy = OneShotSequencePolicy(wrapped)
    encoded = policy.encode_observation(object())
    policy.generate_candidates(encoded)
    assert policy.encode_observation(object()) is None
    assert wrapped.encodes == 1


def test_closed_loop_rejects_unsafe_top_candidate_then_reranks() -> None:
    class Router:
        candidate_logits = torch.tensor([[2.0, 1.0]])

    class Model:
        @staticmethod
        def route_cached(device_batch, output, tensors): return Router()

    class Policy:
        model = Model()
        def reset(self): pass
        def encode_observation(self, observation):
            return SimpleNamespace(device_batch={}, output={})
        def generate_candidates(self, encoded):
            return {"encoded": encoded, "router": Router(), "candidates": {
                "valid": torch.tensor([[True, True]])}}
        def select_action(self, group):
            valid = group["candidates"]["valid"][0]
            index = int(torch.nonzero(valid)[0]) if valid.any() else -1
            return None if index < 0 else {"candidate_index": index,
                "action_type": int(ActionType.TASK_GRASP), "acted_object": 0}
        def update_after_action(self, action, observation): pass

    class Source:
        @staticmethod
        def observe(): return object()

    class Executor:
        def certify(self, action): return (action["candidate_index"] == 1, "collision")
        @staticmethod
        def execute(action): return True

    result = ClosedLoopPlanner(Policy(), Source(), Executor()).run()
    assert result.success
    assert result.actions[0]["candidate_index"] == 1
