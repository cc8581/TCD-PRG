from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcd_prg.baselines import GAPGPolicyWrapper, OneShotSequencePolicy
from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.planners import ClosedLoopPlanner, DenseCandidateGenerator
from tcd_prg.planners.tcd_policy import (
    apply_verified_candidate_count_gate,
)


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
        "point_mask": torch.ones(1, 4, dtype=torch.bool),
        "object_mask": torch.zeros(1, 1, dtype=torch.bool),
        "object_active": torch.zeros(1, 1, dtype=torch.bool),
        "target_object": torch.tensor([0]), "target_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    point_head = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": torch.eye(3).expand(1, 2, 3, 3),
        "width_m": torch.ones(1, 2) * 0.05, "quality_logit": torch.zeros(1, 2),
        "attention_point_index": torch.zeros(1, 2, dtype=torch.long),
    }
    global_head = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": torch.eye(3).expand(1, 2, 3, 3),
        "width_m": torch.ones(1, 2) * 0.05, "quality_logit": torch.zeros(1, 2),
        "attention_point_index": torch.zeros(1, 2, dtype=torch.long),
        "object_logits": torch.zeros(1, 2, 1),
    }
    push = {
        "object_logits": torch.zeros(1, 1), "contact_logits": torch.zeros(1, 4),
        "direction_logits": torch.zeros(1, 4, 16), "direction_residual": torch.zeros(1, 4, 2),
        "utility_delta": torch.zeros(1, 4, 16),
    }
    encoded = SimpleNamespace(object_tokens=torch.zeros(1, 1, 8), task_token=torch.zeros(1, 8))

    class Model:
        @staticmethod
        def candidate_encoder(*args):
            return torch.zeros(1, 1, 8)

    result = generator.generate(Model(), batch, {
        "encoded": encoded, "task_grasp": point_head, "global_grasp": global_head,
        "push": push, "graph": None,
    })
    assert result["type"].shape == (1, 1)
    assert not result["valid"].any()
    assert "push_approach_mode" not in result


def test_offline_evaluator_accepts_overall_only_verifier(monkeypatch) -> None:
    sample = SimpleNamespace(
        observation=SimpleNamespace(
            scene_id=0, state_id=0, task_index=0, target_object=0,
            object_category_id=torch.tensor([0]), task_region_id=0,
        ),
        state_labels=SimpleNamespace(sequence_depth=0, target_visible_ratio=1.0),
    )
    batch = {
        "xyz": torch.zeros(1, 1, 3), "samples": [sample],
        "candidate_mask": torch.zeros(1, 1, dtype=torch.bool),
        "evaluation_status": torch.zeros(1, 1, dtype=torch.long),
        "policy_success_mask": torch.zeros(1, 1, dtype=torch.bool),
        "action_type": torch.zeros(1, 1, dtype=torch.long),
        "acted_object": torch.zeros(1, 1, dtype=torch.long),
    }
    output = {
        "graph": None,
        "verifier": {"overall_logit": torch.zeros(1, 1)},
        "push": {"object_logits": torch.zeros(1, 1)},
    }
    monkeypatch.setattr(
        "tcd_prg.evaluators.offline.build_verifier_labels",
        lambda _: {
            "overall_valid": torch.ones(1, 1, dtype=torch.bool),
            "overall_target": torch.ones(1, 1),
        },
    )
    monkeypatch.setattr(
        "tcd_prg.evaluators.offline.build_push_supervision",
        lambda *_: ({}, {
            "direction_valid": torch.zeros(1, 1, dtype=torch.bool),
            "utility_valid": torch.zeros(1, 1, dtype=torch.bool),
        }),
    )

    OfflineModelEvaluator(ModelConfig(), bootstrap_samples=1).update(batch, output)


def test_dense_generator_uses_graph_frontier_with_bounded_fallback() -> None:
    config = ModelConfig(
        feature_dim=8, task_dim=4, task_grasp_candidates=1,
        pick_remove_candidates=8, push_candidates=8,
        graph_candidate_fallback_objects=1,
    )
    generator = DenseCandidateGenerator(config)
    instance = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    batch = {
        "xyz": torch.randn(1, 8, 3), "instance_id": instance,
        "point_mask": torch.ones(1, 8, dtype=torch.bool),
        "object_mask": torch.ones(1, 4, dtype=torch.bool),
        "object_active": torch.ones(1, 4, dtype=torch.bool),
        "target_object": torch.tensor([0]), "target_mask": instance == 0,
    }
    point_head = {
        "translation_world": batch["xyz"][:, :1] + torch.tensor([0.01, -0.02, 0.03]),
        "rotation_matrix": torch.eye(3).expand(1, 1, 3, 3),
        "width_m": torch.full((1, 1), 0.05), "quality_logit": torch.zeros(1, 1),
        "attention_point_index": torch.zeros(1, 1, dtype=torch.long),
    }
    global_head = {
        "translation_world": batch["xyz"][:, ::2],
        "rotation_matrix": torch.eye(3).expand(1, 4, 3, 3),
        "width_m": torch.full((1, 4), 0.05),
        "quality_logit": torch.arange(4).float()[None],
        "attention_point_index": torch.tensor([[0, 2, 4, 6]]),
        "object_logits": torch.eye(4)[None] * 20.0,
    }
    push = {
        "object_logits": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        "contact_logits": torch.arange(8).float()[None],
        "direction_logits": torch.zeros(1, 8, 16),
        "direction_residual": torch.zeros(1, 8, 2),
        "utility_delta": torch.zeros(1, 8, 16),
    }
    graph = SimpleNamespace(
        derived_actionable_mask=torch.tensor([[False, True, False, False]]),
    )
    encoded = SimpleNamespace(object_tokens=torch.zeros(1, 4, 8), task_token=torch.zeros(1, 8))

    class Model:
        @staticmethod
        def candidate_encoder(object_tokens, candidate_type, *args):
            return torch.zeros(candidate_type.shape + (8,))

    result = generator.generate(Model(), batch, {
        "encoded": encoded, "task_grasp": point_head, "global_grasp": global_head,
        "push": push, "graph": graph,
    })
    task_row = torch.nonzero(
        result["valid"][0] & (result["type"][0] == int(ActionType.TASK_GRASP)),
        as_tuple=False,
    ).flatten().item()
    source_point = int(result["point_index"][0, task_row])
    assert torch.allclose(
        result["pose_world"][0, task_row, :3],
        batch["xyz"][0, source_point] + torch.tensor([0.01, -0.02, 0.03]),
    )
    preparation = result["valid"][0] & (result["type"][0] != int(ActionType.TASK_GRASP))
    objects = set(result["object"][0, preparation].tolist())
    assert 0 in objects  # explicit target self-push recovery rule
    assert 1 in objects  # graph-derived actionable object
    assert 3 in objects  # exactly one highest-scoring recovery object
    assert 2 not in objects

    config.allow_target_push_recovery = False
    without_target_recovery = DenseCandidateGenerator(config).generate(Model(), batch, {
        "encoded": encoded, "task_grasp": point_head, "global_grasp": global_head,
        "push": push, "graph": graph,
    })
    push_rows = without_target_recovery["valid"][0] & (
        without_target_recovery["type"][0] == int(ActionType.PUSH)
    )
    assert 0 not in set(without_target_recovery["object"][0, push_rows].tolist())


def test_adaptive_task_grasp_gate_requires_unique_certified_candidate_counts() -> None:
    kinds = torch.tensor([[2, 2, 2, 1]])
    objects = torch.tensor([[0, 0, 0, 1]])
    poses = torch.tensor([[[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    ], [
        0.004, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    ], [
        0.030, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    ], [
        float("nan"), float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"), float("nan")
    ]]])
    width = torch.tensor([[0.05, 0.052, 0.05, float("nan")]])
    score = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
    valid = torch.ones_like(kinds, dtype=torch.bool)
    masked, count, keep = apply_verified_candidate_count_gate(
        kinds, objects, poses, width, score, valid, torch.tensor([2]),
        translation_threshold_m=0.010, rotation_threshold_deg=12.0,
        width_threshold_m=0.005, approach_threshold_deg=12.0,
    )
    assert count.item() == 2
    assert keep.tolist() == [[True, False, True, False]]
    assert masked.tolist() == [[True, False, True, True]]

    masked, count, _ = apply_verified_candidate_count_gate(
        kinds, objects, poses, width, score, valid, torch.tensor([3]),
        translation_threshold_m=0.010, rotation_threshold_deg=12.0,
        width_threshold_m=0.005, approach_threshold_deg=12.0,
    )
    assert count.item() == 2
    assert masked.tolist() == [[False, False, False, True]]


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
