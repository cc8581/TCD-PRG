from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcd_prg.baselines import GAPGPolicyWrapper, OneShotSequencePolicy
from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.paths import PROJECT_ROOT
from tcd_prg.planners import ClosedLoopPlanner, DenseCandidateGenerator
from tcd_prg.models import PushCondition


def _predicted_encoded(instance_probability, object_mask, dim=8, target=0):
    batch, queries, _ = instance_probability.shape
    weights = torch.zeros(batch, queries)
    weights[:, target] = 1.0
    return SimpleNamespace(
        instance=SimpleNamespace(mask_probability=instance_probability),
        object_tokens=torch.zeros(batch, queries, dim),
        object_mask=object_mask,
        target_query_weights=weights,
        task_token=torch.zeros(batch, dim),
    )


def test_gapg_wrapper_reports_every_missing_external_dependency(tmp_path) -> None:
    wrapper = GAPGPolicyWrapper(
        tmp_path, "grasp.pt", "push.pt", "graspnet.tar", python=tmp_path / "python.exe"
    )
    with pytest.raises(FileNotFoundError, match="GAPG grasp checkpoint"):
        wrapper.paths.validate()


def test_minimal_gapg_runtime_contains_every_worker_import() -> None:
    root = PROJECT_ROOT / "third_party" / "GAPG"
    expected = {
        "utils.py",
        "pytorch3d_compat.py",
        "env/constants.py",
        "models/grasp_networks.py",
        "models/push_networks.py",
        "models/pointnet2_encoder.py",
        "models/pointnet2_utils.py",
    }
    assert {path.relative_to(root).as_posix() for path in root.rglob("*.py")} == expected


def test_dense_generator_handles_a_scene_with_no_candidate() -> None:
    config = ModelConfig(
        feature_dim=8,
        task_dim=4,
        task_grasp_candidates=2,
        pick_remove_candidates=2,
        push_candidates=2,
    )
    generator = DenseCandidateGenerator(config)
    batch = {
        "xyz": torch.zeros(1, 4, 3),
        "instance_id": torch.zeros(1, 4, dtype=torch.long),
        "point_mask": torch.ones(1, 4, dtype=torch.bool),
        "object_mask": torch.zeros(1, 1, dtype=torch.bool),
        "object_active": torch.zeros(1, 1, dtype=torch.bool),
        "target_object": torch.tensor([0]),
        "target_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    point_head = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": torch.eye(3).expand(1, 2, 3, 3),
        "width_m": torch.ones(1, 2) * 0.05,
        "quality_logit": torch.zeros(1, 2),
        "task_valid_probability": torch.full((1, 2), 0.5),
        "attention_point_index": torch.zeros(1, 2, dtype=torch.long),
    }
    global_head = {
        "translation_world": torch.zeros(1, 2, 3),
        "rotation_matrix": torch.eye(3).expand(1, 2, 3, 3),
        "width_m": torch.ones(1, 2) * 0.05,
        "quality_logit": torch.zeros(1, 2),
        "attention_point_index": torch.zeros(1, 2, dtype=torch.long),
        "object_logits": torch.zeros(1, 2, 1),
    }
    push = {
        "object_logits": torch.zeros(1, 1),
        "contact_logits": torch.zeros(1, 4),
        "direction_logits": torch.zeros(1, 4, 16),
        "direction_residual": torch.zeros(1, 4, 16, 2),
        "utility_delta": torch.zeros(1, 4, 16),
        "direction_point_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    encoded = _predicted_encoded(torch.ones(1, 1, 4), batch["object_mask"])

    class Model:
        @staticmethod
        def candidate_encoder(*args):
            return torch.zeros(1, 1, 8)

    result = generator.generate(
        Model(),
        batch,
        {
            "push_condition": PushCondition(
                torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool),
                torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
            ),
            "encoded": encoded,
            "task_grasp": point_head,
            "global_grasp": global_head,
            "push": push,
            "graph": None,
        },
    )
    assert result["type"].shape == (1, 1)
    assert not result["valid"].any()
    assert "push_approach_mode" not in result








def test_push_action_nms_uses_router_order_for_near_duplicate_actions() -> None:
    generator = DenseCandidateGenerator(
        ModelConfig(push_nms_contact_m=0.02, push_nms_direction_deg=10.0)
    )
    candidates = {
        "valid": torch.ones(1, 3, dtype=torch.bool),
        "type": torch.full((1, 3), int(ActionType.PUSH)),
        "object": torch.tensor([[0, 0, 1]]),
        "contact_world": torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "direction_world": torch.tensor([[[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [1.0, 0.0, 0.0]]]),
    }
    keep = generator.apply_push_nms(candidates, torch.tensor([[0.2, 0.9, 0.1]]))
    assert keep.tolist() == [[False, True, True]]
