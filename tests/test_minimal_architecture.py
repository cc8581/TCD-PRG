from __future__ import annotations

from types import SimpleNamespace
import inspect

import numpy as np
import torch

from tcd_prg.baselines.one_shot import OneShotSequencePolicy
from tcd_prg.config import load_config
from tcd_prg.execution.pybullet_certifier import ExternalFR5AG16095Certifier
from tcd_prg.planners.candidate_generator import DenseCandidateGenerator
from tcd_prg.planners.tcd_policy import TCDPRGPolicy


def test_minimal_stage_configs_load() -> None:
    for path, stage in (
        ("configs/stage/perception.yaml", "perception"),
        ("configs/stage/grasp.yaml", "grasp"),
        ("configs/stage/push_evaluator.yaml", "push_evaluator"),
    ):
        config = load_config(path)
        assert config.ablation.use_task_region_condition
        assert config.training.stage == stage



def test_policy_runtime_has_no_grasp_count_gate_interface() -> None:
    for method in (TCDPRGPolicy._sensor_task_batch, TCDPRGPolicy.encode_fused_scene):
        assert "required_grasp_count" not in inspect.signature(method).parameters

def test_pick_remove_target_local_mask_rejects_far_object() -> None:
    # target query 0 occupies x=[0.00,0.02], object 1 is nearby, object 2 is far.
    xyz = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.02, 0.01, 0.0],
            [0.05, 0.00, 0.0],
            [0.06, 0.01, 0.0],
            [0.30, 0.00, 0.0],
            [0.31, 0.01, 0.0],
        ],
        dtype=torch.float32,
    )
    probability = torch.zeros(3, len(xyz))
    probability[0, :2] = 1.0
    probability[1, 2:4] = 1.0
    probability[2, 4:] = 1.0
    mask = DenseCandidateGenerator._target_local_object_mask(
        xyz,
        torch.ones(len(xyz), dtype=torch.bool),
        probability,
        torch.ones(3, dtype=torch.bool),
        target_object=0,
        margin_m=0.05,
    )
    assert mask.tolist() == [True, True, False]


def test_certifier_resolves_query_to_physical_object_from_geometry() -> None:
    observation = SimpleNamespace(
        xyz=np.asarray([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]], np.float32),
        instance_id=np.asarray([3, 7], np.int64),
        point_valid=np.asarray([True, True]),
        object_pose=np.zeros((8, 7), np.float32),
    )
    certifier = SimpleNamespace(observation=observation)
    physical = ExternalFR5AG16095Certifier._physical_acted_object(
        certifier,
        {
            "acted_object": 0,  # predicted query id: deliberately unrelated
            "association_point_world": np.asarray([0.19, 0.0, 0.0], np.float32),
        },
    )
    assert physical == 7


def test_one_shot_uses_effectiveness_for_push_ranking() -> None:
    class CandidatePolicy:
        def __init__(self) -> None:
            self.encodes = 0

        def reset(self) -> None:
            pass

        def encode_observation(self, observation):
            self.encodes += 1
            return observation

        def generate_candidates(self, encoded):
            del encoded
            return {
                "candidates": {
                    "valid": torch.tensor([[True, True, True]]),
                    "type": torch.tensor([[0, 1, 2]]),
                    "object": torch.tensor([[1, 2, 0]]),
                    "proposal_score": torch.tensor([[0.8, 0.6, 0.9]]),
                    "effective_probability": torch.tensor([[float("nan"), 0.9, 0.1]]),
                }
            }

        @staticmethod
        def _action(tensors, index):
            return {
                "action_type": int(tensors["type"][0, index]),
                "acted_object": int(tensors["object"][0, index]),
                "candidate_index": int(index),
            }

        def predict_grasps(self, encoded):
            return []

        def predict_task_grasps(self, encoded):
            return []

        def predict_global_grasps(self, encoded):
            return []

    wrapped = CandidatePolicy()
    policy = OneShotSequencePolicy(wrapped)
    encoded = policy.encode_observation(object())
    sequence = policy.generate_candidates(encoded)
    assert wrapped.encodes == 1
    assert policy.encode_observation(object()) is None
    # Open-loop baseline keeps preparation actions and then one terminal grasp.
    assert [item["action_type"] for item in sequence] == [0, 1, 2]
