from types import SimpleNamespace

import torch

from tcd_prg.config import TCDPRGConfig
from tcd_prg.constants import ActionType
from tcd_prg.models import push_condition_from_gt
from tcd_prg.planners.closed_loop import ClosedLoopPlanner
from tcd_prg.planners.push_decoder import decode_push_candidates
from tcd_prg.planners.tcd_policy import EncodedPolicyState, TCDPRGPolicy
from tcd_prg.runtime import _apply_training_augmentation


def test_decoder_uses_same_single_push_value_for_every_nonzero_budget():
    from test_independent_push import model, scene
    from tcd_prg.trainers.push_evaluator import logged_push_actions

    batch = scene()
    network = model().eval()
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    push = {"actions": actions, "push_value": torch.tensor([.2, .8])}
    sensor = network._sensor(batch)
    first, _ = decode_push_candidates(
        sensor, condition, push, network.push.config, remaining_preparation_actions=1
    )
    fifth, _ = decode_push_candidates(
        sensor, condition, push, network.push.config, remaining_preparation_actions=5
    )
    exhausted, _ = decode_push_candidates(
        sensor, condition, push, network.push.config, remaining_preparation_actions=0
    )
    torch.testing.assert_close(first[0]["proposal_score"], fifth[0]["proposal_score"])
    torch.testing.assert_close(first[0]["proposal_score"], torch.tensor([.8, .2]))
    assert not len(exhausted[0]["object"])


def test_policy_passes_remaining_preparation_budget_to_generator():
    captured = []

    class Generator:
        def generate(self, model, batch, output, *, remaining_preparation_actions):
            del model, batch, output
            captured.append(remaining_preparation_actions)
            return {
                "type": torch.tensor([[int(ActionType.PUSH)]]),
                "valid": torch.ones(1, 1, dtype=torch.bool),
            }

    policy = object.__new__(TCDPRGPolicy)
    policy.model = object()
    policy.device = torch.device("cpu")
    policy.generator = Generator()
    policy.preparation_actions = 2
    encoded = EncodedPolicyState(
        None,
        {},
        {},
        {"task_grasp": {"task_valid_logit": torch.zeros(1, 1)}},
    )
    policy.generate_candidates(encoded)
    assert captured == [3]


class _BudgetPolicy:
    def __init__(self, task_on_last: bool):
        self.task_on_last = task_on_last
        self.step = 0

    def reset(self):
        self.step = 0

    def encode_observation(self, observation):
        return observation

    def generate_candidates(self, encoded):
        del encoded
        kinds = [int(ActionType.PUSH)]
        if self.step == 5 and self.task_on_last:
            kinds.append(int(ActionType.TASK_GRASP))
        return {"candidates": {
            "valid": torch.ones(1, len(kinds), dtype=torch.bool),
            "type": torch.tensor([kinds]),
        }}

    def select_action(self, candidates):
        tensors = candidates["candidates"]
        ids = torch.nonzero(tensors["valid"][0], as_tuple=False).flatten()
        if not len(ids):
            return None
        index = int(ids[0])
        return {"action_type": int(tensors["type"][0, index]), "candidate_index": index}

    def update_after_action(self, action, observation):
        del action, observation
        self.step += 1


class _Executor:
    def __init__(self):
        self.executed = []

    def certify(self, action):
        return True, "ok"

    def execute(self, action):
        self.executed.append(action)
        return True


def test_closed_loop_allows_terminal_grasp_but_never_sixth_preparation():
    observations = SimpleNamespace(observe=lambda: object())
    executor = _Executor()
    result = ClosedLoopPlanner(_BudgetPolicy(True), observations, executor).run()
    assert result.success and result.preparation_actions == 5
    assert [a["action_type"] for a in executor.executed] == [int(ActionType.PUSH)] * 5 + [int(ActionType.TASK_GRASP)]

    executor = _Executor()
    result = ClosedLoopPlanner(_BudgetPolicy(False), observations, executor).run()
    assert not result.success and result.failure_reason == "horizon_exhausted"
    assert len(executor.executed) == 5


def test_stage_c_augmentation_disables_extrinsic_jitter():
    from test_rgb_augmentation_and_pretrain import _disable_all_augmentation, _geometry_batch

    config = TCDPRGConfig()
    _disable_all_augmentation(config.augmentation)
    config.augmentation.debug.save_first_batches = 0
    config.augmentation.extrinsic_jitter.probability = 1.0
    config.augmentation.extrinsic_jitter.translation_std_m = (.01, .01)
    config.augmentation.extrinsic_jitter.rotation_degrees = (5., 5.)
    batch = _geometry_batch()
    batch["action_parameters"] = {
        "push_contact_world": torch.tensor([[[.1, .2, .3]]]),
        "push_direction_world": torch.tensor([[[1., 0., 0.]]]),
    }
    xyz = batch["xyz"].clone()
    contact = batch["action_parameters"]["push_contact_world"].clone()
    direction = batch["action_parameters"]["push_direction_world"].clone()
    _apply_training_augmentation(config, batch, disable_extrinsic_jitter=True)
    assert torch.equal(batch["xyz"], xyz)
    assert torch.equal(batch["action_parameters"]["push_contact_world"], contact)
    assert torch.equal(batch["action_parameters"]["push_direction_world"], direction)
