from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tcd_prg.models.graspnet import FrozenGraspNetProposalGenerator


class _DenseTrainingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1)
        self.is_training = False
        self.grasp_generator = SimpleNamespace(is_training=False)

    def forward(self, batch):
        assert self.is_training
        return {"dense_training_loss": self.projection(batch["point_clouds"]).square().mean()}


def _dense_batch() -> dict:
    return {
        "point_clouds": torch.randn(1, 8, 3),
        "objectness_label": torch.ones(1, 8, dtype=torch.long),
        "object_poses_list": [[]],
        "grasp_points_list": [[]],
        "grasp_offsets_list": [[]],
        "grasp_labels_list": [[]],
        "grasp_tolerance_list": [[]],
    }


def test_opt_in_finetuning_uses_separate_parameters_and_dense_loss():
    generator = FrozenGraspNetProposalGenerator(
        source_root="unused", checkpoint="unused", freeze=False
    )
    network = _DenseTrainingNetwork()
    object.__setattr__(generator, "_network", network)
    object.__setattr__(generator, "_network_device", torch.device("cpu"))
    generator.official_get_loss = lambda output: (output["dense_training_loss"], output)

    generator.prepare_finetuning("cpu")
    parameters = generator.finetune_parameters()
    assert parameters
    assert not generator.state_dict()

    loss, _ = generator.official_training_loss(_dense_batch())
    loss.backward()
    assert all(parameter.grad is not None for parameter in parameters)
    assert not network.is_training
    assert not network.grasp_generator.is_training


def test_sparse_task_pose_batch_is_rejected_for_graspnet_finetuning():
    generator = FrozenGraspNetProposalGenerator(
        source_root="unused", checkpoint="unused", freeze=False
    )
    with pytest.raises(KeyError, match="dense upstream labels"):
        generator.official_training_loss(
            {
                "point_clouds": torch.randn(1, 8, 3),
                "task_grasp_pose_world": torch.randn(1, 8, 7),
            }
        )


def test_tristate_unknown_labels_are_rejected_by_unmasked_official_loss():
    generator = FrozenGraspNetProposalGenerator(
        source_root="unused", checkpoint="unused", freeze=False
    )
    batch = _dense_batch()
    batch["grasp_known_mask_list"] = [[]]
    with pytest.raises(RuntimeError, match="no UNKNOWN state"):
        generator.official_training_loss(batch)
