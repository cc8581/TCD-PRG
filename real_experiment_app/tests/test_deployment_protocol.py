import torch

from tcd_prg.models.graspnet.adapter import FrozenGraspNetProposalGenerator


def test_deployment_preserves_trained_graspnet_fixed_length_sparse_sampling():
    mask = torch.ones((1, 855), dtype=torch.bool)
    selected, valid = FrozenGraspNetProposalGenerator._sample_indices(
        mask, None, 20_000
    )
    assert valid.tolist() == [True]
    assert selected.shape == (1, 20_000)
    assert len(torch.unique(selected)) == 855


def test_training_compatibility_can_retain_historical_sparse_sampling():
    mask = torch.tensor([[True, True, False]])
    selected, valid = FrozenGraspNetProposalGenerator._sample_indices(mask, None, 4)
    assert valid.tolist() == [True]
    assert selected.tolist() == [[0, 1, 0, 1]]
