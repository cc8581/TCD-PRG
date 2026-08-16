import pytest
import torch

from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.acronym_grasp_database import match_object_grasp_priors


def rotation_z(degrees):
    angle = torch.deg2rad(torch.tensor(float(degrees)))
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_match_positive_negative_unknown_and_conflict():
    database_t = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.], [.2, .005, 0.]])
    database_r = torch.stack([torch.eye(3)] * 4)
    database_s = torch.tensor([1, 0, 1, 0], dtype=torch.int8)
    proposal_t = torch.tensor([[.001, 0., 0.], [.101, 0., 0.], [.5, 0., 0.], [.2, 0., 0.]])
    result = match_object_grasp_priors(
        proposal_t, torch.stack([torch.eye(3)] * 4), torch.ones(4, dtype=torch.bool),
        database_t, database_r, database_s,
    )
    assert result["status"].tolist() == [1, 0, -1, 0]
    assert result["match_conflict"].tolist() == [False, False, False, True]
    assert result["positive_translation_error_m"][0].item() == pytest.approx(.001, abs=1e-7)
    assert result["negative_translation_error_m"][1].item() == pytest.approx(.001, abs=1e-7)


def test_parallel_jaw_180_degree_symmetry_matches():
    result = match_object_grasp_priors(
        torch.zeros(1, 3), rotation_z(180)[None], torch.ones(1, dtype=torch.bool),
        torch.zeros(1, 3), torch.eye(3)[None], torch.ones(1, dtype=torch.int8),
    )
    assert result["status"].item() == int(CandidateStatus.POSITIVE)
