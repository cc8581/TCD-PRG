from __future__ import annotations

import pytest
import torch

from tcd_prg.config import AblationConfig, ModelConfig
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.grasp_verifier import GripperSceneTaskVerifier
from tcd_prg.models.policy import MaskedHierarchicalCandidateRouter


def _config() -> ModelConfig:
    return ModelConfig(
        feature_dim=32,
        task_dim=16,
        num_categories=20,
        num_task_regions=20,
        activation_checkpointing=False,
    )


def test_point_gripper_joint_input_shapes() -> None:
    verifier = GripperSceneTaskVerifier(32, 32)
    output = verifier(
        torch.randn(1, 2, 12, 3),
        torch.randn(1, 2, 8, 3),
        torch.randn(1, 2, 12, 32),
        torch.ones(1, 2, 12, dtype=torch.bool),
        torch.rand(1, 2, 12),
        torch.randn(1, 32),
    )
    assert all(value.shape == (1, 2) for value in output.values())


def test_router_never_selects_invalid_candidate() -> None:
    router = MaskedHierarchicalCandidateRouter(32, layers=1)
    valid = torch.tensor([[False, True, False]])
    output = router(
        torch.randn(1, 32), torch.randn(1, 32), torch.randn(1, 2, 32), torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 3, 32), torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 0]]), valid, torch.tensor([5])
    )
    selected = router.select(output, torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 0]]))
    assert selected.item() == 1


@pytest.mark.parametrize(
    "ablation",
    [
        AblationConfig(use_task_region_condition=False),
        AblationConfig(use_dependency_graph=False),
        AblationConfig(use_indirect_dependency_reasoning=False),
        AblationConfig(use_gripper_scene_verifier=False),
        AblationConfig(use_push_potential=False, use_push_risk=False),
        AblationConfig(router_type="fixed_priority"),
        AblationConfig(router_type="flat_candidate_classifier"),
    ],
)
def test_every_ablation_can_forward(tiny_batch, ablation) -> None:
    model = TCDPRGModel(_config(), ablation)
    result = model(tiny_batch)
    assert result["region"]["region_logits"].shape == (1, 24)


def test_fixed_seed_reproducibility(tiny_batch) -> None:
    torch.manual_seed(3)
    first = TCDPRGModel(_config()).eval()
    torch.manual_seed(3)
    second = TCDPRGModel(_config()).eval()
    with torch.no_grad():
        one = first(tiny_batch)["region"]["region_logits"]
        two = second(tiny_batch)["region"]["region_logits"]
    assert torch.equal(one, two)

