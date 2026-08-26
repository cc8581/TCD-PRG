from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tcd_prg.config import ModelConfig, load_config
from tcd_prg.geometry.stageb_grasp import (
    any_distance_below,
    evaluate_stageb_geometry,
    world_to_grasp_numpy,
)
from tcd_prg.losses.task_grasp_binary import stageb_split_metrics
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.task_grasp import TaskGraspEvaluator, world_to_grasp
from tcd_prg.scripts.build_stageb_binary import balance_binary_records
from tcd_prg.trainers.trainer import aggregate_stageb_validation_payloads

ASSET = Path("assets/robots/FR5_AG-160-95/ag16095_open_tcp_128.npz")
DENSE_ASSET = Path("assets/robots/FR5_AG-160-95/ag16095_open_tcp_4096.npz")


def evaluator_inputs(batch: int = 2, candidates: int = 3, points: int = 40, dim: int = 16):
    torch.manual_seed(7)
    xyz = torch.randn(batch, points, 3) * 0.025
    translation = torch.randn(batch, candidates, 3) * 0.01
    rotation = torch.eye(3).reshape(1, 1, 3, 3).expand(batch, candidates, -1, -1).clone()
    proposals = {
        "translation_world": translation,
        "rotation_matrix": rotation,
        "width_m": torch.full((batch, candidates), 0.05),
        "valid": torch.ones(batch, candidates, dtype=torch.bool),
    }
    return (
        proposals,
        torch.randn(batch, points, dim),
        xyz,
        torch.ones(batch, points, dtype=torch.bool),
        torch.rand(batch, points),
        torch.rand(batch, points),
        torch.randn(batch, dim),
        torch.randn(batch, dim),
    )


def test_formal_stageb_config_uses_full_data_and_validation() -> None:
    config = load_config("configs/stage/grasp.yaml")
    assert config.training.max_train_groups is None
    assert config.training.max_validation_groups is None
    assert config.training.validation_interval == 1000


def test_split_metrics_select_threshold_from_complete_split() -> None:
    metrics = stageb_split_metrics(np.asarray([0.1, 0.4, 0.6, 0.9]), np.asarray([0, 1, 0, 1], bool))
    assert metrics["task_grasp_validation_f1"] == pytest.approx(0.8)
    assert metrics["task_grasp_validation_threshold"] == pytest.approx(0.4)
    assert np.isfinite(metrics["task_grasp_validation_auroc"])
    assert np.isfinite(metrics["task_grasp_validation_auprc"])


def test_world_grasp_roundtrip() -> None:
    points = torch.randn(2, 3, 11, 3)
    translation = torch.randn(2, 3, 3)
    rotation = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 3, -1, -1)
    local = world_to_grasp(points[:, 0], translation, rotation)
    restored = torch.einsum("bkni,bkji->bknj", local, rotation) + translation[:, :, None]
    assert torch.allclose(restored, points[:, :1].expand(-1, 3, -1, -1), atol=1e-6)


def test_fixed_ag_geometry_uses_real_parts_and_tcp_dimensions() -> None:
    payload = np.load(ASSET)
    assert payload["points_tcp"].shape == (128, 3)
    assert np.bincount(payload["part_id"]).tolist() == [0, 64, 32, 32]
    assert payload["points_tcp"][:, 0].ptp() > 0.15
    assert -0.20 < payload["points_tcp"][:, 2].min() < -0.18
    assert float(payload["width_m"]) == pytest.approx(0.095)


def test_candidate_aligned_transform_moves_with_pose() -> None:
    points = np.asarray([[0.02, 0.0, 0.0]], np.float32)
    identity = np.eye(3, dtype=np.float32)
    assert np.allclose(world_to_grasp_numpy(points, np.zeros(3), identity), points)
    assert np.allclose(
        world_to_grasp_numpy(points, np.asarray([0.01, 0, 0]), identity), [[0.01, 0, 0]]
    )


def test_chunked_dense_distance_matches_full_matrix() -> None:
    rng = np.random.default_rng(4)
    scene = rng.normal(size=(513, 3)).astype(np.float32)
    gripper = rng.normal(size=(4096, 3)).astype(np.float32)
    for threshold in (0.001, 0.05, 0.5):
        full = bool((((scene[:, None] - gripper[None]) ** 2).sum(-1) < threshold**2).any())
        assert any_distance_below(scene, gripper, threshold, chunk_size=37) == full


def test_ddp_stageb_metrics_are_computed_after_raw_union() -> None:
    summaries = [
        {"stageb_scores": np.asarray([0.9, 0.8]), "stageb_targets": np.asarray([1, 0])},
        {"stageb_scores": np.asarray([0.7, 0.1]), "stageb_targets": np.asarray([1, 0])},
    ]
    merged = aggregate_stageb_validation_payloads(summaries)
    direct = stageb_split_metrics(
        np.concatenate([item["stageb_scores"] for item in summaries]),
        np.concatenate([item["stageb_targets"] for item in summaries]),
    )
    assert merged == direct


def test_geometry_protocol_binary_region_gate() -> None:
    x = np.asarray([-0.03, -0.02, -0.01, -0.007, 0.007, 0.01, 0.02, 0.03], np.float32)
    xyz = np.stack((x, np.zeros_like(x), np.full_like(x, -0.02)), -1)
    mask = np.ones(len(x), bool)
    geometry = np.load(DENSE_ASSET)
    positive = evaluate_stageb_geometry(
        xyz,
        mask,
        mask,
        mask,
        mask,
        np.zeros(3),
        np.eye(3),
        0.095,
        geometry["points_tcp"],
        geometry["part_id"],
    )
    negative = evaluate_stageb_geometry(
        xyz,
        mask,
        mask,
        np.zeros_like(mask),
        mask,
        np.zeros(3),
        np.eye(3),
        0.095,
        geometry["points_tcp"],
        geometry["part_id"],
    )
    assert positive.task_valid
    assert not negative.task_valid
    assert "contact_outside_functional_region" in negative.reasons

    unknown = mask.copy()
    unknown[x < 0] = False
    with pytest.raises(ValueError, match="unknown functional region at contact"):
        evaluate_stageb_geometry(
            xyz,
            mask,
            mask,
            mask,
            unknown,
            np.zeros(3),
            np.eye(3),
            0.095,
            geometry["points_tcp"],
            geometry["part_id"],
        )


def test_binary_record_balance_is_exact_and_deterministic(tmp_path) -> None:
    records = []
    for index, label in enumerate((np.asarray([1, 0, 0, 0], bool), np.asarray([0, 0, 0], bool))):
        path = tmp_path / f"{index}.npz"
        np.savez_compressed(
            path,
            translation_world=np.zeros((len(label), 3), np.float32),
            rotation_matrix=np.tile(np.eye(3, dtype=np.float32), (len(label), 1, 1)),
            width_m=np.full(len(label), 0.05, np.float32),
            task_valid=label,
        )
        records.append({"path": path.name})
    balanced = balance_binary_records(tmp_path, records, seed=3)
    label = np.concatenate(
        [np.load(tmp_path / record["path"])["task_valid"] for record in balanced]
    )
    assert int(label.sum()) == 1
    assert int((~label).sum()) == 2


def test_vectorized_candidates_equal_independent_forward_and_logits_finite() -> None:
    evaluator = TaskGraspEvaluator(16, ASSET).eval()
    inputs = evaluator_inputs(batch=1, candidates=3)
    with torch.no_grad():
        vector = evaluator(*inputs)["task_valid_logit"]
        individual = []
        for index in range(3):
            proposal = {key: value[:, index : index + 1] for key, value in inputs[0].items()}
            individual.append(evaluator(proposal, *inputs[1:])["task_valid_logit"])
    assert torch.isfinite(vector).all()
    assert torch.allclose(vector, torch.cat(individual, 1), atol=1e-6)


def test_evaluator_is_invariant_to_scene_point_permutation() -> None:
    evaluator = TaskGraspEvaluator(16, ASSET).eval()
    inputs = evaluator_inputs(batch=1, candidates=2, points=40)
    permutation = torch.randperm(40)
    permuted = list(inputs)
    for index in (1, 2, 3, 4, 5):
        permuted[index] = inputs[index][:, permutation]
    with torch.no_grad():
        first = evaluator(*inputs)["task_valid_logit"]
        second = evaluator(*permuted)["task_valid_logit"]
    assert torch.allclose(first, second, atol=1e-5)


def test_local_cloud_keeps_sparse_points_padded() -> None:
    evaluator = TaskGraspEvaluator(16, ASSET).eval()
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    translation = torch.zeros(1, 1, 3)
    rotation = torch.eye(3).reshape(1, 1, 3, 3)
    _, _, mask = evaluator._local_cloud(
        xyz, torch.ones(1, 3, dtype=torch.bool), translation, rotation
    )
    assert int(mask[0, 0, : evaluator.scene_points].sum()) == 2


def test_stageb_backward_updates_only_evaluator(fake_graspnet, tiny_batch) -> None:
    model = TCDPRGModel(ModelConfig(feature_dim=32, task_dim=16))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("task_grasp."))
    batch = dict(tiny_batch)
    translation = torch.cat((batch["xyz"][:, :2].clone(), torch.full((1, 1, 3), float("nan"))), 1)
    batch["grasp_candidates"] = {
        "translation_world": translation,
        "rotation_matrix": torch.cat(
            (
                torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, -1, -1),
                torch.full((1, 1, 3, 3), float("nan")),
            ),
            1,
        ),
        "valid": torch.tensor([[True, True, False]]),
        "width_m": torch.tensor([[0.04, 0.05, float("nan")]]),
    }
    output = model.forward_grasp(batch)["task_grasp"]
    torch.nn.functional.binary_cross_entropy_with_logits(
        output["task_valid_logit"][:, :2], torch.tensor([[0.0, 1.0]])
    ).backward()
    assert any(parameter.grad is not None for parameter in model.task_grasp.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.task_grasp.parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.graspnet.parameters())
