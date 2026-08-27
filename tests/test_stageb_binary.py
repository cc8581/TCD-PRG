from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig, load_config
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.datasets.stageb_manifest import SCHEMA_VERSION, build_provenance
from tcd_prg.datasets.torch_dataset import StageBBinaryDataset
from tcd_prg.geometry.stageb_grasp import (
    any_distance_below,
    evaluate_stageb_geometry,
    world_to_grasp_numpy,
)
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.losses.task_grasp_binary import stageb_split_metrics
from tcd_prg.models import StageBCondition, TCDPRGModel, stageb_condition_from_gt
from tcd_prg.models.task_grasp import TaskGraspEvaluator, world_to_grasp
from tcd_prg.planners.candidate_generator import DenseCandidateGenerator
from tcd_prg.scripts.build_stageb_binary import (
    assert_no_train_validation_leakage,
    balance_binary_records,
    finalize_split_records,
)
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
        xyz,
        torch.rand(batch, points, 3),
        torch.ones(batch, points, dtype=torch.bool),
        torch.rand(batch, points),
        torch.rand(batch, points),
        torch.randint(0, 4, (batch,)),
        torch.randint(0, 4, (batch,)),
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


def test_split_metrics_calibrates_threshold_on_deployment_selected_candidates() -> None:
    metrics = stageb_split_metrics(
        np.asarray([0.9, 0.8, 0.7]), np.asarray([1, 0, 0], bool),
        np.asarray([0.3, 0.6]), np.asarray([1, 0], bool),
    )
    assert metrics["task_grasp_validation_threshold"] == pytest.approx(0.3)
    assert metrics["task_grasp_raw_f1"] == pytest.approx(1.0)


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


def test_validation_records_keep_natural_label_distribution(tmp_path) -> None:
    label = np.asarray([1, 0, 0, 0], bool)
    path = tmp_path / "record.npz"
    np.savez_compressed(path, task_valid=label)
    records = [
        {
            "split": "val",
            "scene_id": 1,
            "state_id": 2,
            "task_index": 3,
            "group_index": 4,
            "task_region_id": 5,
            "object_category_id": 6,
            "path": path.name,
            "candidate_count": len(label),
            "positive_count": int(label.sum()),
        }
    ]
    kept = finalize_split_records(tmp_path, "val", records, seed=3)
    assert kept == records
    assert np.array_equal(np.load(path)["task_valid"], label)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "provenance": {},
                "records": kept,
            }
        ),
        encoding="utf-8",
    )
    dataset = StageBBinaryDataset(object(), tmp_path, split="val")
    assert len(dataset) == 1
    assert dataset.records[0]["candidate_count"] == 4
    assert dataset.records[0]["positive_count"] == 1


def test_stageb_compatibility_ignores_paths_but_tracks_sampling_semantics() -> None:
    config = load_config("configs/stage/grasp.yaml")
    moved = copy.deepcopy(config)
    moved.dataset.root = "Z:/another-mount/dataset"
    moved.dataset.stageb_binary_root = "Z:/another-mount/stageb"
    moved.graspnet.source_root = "Z:/another-mount/graspnet"
    assert build_provenance(config)["compatibility"] == build_provenance(moved)["compatibility"]
    changed = copy.deepcopy(config)
    changed.backbone.grid_size_m *= 2
    assert build_provenance(config)["compatibility"] != build_provenance(changed)["compatibility"]


def test_stageb_compatibility_tracks_partition_and_camera_source_budget() -> None:
    config = load_config("configs/stage/grasp.yaml")
    changed_partition = copy.deepcopy(config)
    changed_partition.training.seed += 1
    assert build_provenance(config)["compatibility"] != build_provenance(
        changed_partition
    )["compatibility"]
    changed_budget = copy.deepcopy(config)
    changed_budget.graspnet.scene_input_points += 1
    assert build_provenance(config)["compatibility"] != build_provenance(changed_budget)["compatibility"]


def test_stageb_manifest_rejects_cross_split_task_leakage() -> None:
    records = [
        {"split": "train", "scene_id": 1, "state_id": 2, "task_index": 3},
        {"split": "val", "scene_id": 1, "state_id": 2, "task_index": 3},
    ]
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_train_validation_leakage(records)


def test_deployment_calibration_selection_applies_task_nms_and_cap() -> None:
    config = ModelConfig(task_grasp_candidates=2)
    selector = DenseCandidateGenerator(config)
    translation = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0]])
    rotation = torch.eye(3).expand(3, -1, -1).clone()
    width = torch.full((3,), 0.05)
    score = torch.tensor([0.9, 0.8, 0.7])
    selected = selector.select_task_grasp_indices(
        translation, rotation, width, score, torch.ones(3, dtype=torch.bool)
    )
    assert selected.tolist() == [0, 2]


def test_invalid_task_grasp_cannot_suppress_valid_or_consume_cap() -> None:
    config = ModelConfig(task_grasp_candidates=2)
    selector = DenseCandidateGenerator(config)
    translation = torch.tensor(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )
    rotation = torch.eye(3).expand(3, -1, -1).clone()
    width = torch.full((3,), 0.05)
    score = torch.tensor([0.9, 0.99, 0.7])
    valid = torch.tensor([True, False, True])
    selected = selector.select_task_grasp_indices(
        translation, rotation, width, score, valid
    )
    assert selected.tolist() == [0, 2]


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


def test_local_cloud_keeps_sparse_voxels_padded() -> None:
    evaluator = TaskGraspEvaluator(16, ASSET).eval()
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    translation = torch.zeros(1, 1, 3)
    rotation = torch.eye(3).reshape(1, 1, 3, 3)
    _, _, mask = evaluator._local_cloud(
        xyz, torch.zeros_like(xyz), torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3), torch.ones(1, 3), translation, rotation
    )
    assert int(mask[0, 0, : evaluator.scene_points].sum()) == 1


def test_voxel_preselection_preserves_far_approach_obstacle() -> None:
    evaluator = TaskGraspEvaluator(16, ASSET).eval()
    generator = torch.Generator().manual_seed(11)
    near = torch.rand((1, 1500, 3), generator=generator)
    near[..., 0] = (near[..., 0] - 0.5) * 0.08
    near[..., 1] = (near[..., 1] - 0.5) * 0.04
    near[..., 2] = -0.02 - near[..., 2] * 0.03
    obstacle = torch.tensor([[[0.07, 0.03, -0.220]]])
    xyz = torch.cat((near, obstacle), 1)
    cloud, _, mask = evaluator._local_cloud(
        xyz,
        torch.zeros_like(xyz),
        torch.ones((1, len(xyz[0])), dtype=torch.bool),
        torch.ones((1, len(xyz[0]))),
        torch.ones((1, len(xyz[0]))),
        torch.zeros((1, 1, 3)),
        torch.eye(3).reshape(1, 1, 3, 3),
    )
    selected = cloud[0, 0, : evaluator.scene_points][mask[0, 0, : evaluator.scene_points]]
    assert bool((selected[:, 2] < -0.205).any())


def test_stageb_backward_updates_only_evaluator(fake_graspnet, tiny_batch) -> None:
    model = TCDPRGModel(ModelConfig(feature_dim=32, task_dim=16))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("task_grasp."))
    batch = dict(tiny_batch)
    batch["region_valid"] = batch["point_mask"].clone()
    batch["region_target"] = batch["target_mask"].clone()
    batch["visibility_target"] = torch.ones(1)
    batch["visibility_valid"] = torch.ones(1, dtype=torch.bool)
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
    batch["stageb_condition"] = stageb_condition_from_gt(batch)
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


def test_formal_objective_stageb_path_forward_and_backward(fake_graspnet, tiny_batch) -> None:
    config = ModelConfig(feature_dim=32, task_dim=16)
    model = TCDPRGModel(config)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("task_grasp."))
    batch = dict(tiny_batch)
    batch["region_valid"] = batch["point_mask"].clone()
    batch["region_target"] = batch["target_mask"].clone()
    batch["visibility_target"] = torch.ones(1)
    batch["visibility_valid"] = torch.ones(1, dtype=torch.bool)
    batch["stageb_candidates"] = {
        "translation_world": batch["xyz"][:, :2].clone(),
        "rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3).expand(1, 2, -1, -1),
        "valid": torch.ones((1, 2), dtype=torch.bool),
        "width_m": torch.tensor([[0.04, 0.05]]),
    }
    batch["stageb_candidate_valid"] = torch.ones((1, 2), dtype=torch.bool)
    batch["stageb_label"] = torch.tensor([[False, True]])
    objective = TCDPRGObjective(
        DatasetCapabilities(has_task_grasps=True),
        config,
        AblationConfig(),
        LossConfig(
            instance=0.0, region=0.0, task_grasp=1.0,
            push_object=0.0, push_contact=0.0, push_direction=0.0, push_potential=0.0,
        ),
    )
    loss, _terms, output = objective(model, batch, return_output=True)
    loss.backward()
    assert isinstance(output["stageb_condition"], StageBCondition)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.task_grasp.parameters())


def test_gt_condition_has_the_public_stageb_contract(tiny_batch) -> None:
    batch = dict(tiny_batch)
    batch["region_valid"] = batch["point_mask"].clone()
    batch["region_target"] = batch["target_mask"].clone()
    condition = stageb_condition_from_gt(batch)
    assert isinstance(condition, StageBCondition)
    assert condition.target_probability.shape == batch["point_mask"].shape
    assert condition.region_probability.shape == batch["point_mask"].shape
    assert condition.target_valid.dtype == torch.bool
    assert set(condition.target_probability.unique().tolist()) <= {0.0, 1.0}


def test_task_grasp_signature_has_no_stagea_latents() -> None:
    import inspect

    parameters = inspect.signature(TaskGraspEvaluator.forward).parameters
    assert not {"point_features", "target_token", "task_token", "object_tokens"} & set(parameters)


def test_perception_returns_stageb_condition(fake_graspnet, tiny_batch) -> None:
    model = TCDPRGModel(ModelConfig(feature_dim=32, task_dim=16)).eval()
    with torch.no_grad():
        condition = model.forward_perception(tiny_batch)["stageb_condition"]
    assert isinstance(condition, StageBCondition)
    condition.validate(tiny_batch["xyz"].shape[1])


def test_identical_condition_uses_identical_crop_proposals_and_logits(
    fake_graspnet, tiny_batch
) -> None:
    model = TCDPRGModel(ModelConfig(feature_dim=32, task_dim=16)).eval()
    batch = dict(tiny_batch)
    batch["region_valid"] = batch["point_mask"].clone()
    batch["region_target"] = batch["target_mask"].clone()
    gt = stageb_condition_from_gt(batch)
    perfect_prediction = StageBCondition(
        gt.target_probability.clone(), gt.region_probability.clone(), gt.target_valid.clone(),
        gt.task_category_id.clone(), gt.task_region_id.clone(),
    )
    sensor = model._sensor(batch)
    with torch.no_grad():
        first = model.generate_target_grasp_proposals(sensor, gt)
        second = model.generate_target_grasp_proposals(sensor, perfect_prediction)
        first_score = model.forward_task_grasp_from_condition(sensor, gt, first)
        second_score = model.forward_task_grasp_from_condition(sensor, perfect_prediction, second)
    assert torch.equal(first["target_crop_mask"], second["target_crop_mask"])
    assert torch.equal(first["translation_world"], second["translation_world"])
    assert torch.equal(first_score["task_valid_logit"], second_score["task_valid_logit"])
