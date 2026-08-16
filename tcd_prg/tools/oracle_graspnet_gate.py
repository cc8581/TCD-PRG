"""Audit frozen GraspNet proposals on an exact Camera2 GT target crop."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tcd_prg.config import load_config
from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets import collate_unified, load_object_grasps, match_object_grasp_priors
from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset
from tcd_prg.geometry.camera import (
    camera_to_world_points,
    camera_to_world_rotations,
    graspnet_to_tcd_rotation,
    look_at_rotation_world_camera,
    world_to_camera_points,
)
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix
from tcd_prg.models.graspnet.adapter import FrozenGraspNetProposalGenerator
from tcd_prg.runtime import create_adapter


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--observation-cache", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, np.float64)
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)),
    }


def main() -> None:
    args = _arguments()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the oracle GraspNet gate")
    overrides = [
        f"dataset.root={args.dataset_root}",
        f"cache.directory={args.observation_cache}",
    ]
    config = load_config(args.config, overrides)
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter, split=args.split, max_groups=args.max_groups, include_strata=False
    )
    graspnet = FrozenGraspNetProposalGenerator(
        config.graspnet.source_root, config.graspnet.checkpoint,
        proposal_count=config.graspnet.target_proposals,
        input_points=config.graspnet.target_input_points,
        freeze=True, num_view=config.graspnet.num_view,
        num_angle=config.graspnet.num_angle, num_depth=config.graspnet.num_depth,
        cylinder_radius=config.graspnet.cylinder_radius, hmin=config.graspnet.hmin,
        hmax_list=config.graspnet.hmax_list,
    )
    manifest = json.loads((Path(args.database_root) / "manifest.json").read_text("utf-8"))
    by_model: dict[str, list[tuple[float, Path]]] = {}
    for record in manifest["records"]:
        path = Path(record["path"])
        scale = record.get("object_scale")
        if scale is None:
            scale = float(load_object_grasps(path)["object_scale"])
        by_model.setdefault(str(record["model_id"]), []).append((float(scale), path))
    totals: Counter[str] = Counter()
    hypothesis_totals: dict[str, Counter[str]] = {
        name: Counter() for name in (
            "anchor_vs_acronym_closing_center",
            "anchor_vs_acronym_raw_tcp",
            "anchor_plus_depth_vs_acronym_closing_center",
            "anchor_minus_depth_vs_acronym_closing_center",
        )
    }
    crop_points: list[float] = []
    nearest_pos_translation: list[float] = []
    nearest_pos_rotation: list[float] = []
    depth_values: list[float] = []
    per_group: list[dict[str, object]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        observation = sample.observation
        batch = collate_unified(
            [sample], grid_size_m=config.backbone.grid_size_m, training=False,
            point_count=config.dataset.scene_points,
            graspnet_point_count=config.graspnet.target_input_points,
            graspnet_view_index=config.graspnet.camera_view_index,
        )
        sensor = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        target_object = int(observation.target_object)
        crop = (
            sensor["graspnet_point_mask"].bool()
            & (sensor["graspnet_instance_id"].long() == target_object)
            & sensor["camera2_valid"][:, None].bool()
        )
        count = int(crop.sum())
        crop_points.append(float(count))
        totals["groups"] += 1
        if count < config.graspnet.target_min_crop_points:
            totals["crop_too_small"] += 1
            continue
        rotation_world_camera = look_at_rotation_world_camera(
            sensor["camera2_eye_world"], sensor["camera2_target_world"], sensor["camera2_up_world"]
        )
        xyz_camera = world_to_camera_points(
            sensor["graspnet_xyz_world"], rotation_world_camera, sensor["camera2_eye_world"]
        )
        proposal = graspnet(
            xyz_camera, crop, proposal_count=config.graspnet.target_proposals,
            input_points=config.graspnet.target_input_points,
        )
        translation_world = camera_to_world_points(
            proposal["translation_world"], rotation_world_camera, sensor["camera2_eye_world"]
        )
        rotation_world = camera_to_world_rotations(
            graspnet_to_tcd_rotation(proposal["rotation_matrix"]), rotation_world_camera
        )
        pose = torch.as_tensor(observation.object_pose[target_object], dtype=torch.float32, device=device)
        rotation_world_object = quaternion_xyzw_to_matrix(pose[3:])
        translation_object = torch.einsum(
            "ij,kj->ki", rotation_world_object.transpose(0, 1), translation_world[0] - pose[:3]
        )
        rotation_object = torch.einsum(
            "ij,kjl->kil", rotation_world_object.transpose(0, 1), rotation_world[0]
        )
        model_id = str(observation.metadata["object_model_id"][target_object])
        object_scale = float(observation.metadata["object_scale"][target_object])
        _, database_path = min(
            by_model[model_id], key=lambda item: abs(item[0] - object_scale)
        )
        database = load_object_grasps(database_path)
        database_translation = torch.from_numpy(database["translation_object"]).to(device)
        database_rotation = torch.from_numpy(database["rotation_object"]).to(device)
        database_status = torch.from_numpy(database["status"]).to(device)
        approach_object = rotation_object[:, :, 2]
        depth = proposal["depth_m"][0, :, None]
        raw_tcp = database_translation - database_rotation[:, :, 2] * 0.089
        hypotheses = {
            "anchor_vs_acronym_closing_center": (translation_object, database_translation),
            "anchor_vs_acronym_raw_tcp": (translation_object, raw_tcp),
            "anchor_plus_depth_vs_acronym_closing_center": (
                translation_object + approach_object * depth, database_translation
            ),
            "anchor_minus_depth_vs_acronym_closing_center": (
                translation_object - approach_object * depth, database_translation
            ),
        }
        hypothesis_matches = {
            name: match_object_grasp_priors(
                proposal_translation, rotation_object, proposal["valid"][0],
                gt_translation, database_rotation, database_status,
            )
            for name, (proposal_translation, gt_translation) in hypotheses.items()
        }
        matched = hypothesis_matches["anchor_vs_acronym_closing_center"]
        valid = proposal["valid"][0]
        status = matched["status"]
        positive = int(((status == int(CandidateStatus.POSITIVE)) & valid).sum())
        negative = int(((status == int(CandidateStatus.NEGATIVE)) & valid).sum())
        unknown = int(((status == int(CandidateStatus.UNKNOWN_UNTESTED)) & valid).sum())
        conflicts = int((matched["match_conflict"] & valid).sum())
        totals.update(valid=int(valid.sum()), positive=positive, negative=negative,
                      unknown=unknown, conflicts=conflicts)
        for name, candidate_match in hypothesis_matches.items():
            candidate_status = candidate_match["status"]
            hypothesis_totals[name].update(
                valid=int(valid.sum()),
                positive=int(((candidate_status == int(CandidateStatus.POSITIVE)) & valid).sum()),
                negative=int(((candidate_status == int(CandidateStatus.NEGATIVE)) & valid).sum()),
                unknown=int(((candidate_status == int(CandidateStatus.UNKNOWN_UNTESTED)) & valid).sum()),
                conflicts=int((candidate_match["match_conflict"] & valid).sum()),
            )
        finite_t = matched["positive_translation_error_m"][valid]
        finite_r = matched["positive_rotation_error_deg"][valid]
        nearest_pos_translation.extend(finite_t[torch.isfinite(finite_t)].cpu().tolist())
        nearest_pos_rotation.extend(finite_r[torch.isfinite(finite_r)].cpu().tolist())
        depth_values.extend(proposal["depth_m"][0, valid].cpu().tolist())
        per_group.append({
            "index": index, "model_id": model_id, "crop_points": count,
            "valid": int(valid.sum()), "positive": positive, "negative": negative,
            "unknown": unknown, "conflicts": conflicts,
        })
        print(json.dumps(per_group[-1], ensure_ascii=False), flush=True)

    valid_total = max(1, totals["valid"])
    report = {
        "protocol": "Camera2 exact GT instance crop -> frozen official GraspNet -> proposal x 2000 ACRONYM GT",
        "device": str(device), "checkpoint": config.graspnet.checkpoint,
        "groups_requested": args.max_groups, "totals": dict(totals),
        "rates_over_valid": {
            "positive": totals["positive"] / valid_total,
            "negative": totals["negative"] / valid_total,
            "known": (totals["positive"] + totals["negative"]) / valid_total,
            "conflict": totals["conflicts"] / valid_total,
        },
        "translation_definition_hypotheses": {
            name: {
                "totals": dict(counts),
                "known_rate": (counts["positive"] + counts["negative"]) / max(1, counts["valid"]),
                "positive_rate": counts["positive"] / max(1, counts["valid"]),
            }
            for name, counts in hypothesis_totals.items()
        },
        "crop_points": _summary(crop_points),
        "nearest_positive_translation_m": _summary(nearest_pos_translation),
        "nearest_positive_rotation_deg": _summary(nearest_pos_rotation),
        "proposal_depth_m": _summary(depth_values),
        "per_group": per_group,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("totals", "rates_over_valid")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
