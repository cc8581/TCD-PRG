"""Build the one-time GraspNet-distribution Stage-B binary dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import ActionStateGroupDataset
from tcd_prg.datasets.stageb_manifest import (
    SCHEMA_VERSION,
    build_provenance,
    compatibility_provenance,
)
from tcd_prg.geometry.stageb_grasp import evaluate_stageb_geometry
from tcd_prg.models import TCDPRGModel, stageb_condition_from_gt
from tcd_prg.runtime import UnifiedBatchCollator, create_adapter


def model_sensor(batch: dict) -> dict:
    return {
        key: batch[key]
        for key in (
            "xyz",
            "rgb",
            "point_mask",
            "source_view",
            "grid_coord",
            "graspnet_xyz_world",
            "graspnet_point_mask",
            "camera2_eye_world",
            "camera2_target_world",
            "camera2_up_world",
            "camera2_valid",
        )
        if key in batch
    }


def balance_binary_records(root: Path, records: list[dict], seed: int) -> list[dict]:
    """Balance within task-region strata, never globally."""
    payloads = []
    strata: dict[int, dict[bool, list[tuple[int, int]]]] = {}
    for record_index, record in enumerate(records):
        path = root / record["path"]
        payload = {key: value for key, value in np.load(path).items()}
        labels = payload["task_valid"].astype(bool)
        key = int(record.get("task_region_id", -1))
        bucket = strata.setdefault(key, {True: [], False: []})
        bucket[True].extend((record_index, int(index)) for index in np.flatnonzero(labels))
        bucket[False].extend((record_index, int(index)) for index in np.flatnonzero(~labels))
        payloads.append(payload)
    rng = np.random.default_rng(seed)
    kept_positive: set[tuple[int, int]] = set()
    kept_negative: set[tuple[int, int]] = set()
    for key, bucket in sorted(strata.items()):
        positive, negative = bucket[True], bucket[False]
        if not positive or not negative:
            raise RuntimeError(f"Stage-B stratum {key} must contain both binary labels")
        rng.shuffle(positive)
        rng.shuffle(negative)
        positive_count = min(len(positive), len(negative))
        kept_positive.update(positive[:positive_count])
        kept_negative.update(negative[: 2 * positive_count])
    balanced: list[dict] = []
    for record_index, (record, payload) in enumerate(zip(records, payloads, strict=True)):
        label = payload["task_valid"].astype(bool)
        keep = np.zeros_like(label)
        for candidate in np.flatnonzero(label):
            keep[candidate] = (record_index, int(candidate)) in kept_positive
        for candidate in np.flatnonzero(~label):
            keep[candidate] = (record_index, int(candidate)) in kept_negative
        if not keep.any():
            (root / record["path"]).unlink()
            continue
        balanced_payload = {
            key: (value[keep] if value.ndim and value.shape[0] == len(label) else value)
            for key, value in payload.items()
        }
        np.savez_compressed(root / record["path"], **balanced_payload)
        balanced.append(
            {
                **record,
                "candidate_count": int(keep.sum()),
                "positive_count": int(label[keep].sum()),
            }
        )
    return balanced


def finalize_split_records(
    root: Path, split: str, records: list[dict], seed: int
) -> list[dict]:
    """Balance optimization data only; keep validation at natural proposal prior."""
    return balance_binary_records(root, records, seed) if split == "train" else records


def assert_no_train_validation_leakage(records: list[dict]) -> None:
    """A task state may belong to exactly one Stage-B split."""
    split_by_key: dict[tuple[int, int, int], str] = {}
    for record in records:
        key = (int(record["scene_id"]), int(record["state_id"]), int(record["task_index"]))
        split = str(record["split"])
        prior = split_by_key.setdefault(key, split)
        if prior != split:
            raise RuntimeError(
                "Stage-B train/validation leakage detected for "
                f"scene/state/task={key}: {prior!r} and {split!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage/grasp.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    root = Path(args.output)
    manifest_path = root / "manifest.json"
    provenance = build_provenance(config)
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if compatibility_provenance(old.get("provenance", {})) != compatibility_provenance(
            provenance
        ):
            raise RuntimeError("Existing Stage-B split was built with different provenance")
    adapter = create_adapter(config, allow_render=False)
    dataset = ActionStateGroupDataset(
        adapter,
        split=args.split,
        max_groups=args.max_groups,
        deduplicate_state_task=True,
        global_grasp_mode="never",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=UnifiedBatchCollator(config, training=False, include_graspnet=True),
    )
    model = TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ag_geometry = np.load(config.model.stageb_label_gripper_geometry)
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    records = []
    audit_path = root / "build_audit.jsonl"
    audit = audit_path.open("a", encoding="utf-8")
    for index, batch in enumerate(loader):
        batch = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        sensor = model_sensor(batch)
        condition = stageb_condition_from_gt(batch)
        proposals = model.generate_target_grasp_proposals(sensor, condition)
        # Label the complete deployment proposal distribution, including the
        # diversity tail emitted after the quality prefix.
        keep = torch.nonzero(proposals["valid"][0], as_tuple=False).flatten()
        if not len(keep):
            diagnostic = {
                "record": index,
                "dropped": "no_valid_graspnet_proposals",
                "target_identity_valid": bool(proposals["target_identity_valid"][0]),
                "target_crop_points": int(proposals["target_crop_points"][0]),
            }
            audit.write(json.dumps(diagnostic) + "\n")
            print(f"[stageb-data] {index + 1}/{len(dataset)} {diagnostic}", flush=True)
        labels: list[bool] = []
        reasons: list[tuple[str, ...]] = []
        kept_indices: list[int] = []
        for candidate in keep.tolist():
            try:
                result = evaluate_stageb_geometry(
                    batch["xyz"][0].cpu().numpy(),
                    batch["point_mask"][0].cpu().numpy(),
                    batch["target_mask"][0].cpu().numpy(),
                    batch["region_target"][0].cpu().numpy(),
                    batch["region_valid"][0].cpu().numpy(),
                    proposals["translation_world"][0, candidate].cpu().numpy(),
                    proposals["rotation_matrix"][0, candidate].cpu().numpy(),
                    float(proposals["width_m"][0, candidate]),
                    ag_geometry["points_tcp"],
                    ag_geometry["part_id"],
                )
            except (ValueError, FloatingPointError) as error:
                audit.write(
                    json.dumps({"record": index, "candidate": candidate, "dropped": str(error)})
                    + "\n"
                )
                continue
            kept_indices.append(candidate)
            labels.append(result.task_valid)
            reasons.append(result.reasons)
        if not kept_indices:
            continue
        selected = list(range(len(labels)))
        unit = dataset.units[index]
        path = (
            records_dir
            / f"{args.split}_{unit.scene_id:04d}_{unit.state_id:03d}_{unit.task_index:03d}.npz"
        )
        proposal_index = torch.tensor([kept_indices[i] for i in selected], device=device)
        np.savez_compressed(
            path,
            translation_world=proposals["translation_world"][0, proposal_index]
            .cpu()
            .numpy()
            .astype(np.float32),
            rotation_matrix=proposals["rotation_matrix"][0, proposal_index]
            .cpu()
            .numpy()
            .astype(np.float32),
            width_m=proposals["width_m"][0, proposal_index].cpu().numpy().astype(np.float32),
            task_valid=np.asarray([labels[i] for i in selected], np.bool_),
            context_xyz=batch["xyz"][0].cpu().numpy().astype(np.float32),
            context_rgb=batch["rgb"][0].cpu().numpy().astype(np.float32),
            context_point_mask=batch["point_mask"][0].cpu().numpy().astype(np.bool_),
            context_source_view=batch["source_view"][0].cpu().numpy().astype(np.int16),
            context_target_mask=batch["target_mask"][0].cpu().numpy().astype(np.bool_),
            context_region_target=batch["region_target"][0].cpu().numpy().astype(np.bool_),
            context_region_valid=batch["region_valid"][0].cpu().numpy().astype(np.bool_),
            context_grid_coord=batch["grid_coord"][0].cpu().numpy().astype(np.int32),
        )
        for local in selected:
            audit.write(
                json.dumps(
                    {
                        "record": index,
                        "candidate": kept_indices[local],
                        "task_valid": labels[local],
                        "reasons": reasons[local],
                    }
                )
                + "\n"
            )
        records.append(
            {
                "split": args.split,
                "scene_id": unit.scene_id,
                "state_id": unit.state_id,
                "task_index": unit.task_index,
                "group_index": unit.group_index,
                "task_region_id": int(batch["task_region_id"][0]),
                "object_category_id": int(batch["task_category_id"][0]),
                "path": str(path.relative_to(root)).replace("\\", "/"),
            }
        )
        positive_count = sum(labels[i] for i in selected)
        print(
            f"[stageb-data] {index + 1}/{len(dataset)} "
            f"evaluated={len(selected)} positive={positive_count}",
            flush=True,
        )
    audit.close()
    if not records:
        raise RuntimeError(
            "Stage-B construction produced no valid binary records; inspect build_audit.jsonl"
        )
    records = finalize_split_records(root, args.split, records, config.training.seed)
    old_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    existing = old_manifest.get("records", [])
    existing = [row for row in existing if row["split"] != args.split]
    prior_commits = old_manifest.get("provenance", {}).get("audit", {}).get(
        "producer_git_commits", []
    )
    current_commits = provenance["audit"]["producer_git_commits"]
    provenance["audit"]["producer_git_commits"] = sorted(
        {str(value) for value in (*prior_commits, *current_commits)}
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "records": existing + records,
    }
    assert_no_train_validation_leakage(payload["records"])
    temporary = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
