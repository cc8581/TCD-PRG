"""Unified full-model and ablation training entry point."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcd_prg.config import load_config
from tcd_prg.datasets import (
    ActionStateGroupDataset,
    DistributedEvaluationSampler,
    DistributedTaskStateBatchSampler,
    StageBAcronymDataset,
    StageBBinaryDataset,
)
from tcd_prg.datasets.stageb_manifest import build_dynamic_acronym_provenance, build_provenance
from tcd_prg.evaluators import OfflineModelEvaluator
from tcd_prg.losses import TCDPRGObjective
from tcd_prg.models import StandalonePushModel, TCDPRGModel
from tcd_prg.observation.cached import CachedObservationProvider
from tcd_prg.planners.candidate_generator import DenseCandidateGenerator
from tcd_prg.planners.push_decoder import (
    decode_push_candidates,
    proposal_recall_counts,
)
from tcd_prg.pretrained import load_pretrained_backbone, prepare_pretrained_checkpoint
from tcd_prg.runtime import (
    StageBBinaryBatchCollator,
    UnifiedBatchCollator,
    create_adapter,
)
from tcd_prg.trainers import Trainer, finalize_push_validation_metrics


def validate_read_through_observation_cache(adapter) -> dict[str, object]:
    """Validate the bounded cache and its renderer without scanning the dataset."""

    provider = adapter.observation_provider
    if not isinstance(provider, CachedObservationProvider):
        raise TypeError("Formal training requires CachedObservationProvider")
    usage = shutil.disk_usage(provider.cache_dir)
    if usage.free <= provider.min_free_bytes:
        raise OSError(
            f"Observation cache {provider.cache_dir} has {usage.free} free bytes, "
            f"below the configured reserve {provider.min_free_bytes}"
        )
    strict = provider.fallback is None
    return {
        "mode": "strict-cache-only" if strict else "read-through-lru",
        "directory": str(provider.cache_dir.resolve()),
        "max_gb": None if strict else round(provider.max_bytes / (1 << 30), 3),
        "free_gb": round(usage.free / (1 << 30), 3),
        "missing": "error" if strict else "render-on-demand",
    }


def load_or_create_validation_scene_subset(
    scene_ids: list[int] | tuple[int, ...],
    count: int | None,
    seed: int,
    manifest_path: str | Path,
) -> tuple[int, ...]:
    """Select complete validation scenes once and reuse the audited manifest."""

    available = tuple(sorted({int(scene_id) for scene_id in scene_ids}))
    if not available:
        raise ValueError("The validation split contains no scenes")
    if count is None:
        return available
    if count <= 0 or count > len(available):
        raise ValueError(f"training.validation_scene_count={count} must be in [1,{len(available)}]")
    path = Path(manifest_path)
    fingerprint = hashlib.sha256(
        json.dumps(available, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError("validation scene subset seed does not match the manifest")
        if int(payload.get("selected_scene_count", -1)) != int(count):
            raise ValueError("validation scene subset count does not match the manifest")
        if payload.get("source_scene_fingerprint") != fingerprint:
            raise ValueError("validation split scenes have changed since the manifest was written")
        selected = tuple(int(value) for value in payload.get("scene_ids", []))
        if len(selected) != count or len(set(selected)) != count:
            raise ValueError("validation scene subset manifest has invalid scene IDs")
        if not set(selected).issubset(available):
            raise ValueError("validation scene subset contains a scene outside the split")
        return selected

    selected = tuple(
        sorted(
            int(value)
            for value in np.random.default_rng(seed).choice(
                np.asarray(available, dtype=np.int64), size=count, replace=False
            )
        )
    )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "seed": int(seed),
            "source_scene_count": len(available),
            "source_scene_fingerprint": fingerprint,
            "selected_scene_count": int(count),
            "scene_ids": list(selected),
        }
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return selected


def restrict_action_dataset_to_stage(
    dataset,
    allowed_strata: tuple[str, ...],
    *,
    deduplicate_state_task: bool = False,
) -> None:
    """Restrict action groups without changing their published labels."""
    units = dataset.units
    if allowed_strata:
        allowed = frozenset(str(value) for value in allowed_strata)
        units = tuple(unit for unit in units if unit.stratum in allowed)
        if not units:
            raise RuntimeError(f"Stage filter {sorted(allowed)} removed every action-state group")
    if deduplicate_state_task:
        unique = {}
        for unit in units:
            unique.setdefault((unit.scene_id, unit.state_id, unit.task_index), unit)
        units = tuple(unique.values())
    dataset.units = units
    dataset.selected_group_count = len(units)


def validate_checkpoint_gate(
    stage: str,
    *,
    resume_payload: dict | None,
    weights_payload: dict | None = None,
) -> None:
    """Enforce independent A/B starts and the remaining B -> C transition."""

    if resume_payload is not None:
        source = resume_payload.get("training_stage")
        if source != stage:
            raise RuntimeError(f"Resume requires a {stage!r} checkpoint, got {source!r}")
        return
    if weights_payload is not None:
        source = weights_payload.get("training_stage")
        if source != stage:
            raise RuntimeError(
                f"Weights-only start requires a {stage!r} checkpoint, got {source!r}"
            )
        if int(weights_payload.get("schema_version", -1)) != 12:
            raise RuntimeError("Weights-only start requires a schema-12 TCD-PRG checkpoint")
    return


def load_model_weights_only(model: torch.nn.Module, payload: dict, stage: str) -> None:
    """Strictly initialize one stage without restoring any trainer state."""

    source = model.module if hasattr(model, "module") else model
    current = source.state_dict()
    supplied = payload.get("model")
    if not isinstance(supplied, dict):
        raise RuntimeError("Weights-only checkpoint does not contain a model state_dict")
    if set(current) != set(supplied):
        missing = sorted(set(current) - set(supplied))
        extra = sorted(set(supplied) - set(current))
        raise RuntimeError(
            f"{stage} weights-only parameter mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    mismatched = [name for name in current if current[name].shape != supplied[name].shape]
    if mismatched:
        raise RuntimeError(f"{stage} weights-only tensor shape mismatch: {mismatched[:5]}")
    source.load_state_dict(supplied, strict=True)


def build_optimizer_parameter_groups(model, config, pretrained_parameter_names=()):
    """Build stage-aware groups without assuming every model owns an encoder."""
    named = dict(model.named_parameters())
    missing = [name for name in pretrained_parameter_names if name not in named]
    if missing:
        raise RuntimeError(
            "Checkpoint pre-trained parameter names do not match this model: "
            + ", ".join(missing[:8])
        )
    if pretrained_parameter_names:
        low = [named[name] for name in pretrained_parameter_names if named[name].requires_grad]
        low_lr = config.optimizer.backbone_learning_rate
    elif config.training.stage == "push":
        low, low_lr = [], config.optimizer.learning_rate
    else:
        low = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
        low_lr = config.optimizer.learning_rate
    low_ids = {id(parameter) for parameter in low}
    other = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in low_ids
    ]
    groups = []
    if low:
        groups.append({"params": low, "lr": low_lr, "name": "pretrained_trunk"})
    if other:
        groups.append(
            {"params": other, "lr": config.optimizer.learning_rate, "name": "new_modules"}
        )
    if not groups:
        raise RuntimeError("Selected training stage has no trainable parameters")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume")
    parser.add_argument(
        "--weights-only-checkpoint",
        help="Initialize the complete same-stage model, but start optimizer/scheduler/steps fresh.",
    )
    # Windows 原生启动器显式传递进程拓扑；torchrun 的环境变量仅作为 Linux 兼容路径。
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument("--ddp-init-method")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one validation pass from --resume and skip optimizer updates.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.weights_only_checkpoint:
        raise ValueError("--resume and --weights-only-checkpoint are mutually exclusive")
    if args.validate_only and not args.resume:
        raise ValueError("--validate-only requires --resume")
    config = load_config(args.config, args.overrides)
    if config.training.stage == "joint":
        raise RuntimeError(
            "Legacy joint training is incompatible with standalone Stage-B/Stage-C condition protocols; train perception, grasp and push independently."
        )
    if config.observation.provider != "cached":
        raise ValueError(
            "Formal training requires observation.provider=cached for bounded read-through caching"
        )
    resume_payload = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    weights_payload = (
        torch.load(args.weights_only_checkpoint, map_location="cpu", weights_only=False)
        if args.weights_only_checkpoint else None
    )
    validate_checkpoint_gate(
        config.training.stage,
        resume_payload=resume_payload,
        weights_payload=weights_payload,
    )
    world_size = (
        args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    )
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    local_rank = (
        args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", "0"))
    )
    use_cuda = torch.cuda.is_available() and config.training.device.startswith("cuda")
    # Windows CUDA DDP 默认 gloo；Linux CUDA 优先 NCCL。每个 rank 绑定唯一显卡。
    if world_size > 1:
        backend = config.training.ddp_backend
        if backend == "auto":
            backend = "gloo" if os.name == "nt" else ("nccl" if use_cuda else "gloo")
        if use_cuda and local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices exist"
            )
        init_method = args.ddp_init_method
        if init_method:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
        else:
            torch.distributed.init_process_group(backend=backend)
        atexit.register(_destroy_process_group)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            config.training.device = f"cuda:{local_rank}"
    # 离线点云生成完成后可关闭 on-miss renderer。严格模式下缺失缓存直接报错，
    # 且训练启动时不会对离线缓存执行 LRU eviction。
    adapter = create_adapter(config, allow_render=config.observation.allow_render_on_miss)
    provider = adapter.observation_provider
    if (
        rank == 0
        and isinstance(provider, CachedObservationProvider)
        and provider.fallback is not None
    ):
        provider.evict()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    observation_cache = validate_read_through_observation_cache(adapter)
    stageb = config.training.stage == "grasp"
    stageb_provenance = None
    if stageb:
        if resume_payload is not None:
            stageb_provenance = resume_payload.get("stageb_provenance")
            if not stageb_provenance:
                raise RuntimeError("Stage-B resume checkpoint is missing dataset provenance")
        else:
            stageb_provenance = (
                build_dynamic_acronym_provenance(config)
                if config.dataset.stageb_source == "acronym_dynamic"
                else build_provenance(config)
            )
    perception_only_stage = (
        config.losses.task_grasp == 0
        and config.losses.push_object == 0
        and config.losses.push_contact == 0
        and config.losses.push_direction == 0
        and config.losses.push_potential == 0
    )
    stageb_root = config.dataset.stageb_acronym_root or str(
        Path(config.dataset.root) / config.dataset.step_labels_subdir / "acronym_binary_grasps"
    )
    train_dataset = (
        StageBAcronymDataset(
            adapter,
            stageb_root,
            "train",
            config.training.max_train_groups,
            config.dataset.stageb_positive_per_state,
            config.dataset.stageb_negative_per_state,
            config.training.seed,
        )
        if stageb and config.dataset.stageb_source == "acronym_dynamic"
        else StageBBinaryDataset(
            adapter,
            config.dataset.stageb_binary_root,
            "train",
            config.training.max_train_groups,
            stageb_provenance,
        )
        if stageb
        else ActionStateGroupDataset(
            adapter,
            split="train",
            max_groups=config.training.max_train_groups,
            allowed_strata=config.training.allowed_action_strata,
            deduplicate_state_task=perception_only_stage,
            global_grasp_mode="never",
        )
    )
    validation_scene_ids = (
        load_or_create_validation_scene_subset(
            adapter.scene_splits["val"],
            config.training.validation_scene_count,
            config.training.validation_scene_seed,
            os.path.join(config.output_dir, "validation_scene_subset.json"),
        )
        if config.training.validation_interval > 0 and not stageb
        else ()
    )
    if config.training.validation_interval <= 0:
        validation_dataset = None
    elif stageb:
        validation_dataset = (
            StageBAcronymDataset(
                adapter,
                stageb_root,
                "val",
                config.training.max_validation_groups,
                config.dataset.stageb_positive_per_state,
                config.dataset.stageb_negative_per_state,
                config.training.seed + 1,
            )
            if config.dataset.stageb_source == "acronym_dynamic"
            else StageBBinaryDataset(
                adapter,
                config.dataset.stageb_binary_root,
                "val",
                config.training.max_validation_groups,
                stageb_provenance,
            )
        )
    else:
        validation_dataset = ActionStateGroupDataset(
            adapter,
            split="val",
            scene_ids=frozenset(validation_scene_ids),
            max_groups=config.training.max_validation_groups,
            allowed_strata=config.training.allowed_action_strata,
            deduplicate_state_task=perception_only_stage,
            stratified_max_groups=config.training.max_validation_groups is not None,
            stratum_quota=config.training.validation_stratum_quota,
            subset_manifest_path=(
                os.path.join(config.output_dir, "validation_subset.json")
                if config.training.max_validation_groups is not None
                else None
            ),
            global_grasp_mode="never",
            global_grasp_width_bounds=(
                config.model.min_grasp_width_m,
                config.model.max_grasp_width_m,
            ),
        )
    if not len(train_dataset):
        raise RuntimeError("The completed-file snapshot contains no training groups")
    if rank == 0:
        print(
            f"[observation-cache] {json.dumps(observation_cache, ensure_ascii=False)}", flush=True
        )
        print(
            f"[train-data] fraction={config.training.data_fraction:g} "
            f"split_ratios={list(config.training.split_ratios)} "
            f"scenes={len(adapter.scene_splits['train'])} groups={len(train_dataset)} "
            f"source_groups={train_dataset.source_group_count}",
            flush=True,
        )
        print(
            f"[train-strata] {json.dumps(train_dataset.stratum_counts, ensure_ascii=False)}",
            flush=True,
        )
        if validation_dataset is not None:
            print(
                f"[val-data] scenes={len(validation_scene_ids)}/"
                f"{len(adapter.scene_splits['val'])} "
                f"groups={len(validation_dataset)}",
                flush=True,
            )
    train_collator = (
        StageBBinaryBatchCollator(config, training=True)
        if stageb
        else UnifiedBatchCollator(config, training=True)
    )
    validation_collator = (
        StageBBinaryBatchCollator(config, training=False)
        if stageb
        else UnifiedBatchCollator(config, training=False)
    )
    if stageb:
        train_sampler = (
            torch.utils.data.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.training.seed,
            )
            if world_size > 1
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.num_workers > 0,
            collate_fn=train_collator,
        )
    else:
        train_batch_sampler = DistributedTaskStateBatchSampler(
            train_dataset.units,
            batch_size=config.training.batch_size,
            coverage_strata=config.training.action_batch_coverage_strata,
            rank=rank,
            world_size=world_size,
            seed=config.training.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.num_workers > 0,
            collate_fn=train_collator,
        )
    validation_sampler = (
        DistributedEvaluationSampler(len(validation_dataset), rank, world_size)
        if world_size > 1
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            # Validation starts while both persistent training loaders remain
            # alive.  A separate setting prevents a third Windows worker pool
            # and its prefetched queue copies from exhausting commit memory.
            num_workers=config.training.validation_num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.validation_num_workers > 0,
            collate_fn=validation_collator,
        )
        if validation_dataset is not None and len(validation_dataset)
        else None
    )
    model = (
        StandalonePushModel(config.model)
        if config.training.stage == "push"
        else TCDPRGModel(config.model, config.ablation, config.backbone, config.graspnet)
    )
    if weights_payload is not None:
        load_model_weights_only(model, weights_payload, config.training.stage)
        if rank == 0:
            print(
                f"[weights-only] initialized complete {config.training.stage} model from "
                f"{Path(args.weights_only_checkpoint).resolve()}; trainer state starts at step 0",
                flush=True,
            )
    if config.training.stage == "push":
        model.push_evaluator.requires_grad_(False)
    pretrained_report = None
    resume_pretrained_names: list[str] = []
    resume_validation_protocol_changed = False
    if args.resume:
        assert resume_payload is not None
        resume_config = resume_payload.get("config", {})
        resume_extra = resume_config.get("extra", {}) if isinstance(resume_config, dict) else {}
        resume_pretrained_names = list(resume_extra.get("pretrained_matched_parameter_names", []))
        # Older pre-minimal-architecture checkpoints may have been written before the
        # pre-trained parameter-name list was copied into ``config.extra``.
        # The optimizer still contains the original split, so falling back to
        # every encoder parameter changes the parameter-group sizes and makes
        # optimizer state restoration fail.  The run artifact contains the
        # exact audited list used to construct that optimizer.
        if not resume_pretrained_names:
            pretrained_report_path = Path(config.output_dir) / "pretrained_backbone.json"
            if pretrained_report_path.is_file():
                persisted_report = json.loads(pretrained_report_path.read_text(encoding="utf-8"))
                resume_pretrained_names = list(persisted_report.get("matched_parameter_names", []))
        resume_training = (
            resume_config.get("training", {}) if isinstance(resume_config, dict) else {}
        )
        resume_model = resume_config.get("model", {}) if isinstance(resume_config, dict) else {}
        resume_evaluation = (
            resume_config.get("evaluation", {}) if isinstance(resume_config, dict) else {}
        )
        resume_ablation = (
            resume_config.get("ablation", {}) if isinstance(resume_config, dict) else {}
        )
        if isinstance(resume_training, dict):
            old_validation_signature = (
                resume_training.get("validation_scene_count"),
                int(resume_training.get("validation_scene_seed", 2026)),
                resume_training.get("max_validation_groups"),
                float(resume_training.get("push_coverage_penalty_weight", 1.0)),
                resume_training.get("validation_family_weights"),
                tuple(resume_training.get("allowed_action_strata", ())),
                resume_training.get("validation_stratum_quota"),
                resume_model.get("push_object_topk"),
                resume_model.get("push_candidates"),
                resume_model.get("push_directions_per_contact"),
                resume_model.get("max_push_candidates"),
                resume_model.get("push_candidate_probability_threshold"),
                resume_model.get("push_utility_threshold"),
                resume_model.get("push_nms_contact_m"),
                resume_model.get("push_nms_direction_deg"),
                resume_evaluation.get("push_match_contact_m"),
                resume_evaluation.get("push_match_direction_deg"),
                resume_ablation.get("use_push_potential"),
            )
            current_validation_signature = (
                config.training.validation_scene_count,
                int(config.training.validation_scene_seed),
                config.training.max_validation_groups,
                float(config.training.push_coverage_penalty_weight),
                config.training.validation_family_weights,
                tuple(config.training.allowed_action_strata),
                config.training.validation_stratum_quota,
                config.model.push_object_topk,
                config.model.push_candidates,
                config.model.push_directions_per_contact,
                config.model.max_push_candidates,
                config.model.push_candidate_probability_threshold,
                config.model.push_utility_threshold,
                config.model.push_nms_contact_m,
                config.model.push_nms_direction_deg,
                config.evaluation.push_match_contact_m,
                config.evaluation.push_match_direction_deg,
                config.ablation.use_push_potential,
            )
            resume_validation_protocol_changed = (
                old_validation_signature != current_validation_signature
            )
            config.training.frozen_modules = tuple(
                resume_training.get("frozen_modules", config.training.frozen_modules)
            )
            config.training.unfreeze_at_optimizer_step = resume_training.get(
                "unfreeze_at_optimizer_step",
                config.training.unfreeze_at_optimizer_step,
            )
    if (
        not args.resume
        and weights_payload is None
        and config.training.stage not in {"grasp", "push"}
    ):
        distributed = torch.distributed.is_initialized()
        checkpoint_path = None
        # Only rank zero may resolve a missing managed checkpoint. Other ranks
        # wait until the atomic download and checksum validation are complete.
        if rank == 0 or not distributed:
            checkpoint_path = prepare_pretrained_checkpoint(config.backbone, allow_download=True)
        if distributed:
            torch.distributed.barrier()
        if rank != 0:
            checkpoint_path = prepare_pretrained_checkpoint(config.backbone, allow_download=False)
        if checkpoint_path is not None:
            pretrained_report = load_pretrained_backbone(model, checkpoint_path, config.backbone)
            config.extra["pretrained_matched_parameter_names"] = list(
                pretrained_report["matched_parameter_names"]
            )
            if rank == 0:
                os.makedirs(config.output_dir, exist_ok=True)
                with open(
                    os.path.join(config.output_dir, "pretrained_backbone.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(pretrained_report, handle, ensure_ascii=False, indent=2)
                print(
                    "[pretrained] matched_parameter_fraction="
                    f"{pretrained_report['matched_parameter_fraction']:.1%}",
                    flush=True,
                )
    if config.backbone.freeze and "encoder" not in config.training.frozen_modules:
        config.training.frozen_modules = (*config.training.frozen_modules, "encoder")
    elif pretrained_report and config.backbone.pretrained_freeze_steps > 0:
        prefixes = tuple(pretrained_report["freeze_prefixes"])
        config.training.frozen_modules = tuple(
            dict.fromkeys((*config.training.frozen_modules, *prefixes))
        )
        config.training.unfreeze_at_optimizer_step = max(
            config.backbone.pretrained_freeze_steps,
            config.training.unfreeze_at_optimizer_step or 0,
        )

    # For fixed stages, freeze unused modules before optimizer/DDP construction.
    # This avoids optimizer state and DDP reducer bookkeeping for branches that
    # the stage never executes. Dynamic unfreeze experiments retain Trainer's
    # delayed freeze path so those parameters stay inside the optimizer.
    if config.training.frozen_modules and config.training.unfreeze_at_optimizer_step is None:
        for parameter_name, parameter in model.named_parameters():
            if any(
                parameter_name == prefix or parameter_name.startswith(prefix + ".")
                for prefix in config.training.frozen_modules
            ):
                parameter.requires_grad_(False)

    pretrained_parameter_names = (
        list(pretrained_report["matched_parameter_names"])
        if pretrained_report is not None
        else resume_pretrained_names
    )
    optimizer_groups = build_optimizer_parameter_groups(model, config, pretrained_parameter_names)
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config.optimizer.weight_decay)

    def learning_rate(step: int) -> float:
        # 线性 warmup 后按优化器步执行 cosine decay；AMP skip 不会推进此调度器。
        if step < config.scheduler.warmup_steps:
            return max(1e-8, step / max(1, config.scheduler.warmup_steps))
        progress = (step - config.scheduler.warmup_steps) / max(
            1, config.training.max_optimizer_steps - config.scheduler.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    if world_size > 1:
        model = model.to(torch.device(config.training.device))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=config.training.ddp_find_unused_parameters,
        )
    # 数据能力与消融开关共同决定实际启用的任务损失，结果写入 loss_routing.json。
    objective = TCDPRGObjective(
        adapter.capabilities,
        config.model,
        config.ablation,
        config.losses,
        config.region_head,
    )
    if rank == 0:
        enabled = {name: objective.total.enabled(name) for name in objective.total.DEFAULT_WEIGHTS}
        output = os.path.join(config.output_dir, "loss_routing.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_capabilities": asdict(adapter.capabilities),
                    "enabled": enabled,
                    "family_weights": objective.total.weights,
                    "action_sampling": "task_state_first_best_effort_stratum_coverage",
                    "action_batch_coverage_strata": list(
                        config.training.action_batch_coverage_strata
                    ),
                    "disabled": {
                        name: "dataset capability or ablation"
                        for name, value in enabled.items()
                        if not value
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    trainer = Trainer(
        model,
        optimizer,
        config,
        objective,
        scheduler=scheduler,
        output_dir=config.output_dir,
        stageb_provenance=stageb_provenance,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
        if resume_validation_protocol_changed:
            previous_best = trainer.state.best_validation
            trainer.state.best_validation = float("inf")
            trainer.state.validation_without_improvement = 0
            trainer._write_event(
                "validation_protocol_reset",
                previous_best_validation=previous_best,
                validation_scene_count=config.training.validation_scene_count,
                validation_scene_seed=config.training.validation_scene_seed,
                max_validation_groups=config.training.max_validation_groups,
            )
            if rank == 0:
                print(
                    "[validation-protocol] subset changed on resume; reset best "
                    f"validation from {previous_best:.6f} and early-stopping counter",
                    flush=True,
                )

    def validate(module: torch.nn.Module) -> dict[str, object]:
        if validation_loader is None:
            return {
                "score_sum": float("inf"),
                "score_count": 1,
                "metric_sums": {},
                "metric_counts": {},
            }
        module.eval()
        total, count = 0.0, 0
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        evaluator = OfflineModelEvaluator(
            config.model,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.evaluation,
        )
        validation_batches = len(validation_loader)
        validation_groups = len(validation_loader.sampler)
        stageb_scores: list[np.ndarray] = []
        stageb_targets: list[np.ndarray] = []
        stageb_deployment_scores: list[np.ndarray] = []
        stageb_deployment_targets: list[np.ndarray] = []
        deployment_selector = DenseCandidateGenerator(config.model) if stageb else None
        with torch.no_grad():
            for validation_step, raw in enumerate(validation_loader, start=1):
                batch = trainer._move(raw, trainer.device)
                _, terms, model_output = objective(module, batch, return_output=True)
                terms = dict(terms)
                if config.training.stage == "push":
                    pre_nms, final = decode_push_candidates(
                        model_output["sensor"],
                        model_output["push_condition"],
                        model_output["push"],
                        config.model,
                        use_push_potential=config.ablation.use_push_potential,
                    )
                    pre_hits, proposal_total = proposal_recall_counts(
                        pre_nms,
                        batch,
                        contact_threshold_m=config.evaluation.push_match_contact_m,
                        direction_threshold_deg=config.evaluation.push_match_direction_deg,
                    )
                    final_hits, final_total = proposal_recall_counts(
                        final,
                        batch,
                        contact_threshold_m=config.evaluation.push_match_contact_m,
                        direction_threshold_deg=config.evaluation.push_match_direction_deg,
                    )
                    if not bool(torch.equal(proposal_total, final_total)):
                        raise RuntimeError("PUSH proposal recall denominator changed across NMS")
                    terms.update(
                        {
                            "push_proposal_positive_total_count": proposal_total,
                            "push_proposal_positive_pre_nms_hits_count": pre_hits,
                            "push_proposal_positive_final_hits_count": final_hits,
                        }
                    )
                if stageb:
                    valid = (
                        batch["stageb_candidate_valid"].bool()
                        & model_output["task_grasp"]["valid"].bool()
                    )
                    stageb_scores.append(
                        model_output["task_grasp"]["task_valid_probability"][valid]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    stageb_targets.append(batch["stageb_label"][valid].detach().cpu().numpy())
                    task = model_output["task_grasp"]
                    assert deployment_selector is not None
                    for row in range(valid.shape[0]):
                        selected = deployment_selector.select_task_grasp_indices(
                            task["translation_world"][row],
                            task["rotation_matrix"][row],
                            task["width_m"][row],
                            task["task_valid_probability"][row],
                            valid[row],
                        )
                        if len(selected):
                            stageb_deployment_scores.append(
                                task["task_valid_probability"][row, selected].detach().cpu().numpy()
                            )
                            stageb_deployment_targets.append(
                                batch["stageb_label"][row, selected].detach().cpu().numpy()
                            )
                evaluator.update(batch, model_output, terms)
                score = sum(
                    weight * float(terms[f"loss_{family}"])
                    for family, weight in config.training.validation_family_weights.items()
                    if f"loss_{family}" in terms
                )
                # 验证分数按真实 state-group 数加权，最后一个不满 batch 不得与完整 batch 等权。
                groups = int(batch["xyz"].shape[0])
                total += score * groups
                count += groups
                for key, value in terms.items():
                    if key in Trainer.COUNT_TERMS:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                        metric_counts[key] = metric_counts.get(key, 0) + 1
                    else:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * groups
                        metric_counts[key] = metric_counts.get(key, 0) + groups
                if rank == 0 and (
                    validation_step == 1
                    or validation_step % config.logging.validation_log_interval == 0
                    or validation_step == validation_batches
                ):
                    print(
                        f"[validation batch={validation_step:07d}/{validation_batches:07d}] "
                        f"groups={count}/{validation_groups} "
                        f"score={total / max(1, count):.6f}",
                        flush=True,
                    )
        module.train()
        result = {
            "score_sum": total,
            "score_count": count,
            "metric_sums": metric_sums,
            "metric_counts": metric_counts,
            "evaluation_records": evaluator.evaluator.records,
        }
        if stageb_scores:
            if not stageb_deployment_scores:
                raise RuntimeError("Stage-B validation has no deployment-selected candidates")
            result["stageb_scores"] = np.concatenate(stageb_scores)
            result["stageb_targets"] = np.concatenate(stageb_targets)
            result["stageb_deployment_scores"] = np.concatenate(stageb_deployment_scores)
            result["stageb_deployment_targets"] = np.concatenate(stageb_deployment_targets)
        return result

    if args.validate_only:
        validation = validate(trainer.ema.model if trainer.ema else trainer.model)
        validation_summaries = [validation]
        if torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, validation)
            validation_summaries = [item for item in gathered if item is not None]
        if rank == 0:
            metric_sums: dict[str, float] = {}
            metric_counts: dict[str, int] = {}
            for summary in validation_summaries:
                for key, value in summary.get("metric_sums", {}).items():
                    metric_sums[str(key)] = metric_sums.get(str(key), 0.0) + float(value)
                for key, value in summary.get("metric_counts", {}).items():
                    metric_counts[str(key)] = metric_counts.get(str(key), 0) + int(value)
            metric_details = {
                key: value
                if key in Trainer.COUNT_TERMS
                else value / max(1, metric_counts.get(key, 0))
                for key, value in metric_sums.items()
            }
            serializable = {
                "score_sum": float(sum(float(item["score_sum"]) for item in validation_summaries)),
                "score_count": int(sum(int(item["score_count"]) for item in validation_summaries)),
                "metric_sums": metric_sums,
                "metric_counts": metric_counts,
                "metrics": finalize_push_validation_metrics(metric_details),
            }
            if any("stageb_scores" in item for item in validation_summaries):
                serializable.update(
                    {
                        "stageb_candidates": int(
                            sum(len(item.get("stageb_scores", ())) for item in validation_summaries)
                        ),
                        "stageb_deployment_candidates": int(
                            sum(
                                len(item.get("stageb_deployment_scores", ()))
                                for item in validation_summaries
                            )
                        ),
                    }
                )
            output = os.path.join(config.output_dir, "validation_only.json")
            Path(output).write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(serializable, ensure_ascii=False), flush=True)
        return

    state = trainer.train(
        train_loader,
        validate=validate if validation_loader is not None else None,
        groups_per_effective_epoch=(
            len(train_dataset) if stageb else train_batch_sampler.global_samples_per_epoch
        ),
    )
    final_checkpoint = os.path.join(config.output_dir, "last.pt")
    trainer.save_checkpoint(final_checkpoint)
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        print(
            f"Saved final checkpoint: {os.path.abspath(final_checkpoint)} "
            f"(step {state.optimizer_steps:07d})",
            flush=True,
        )
    if world_size > 1:
        _destroy_process_group()
        atexit.unregister(_destroy_process_group)


def _destroy_process_group() -> None:
    """Best-effort DDP cleanup for both normal and exceptional exits."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
