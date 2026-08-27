"""Run the required Stage-B overfit and 64-group stability checks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tcd_prg.config import load_config
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.task_grasp import TaskGraspEvaluator


def inputs(batch: int, candidates: int, points: int, dim: int, device: torch.device):
    del dim
    xyz = torch.randn(batch, points, 3, device=device) * 0.025
    translation = torch.randn(batch, candidates, 3, device=device) * 0.02
    rotation = (
        torch.eye(3, device=device).reshape(1, 1, 3, 3).expand(batch, candidates, -1, -1).clone()
    )
    proposal = {
        "translation_world": translation,
        "rotation_matrix": rotation,
        "width_m": torch.rand(batch, candidates, device=device) * 0.075 + 0.02,
        "valid": torch.ones(batch, candidates, dtype=torch.bool, device=device),
    }
    return (
        proposal,
        xyz,
        torch.rand(batch, points, 3, device=device),
        torch.ones(batch, points, dtype=torch.bool, device=device),
        torch.rand(batch, points, device=device),
        torch.rand(batch, points, device=device),
        torch.zeros(batch, dtype=torch.long, device=device),
        torch.zeros(batch, dtype=torch.long, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/stage/grasp.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile-full-scale", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(16095)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = config.model.feature_dim
    model = TaskGraspEvaluator(dim, config.model.task_grasp_gripper_geometry).to(device)
    overfit = inputs(8, 1, 64, dim, device)
    positions = torch.linspace(-0.04, 0.04, 8, device=device)
    overfit[0]["translation_world"][:, 0, 0] = positions
    label = (positions > 0).float()[:, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = None
    for _step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logit = model(*overfit)["task_valid_logit"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, label)
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(*overfit)["task_valid_logit"] >= 0
    truth = label.bool()
    accuracy = float((prediction == truth).float().mean())
    tp = (prediction & truth).float().sum()
    fp = (prediction & ~truth).float().sum()
    fn = (~prediction & truth).float().sum()
    f1 = float(2 * tp / (2 * tp + fp + fn).clamp_min(1))

    stable = inputs(2, 32, 128, dim, device)
    model.zero_grad(set_to_none=True)
    stable_logit = model(*stable)["task_valid_logit"]
    stable_logit.square().mean().backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    report = {
        "device": str(device),
        "overfit_groups": 8,
        "overfit_steps": args.steps,
        "initial_loss": initial,
        "final_loss": float(loss.detach()),
        "accuracy": accuracy,
        "f1": f1,
        "stable_groups": 64,
        "stable_logits_finite": bool(torch.isfinite(stable_logit).all()),
        "stable_gradients_finite": gradients_finite,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / (1 << 20) if device.type == "cuda" else None
        ),
    }
    if args.profile_full_scale:
        model.zero_grad(set_to_none=True)
        profile = inputs(8, 32, 16384, dim, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            profile_logit = model(*profile)["task_valid_logit"]
            profile_loss = profile_logit.square().mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        profile_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        report.update(
            {
                "profile_shape": "B=8,K=32,N=16384",
                "profile_feature_dim": dim,
                "profile_forward_seconds": forward_seconds,
                "profile_backward_seconds": time.perf_counter() - started,
                "profile_peak_cuda_memory_mb": (
                    torch.cuda.max_memory_allocated() / (1 << 20) if device.type == "cuda" else None
                ),
                "profile_logits_finite": bool(torch.isfinite(profile_logit).all()),
                "profile_gradients_finite": all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ),
            }
        )
        del profile, profile_logit, profile_loss
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        full_model = TCDPRGModel(
            config.model, config.ablation, config.backbone, config.graspnet
        ).to(device)
        for name, parameter in full_model.named_parameters():
            parameter.requires_grad_(name.startswith("task_grasp."))
        scene_xyz = torch.randn(8, 16384, 3, device=device) * 0.15
        proposals = inputs(8, 32, 1, dim, device)[0]
        grid = torch.floor(
            (scene_xyz - scene_xyz.amin(1, keepdim=True)) / config.backbone.grid_size_m
        )
        full_batch = {
            "model_inputs": {
                "xyz": scene_xyz,
                "rgb": torch.rand_like(scene_xyz),
                "point_mask": torch.ones((8, 16384), dtype=torch.bool, device=device),
                "source_view": torch.zeros((8, 16384), dtype=torch.long, device=device),
                "grid_coord": grid.to(torch.int32),
            },
            "task_inputs": {
                "task_category_id": torch.zeros(8, dtype=torch.long, device=device),
                "task_region_id": torch.zeros(8, dtype=torch.long, device=device),
            },
            "target_mask": torch.ones((8, 16384), dtype=torch.bool, device=device),
            "region_target": torch.ones((8, 16384), dtype=torch.bool, device=device),
            "region_valid": torch.ones((8, 16384), dtype=torch.bool, device=device),
            "point_mask": torch.ones((8, 16384), dtype=torch.bool, device=device),
            "xyz": scene_xyz,
            "rgb": torch.rand_like(scene_xyz),
            "task_category_id": torch.zeros(8, dtype=torch.long, device=device),
            "task_region_id": torch.zeros(8, dtype=torch.long, device=device),
            "grasp_candidates": proposals,
        }
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            full_output = full_model.forward_grasp(full_batch)["task_grasp"]
            full_loss = full_output["task_valid_logit"].square().mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
        full_forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        full_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        report.update(
            {
                "full_stageb_forward_seconds": full_forward_seconds,
                "full_stageb_backward_seconds": time.perf_counter() - started,
                "full_stageb_peak_cuda_memory_mb": (
                    torch.cuda.max_memory_allocated() / (1 << 20) if device.type == "cuda" else None
                ),
                "full_stageb_logits_finite": bool(
                    torch.isfinite(full_output["task_valid_logit"]).all()
                ),
            }
        )
    if accuracy < 0.95 or f1 < 0.95 or not report["stable_logits_finite"] or not gradients_finite:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
