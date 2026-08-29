"""Bounded training-only snapshots of point-cloud augmentation inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def claim_debug_batch(output_dir: str | Path, limit: int) -> Path | None:
    """Atomically claim one of a run's globally bounded debug batch slots."""

    if limit <= 0:
        return None
    root = Path(output_dir) / "augmentation_debug"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(limit):
        candidate = root / f"batch_{index:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    return None


def save_debug_batch(
    directory: Path,
    batch: dict[str, Any],
    rgb_before: Tensor,
    xyz_before: Tensor | None = None,
    point_mask_before: Tensor | None = None,
) -> None:
    """Save complete tensors, statistics, and a compact before/after preview."""

    rgb_after = _tensor(batch, "rgb")
    xyz_after = _tensor(batch, "xyz")
    point_mask_after = _tensor(batch, "point_mask").bool()
    instance_id = batch.get("instance_id")
    instance_array = (
        instance_id.detach().cpu().numpy()
        if isinstance(instance_id, Tensor)
        else np.empty((0,), dtype=np.int64)
    )
    before = rgb_before.detach().cpu()
    after = rgb_after.detach().cpu()
    xyz_after_cpu = xyz_after.detach().cpu()
    mask_after_cpu = point_mask_after.detach().cpu()
    xyz_before_cpu = (
        xyz_before.detach().cpu() if xyz_before is not None else xyz_after_cpu.clone()
    )
    mask_before_cpu = (
        point_mask_before.detach().cpu().bool()
        if point_mask_before is not None
        else mask_after_cpu.clone()
    )
    valid = mask_after_cpu.unsqueeze(-1).expand_as(before)
    difference = (after - before).abs()
    valid_values = difference[valid]
    shared = mask_before_cpu & mask_after_cpu
    xyz_difference = torch.linalg.vector_norm(xyz_after_cpu - xyz_before_cpu, dim=-1)
    xyz_values = xyz_difference[shared]
    report = {
        "batch_size": int(len(after)),
        "point_counts_before": mask_before_cpu.sum(1).tolist(),
        "point_counts_after": mask_after_cpu.sum(1).tolist(),
        "rgb_mean_absolute_difference": (
            float(valid_values.mean()) if valid_values.numel() else 0.0
        ),
        "rgb_max_absolute_difference": (
            float(valid_values.max()) if valid_values.numel() else 0.0
        ),
        "xyz_mean_displacement_m": float(xyz_values.mean()) if xyz_values.numel() else 0.0,
        "xyz_max_displacement_m": float(xyz_values.max()) if xyz_values.numel() else 0.0,
        "removed_points": int((mask_before_cpu & ~mask_after_cpu).sum()),
        "xyz_changed": bool(
            (mask_before_cpu != mask_after_cpu).any()
            or (xyz_values.numel() and bool((xyz_values > 0).any()))
        ),
        "note": (
            "Training-only snapshot after collation; validation and inference "
            "bypass augmentation."
        ),
    }
    np.savez_compressed(
        directory / "batch_before_after.npz",
        xyz_before=xyz_before_cpu.numpy(),
        xyz_after=xyz_after_cpu.numpy(),
        xyz=xyz_after_cpu.numpy(),
        point_mask_before=mask_before_cpu.numpy(),
        point_mask_after=mask_after_cpu.numpy(),
        point_mask=mask_after_cpu.numpy(),
        instance_id=instance_array,
        rgb_before=before.numpy(),
        rgb_after=after.numpy(),
    )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_preview(
        directory / "preview.svg",
        xyz_before_cpu,
        xyz_after_cpu,
        before,
        after,
        mask_before_cpu,
        mask_after_cpu,
    )


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Augmentation debug requires tensor batch field {key!r}")
    return value


def _save_preview(
    path: Path,
    xyz_before: Tensor,
    xyz_after: Tensor,
    before: Tensor,
    after: Tensor,
    point_mask_before: Tensor,
    point_mask_after: Tensor,
) -> None:
    samples = min(4, len(xyz_after))
    panel_width, panel_height = 320, 260
    width, height = panel_width * samples, panel_height * 2
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for row, (xyz, colors, point_mask) in enumerate(
        (
            (xyz_before, before, point_mask_before),
            (xyz_after, after, point_mask_after),
        )
    ):
        for sample in range(samples):
            valid = torch.nonzero(point_mask[sample], as_tuple=False).flatten()
            if len(valid) > 1500:
                positions = torch.linspace(0, len(valid) - 1, 1500).long()
                valid = valid[positions]
            points = xyz[sample, valid].numpy()
            rgb = (colors[sample, valid].clamp(0, 1).numpy() * 255).astype(np.uint8)
            projected = np.column_stack(
                (points[:, 0] + 0.35 * points[:, 2], points[:, 1] + 0.20 * points[:, 2])
            )
            low, high = projected.min(0), projected.max(0)
            normalized = (projected - low) / np.maximum(high - low, 1e-8)
            origin_x, origin_y = sample * panel_width, row * panel_height
            elements.append(
                f'<text x="{origin_x + 10}" y="{origin_y + 20}" '
                f'font-family="sans-serif" font-size="14" fill="black">'
                f'sample {sample} | {"before" if row == 0 else "after"}</text>'
            )
            for point, color in zip(normalized, rgb, strict=True):
                x = origin_x + 15 + point[0] * (panel_width - 30)
                y = origin_y + 30 + (1 - point[1]) * (panel_height - 45)
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="0.8" '
                    f'fill="rgb({color[0]},{color[1]},{color[2]})"/>'
                )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")
