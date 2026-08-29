"""Bounded training-only snapshots of RGB augmentation inputs and outputs."""

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
) -> None:
    """Save complete tensors, statistics, and a compact before/after preview."""

    rgb_after = _tensor(batch, "rgb")
    xyz = _tensor(batch, "xyz")
    point_mask = _tensor(batch, "point_mask").bool()
    instance_id = batch.get("instance_id")
    instance_array = (
        instance_id.detach().cpu().numpy()
        if isinstance(instance_id, Tensor)
        else np.empty((0,), dtype=np.int64)
    )
    before = rgb_before.detach().cpu()
    after = rgb_after.detach().cpu()
    xyz_cpu = xyz.detach().cpu()
    mask_cpu = point_mask.detach().cpu()
    valid = mask_cpu.unsqueeze(-1).expand_as(before)
    difference = (after - before).abs()
    valid_values = difference[valid]
    report = {
        "batch_size": int(len(after)),
        "point_counts": mask_cpu.sum(1).tolist(),
        "rgb_mean_absolute_difference": (
            float(valid_values.mean()) if valid_values.numel() else 0.0
        ),
        "rgb_max_absolute_difference": (
            float(valid_values.max()) if valid_values.numel() else 0.0
        ),
        "xyz_changed": False,
        "note": (
            "Training-only snapshot after collation; validation and inference "
            "bypass augmentation."
        ),
    }
    np.savez_compressed(
        directory / "batch_before_after.npz",
        xyz=xyz_cpu.numpy(),
        point_mask=mask_cpu.numpy(),
        instance_id=instance_array,
        rgb_before=before.numpy(),
        rgb_after=after.numpy(),
    )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_preview(directory / "preview.svg", xyz_cpu, before, after, mask_cpu)


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Augmentation debug requires tensor batch field {key!r}")
    return value


def _save_preview(
    path: Path,
    xyz: Tensor,
    before: Tensor,
    after: Tensor,
    point_mask: Tensor,
) -> None:
    samples = min(4, len(xyz))
    panel_width, panel_height = 320, 260
    width, height = panel_width * samples, panel_height * 2
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for row, colors in enumerate((before, after)):
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
