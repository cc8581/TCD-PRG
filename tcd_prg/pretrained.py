"""Download and partially load compatible PTv3 pre-trained weights."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import BackboneConfig
from .paths import project_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pretrained_checkpoint(
    config: BackboneConfig, *, allow_download: bool
) -> Path | None:
    """Resolve, download and verify the configured backbone checkpoint."""

    explicit = config.pretrained_checkpoint
    if explicit:
        path = project_path(explicit)
    elif config.pretrained_auto_download:
        filename = Path(config.pretrained_url.split("?", 1)[0]).name
        if not filename:
            raise ValueError("backbone.pretrained_url has no filename")
        path = project_path(config.pretrained_cache_dir) / filename
    else:
        if config.pretrained_required:
            raise FileNotFoundError(
                "A pre-trained backbone is required, but neither "
                "backbone.pretrained_checkpoint nor auto-download is enabled"
            )
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Pre-trained checkpoint is not ready on this rank: {path}"
            )
        if not config.pretrained_url:
            raise FileNotFoundError(path)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        request = urllib.request.Request(
            config.pretrained_url,
            headers={"User-Agent": "TCD-PRG-pretrained-downloader/1.0"},
        )
        print(f"[pretrained] downloading {config.pretrained_url} -> {path}", flush=True)
        try:
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    output.write(block)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    expected = config.pretrained_sha256.strip().lower()
    if expected:
        actual = _sha256(path)
        if actual != expected:
            # An explicit checkpoint belongs to the user. Only discard a bad
            # file created in the managed auto-download cache.
            if not explicit:
                path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Pre-trained checkpoint SHA-256 mismatch: expected={expected}, actual={actual}"
            )
    return path


def _extract_tensor_mapping(value: Any) -> Mapping[str, Tensor]:
    if isinstance(value, Mapping):
        if any(torch.is_tensor(item) for item in value.values()):
            return value  # type: ignore[return-value]
        for key in (
            "state_dict", "model", "ema", "student", "teacher", "module", "backbone",
        ):
            if key in value:
                try:
                    return _extract_tensor_mapping(value[key])
                except (TypeError, ValueError):
                    pass
    raise ValueError("Checkpoint contains no tensor state_dict")


def _key_variants(raw: str) -> tuple[str, ...]:
    prefixes = (
        "module.", "model.", "state_dict.", "student.", "teacher.",
        "backbone.", "encoder.", "scene_backbone.",
    )
    variants = [raw]
    current = raw
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                variants.append(current)
                changed = True
                break
    for token in (".backbone.", ".scene_backbone."):
        if token in raw:
            variants.append(raw.split(token, 1)[1])
    return tuple(dict.fromkeys(variants))


def _shape_matched_state(
    state: Mapping[str, Tensor], target: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, tuple[tuple[int, ...], tuple[int, ...]]]]:
    matched: dict[str, Tensor] = {}
    mismatched: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        for key in _key_variants(str(raw_key)):
            if key not in target:
                continue
            if tuple(value.shape) == tuple(target[key].shape):
                matched.setdefault(key, value)
            else:
                mismatched[key] = (tuple(value.shape), tuple(target[key].shape))
            break
    return matched, mismatched


def load_pretrained_backbone(
    model: nn.Module, checkpoint_path: Path, config: BackboneConfig
) -> dict[str, Any]:
    """Load a TCD-PRG encoder checkpoint or a Sonata/PTv3 encoder checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_tensor_mapping(checkpoint)
    checkpoint_format = config.pretrained_format.lower()
    if checkpoint_format == "auto":
        checkpoint_format = (
            "tcd_prg" if any(str(key).startswith("encoder.") for key in state) else "sonata"
        )

    if checkpoint_format == "tcd_prg":
        target_module = model.encoder
        target_state = target_module.state_dict()
        normalized = {
            str(key).removeprefix("encoder."): value
            for key, value in state.items()
            if str(key).startswith("encoder.") and torch.is_tensor(value)
        }
        matched, mismatched = _shape_matched_state(normalized, target_state)
    elif checkpoint_format in {"sonata", "pointcept", "ptv3"}:
        scene_backbone = getattr(model.encoder, "scene_backbone", None)
        target_module = getattr(scene_backbone, "backbone", None)
        if target_module is None:
            raise TypeError(
                "Sonata/PTv3 loading requires the official point_transformer_v3 backbone"
            )
        target_state = target_module.state_dict()
        matched, mismatched = _shape_matched_state(state, target_state)
    else:
        raise ValueError(
            "backbone.pretrained_format must be auto, tcd_prg, sonata, pointcept, or ptv3"
        )

    if not matched:
        raise RuntimeError("No checkpoint tensor matches the configured backbone")
    matched_numel = sum(int(value.numel()) for value in matched.values())
    target_numel = sum(int(value.numel()) for value in target_state.values())
    fraction = matched_numel / max(1, target_numel)
    if fraction < config.pretrained_min_parameter_fraction:
        raise RuntimeError(
            "Compatible pre-trained parameter coverage is too low: "
            f"{fraction:.1%} < {config.pretrained_min_parameter_fraction:.1%}"
        )
    missing, unexpected = target_module.load_state_dict(matched, strict=False)
    if checkpoint_format == "tcd_prg":
        matched_parameter_names = [f"encoder.{key}" for key in matched]
    else:
        matched_parameter_names = [
            f"encoder.scene_backbone.backbone.{key}" for key in matched
        ]
    # Shape matching can restore only part of a module. Freeze exactly those
    # tensors so randomly initialized siblings continue learning immediately.
    freeze_prefixes = tuple(sorted(matched_parameter_names))
    return {
        "path": str(checkpoint_path.resolve()),
        "format": checkpoint_format,
        "matched_tensors": len(matched),
        "matched_parameter_fraction": fraction,
        "matched_parameter_names": sorted(matched_parameter_names),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "shape_mismatches": {
            key: {"checkpoint": source, "model": target}
            for key, (source, target) in sorted(mismatched.items())
        },
        "freeze_prefixes": list(freeze_prefixes),
    }
