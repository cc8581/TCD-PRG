"""Thin adapter around the pinned official graspnet-baseline implementation.

No GraspNet layer is reimplemented. The official network is loaded lazily from the
pinned external dependency and deliberately kept outside the TCD-PRG state_dict:
it is frozen, reproducible from its own checkpoint, and should not be duplicated by
optimizer/EMA/checkpoint machinery.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.paths import project_path


def _load_official_graspnet(source_root: str | Path):
    root = Path(source_root)
    if not root.is_absolute():
        root = project_path(root)
    model_file = root / "models" / "graspnet.py"
    if not model_file.is_file():
        raise RuntimeError(
            f"Official graspnet-baseline is missing at {model_file}. "
            "Install the pinned dependency from third_party.lock.yaml first."
        )
    for candidate in (root, root / "models", root / "utils", root / "pointnet2"):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)
    module = importlib.import_module("graspnet")
    return module.GraspNet, module.pred_decode, module.get_loss


def _safe_logit(probability: Tensor, eps: float = 1e-5) -> Tensor:
    # FP16 cannot represent 1-eps for eps=1e-5 and rounds it back to 1,
    # producing +inf logits for high-confidence frozen proposals.
    probability = probability.float().clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class FrozenGraspNetProposalGenerator(nn.Module):
    """Official GraspNet inference plus opt-in, dense-label fine-tuning support.

    The historical class name is retained for checkpoint/API compatibility.  In
    fine-tuning mode the external network remains outside the TCD-PRG module
    tree, so it requires its own optimizer and checkpoint.
    """

    def __init__(
        self,
        source_root: str,
        checkpoint: str,
        proposal_count: int = 128,
        input_points: int = 20000,
        freeze: bool = True,
        num_view: int = 300,
        num_angle: int = 12,
        num_depth: int = 4,
        cylinder_radius: float = 0.05,
        hmin: float = -0.02,
        hmax_list: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
    ) -> None:
        super().__init__()
        self.source_root = str(source_root)
        self.checkpoint = str(checkpoint)
        self.proposal_count = int(proposal_count)
        self.input_points = int(input_points)
        self.freeze = bool(freeze)
        self.num_view = int(num_view)
        self.num_angle = int(num_angle)
        self.num_depth = int(num_depth)
        self.cylinder_radius = float(cylinder_radius)
        self.hmin = float(hmin)
        self.hmax_list = tuple(float(v) for v in hmax_list)
        self.pred_decode = None
        self.official_get_loss = None
        # Unregistered on purpose: frozen external model must not enter TCD-PRG
        # optimizer, EMA, DDP parameter broadcasts or checkpoints.
        object.__setattr__(self, "_network", None)
        object.__setattr__(self, "_network_device", None)

    @property
    def network(self) -> nn.Module | None:
        return object.__getattribute__(self, "_network")

    def _ensure_loaded(self, device: torch.device) -> nn.Module:
        network = self.network
        current = object.__getattribute__(self, "_network_device")
        if network is None:
            official, pred_decode, get_loss = _load_official_graspnet(self.source_root)
            network = official(
                input_feature_dim=0,
                num_view=self.num_view,
                num_angle=self.num_angle,
                num_depth=self.num_depth,
                cylinder_radius=self.cylinder_radius,
                hmin=self.hmin,
                hmax_list=list(self.hmax_list),
                is_training=False,
            )
            checkpoint_path = Path(self.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_path(checkpoint_path)
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    f"GraspNet checkpoint not found: {checkpoint_path}. "
                    "Set graspnet.checkpoint to the official pretrained checkpoint."
                )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload.get("model", payload))
            missing, unexpected = network.load_state_dict(state, strict=False)
            model_keys = len(network.state_dict())
            matched = model_keys - len(missing)
            if matched / max(1, model_keys) < 0.95:
                raise RuntimeError(
                    "GraspNet checkpoint is incompatible with the pinned official source: "
                    f"matched={matched}/{model_keys}, missing={len(missing)}, "
                    f"unexpected={len(unexpected)}"
                )
            network.requires_grad_(not self.freeze).eval()
            self.pred_decode = pred_decode
            self.official_get_loss = get_loss
            object.__setattr__(self, "_network", network)
            current = None
        if current != device:
            network.to(device)
            object.__setattr__(self, "_network_device", device)
        network.eval()
        return network

    @staticmethod
    def _set_official_training_mode(network: nn.Module, enabled: bool) -> None:
        network.is_training = bool(enabled)
        if hasattr(network, "grasp_generator"):
            network.grasp_generator.is_training = bool(enabled)
        network.train(enabled)

    def prepare_finetuning(self, device: torch.device | str) -> nn.Module:
        """Expose the external network for a separate dense-label optimizer."""

        if self.freeze:
            raise RuntimeError("Set graspnet.freeze=false to enable fine-tuning")
        network = self._ensure_loaded(torch.device(device))
        network.requires_grad_(True)
        self._set_official_training_mode(network, True)
        return network

    def finish_finetuning(self) -> None:
        """Restore the official network's label-free inference protocol."""

        if self.network is not None:
            self._set_official_training_mode(self.network, False)

    def finetune_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return parameters only after ``prepare_finetuning`` initialized them."""

        if self.freeze:
            return ()
        network = self.network
        if network is None:
            raise RuntimeError("Call prepare_finetuning(device) before creating the optimizer")
        return tuple(parameter for parameter in network.parameters() if parameter.requires_grad)

    def official_training_loss(
        self, dense_batch: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute the upstream GraspNet loss from its complete dense label schema.

        Sparse TCD task-grasp poses are intentionally rejected: upstream training
        requires per-object grasp points, offsets, scores and tolerances.
        """

        if "grasp_known_mask_list" in dense_batch:
            raise RuntimeError(
                "The upstream GraspNet loss has no UNKNOWN state and would treat "
                "uncovered lattice cells as negatives. Tri-state proposal labels "
                "must use proposal-level masked losses, not official_training_loss."
            )
        required = {
            "point_clouds",
            "objectness_label",
            "object_poses_list",
            "grasp_points_list",
            "grasp_offsets_list",
            "grasp_labels_list",
            "grasp_tolerance_list",
        }
        missing = required - dense_batch.keys()
        if missing:
            raise KeyError(
                "GraspNet fine-tuning requires dense upstream labels; missing "
                + ", ".join(sorted(missing))
            )
        point_clouds = dense_batch["point_clouds"]
        network = self.prepare_finetuning(point_clouds.device)
        if self.official_get_loss is None:
            raise RuntimeError("Official GraspNet loss was not loaded")
        try:
            end_points = network(dense_batch)
            loss, end_points = self.official_get_loss(end_points)
        finally:
            self.finish_finetuning()
        return loss, end_points

    def finetune_state_dict(self) -> dict[str, Tensor]:
        """Return a standalone GraspNet checkpoint payload."""

        if self.network is None:
            raise RuntimeError("GraspNet network has not been initialized")
        return self.network.state_dict()

    def load_finetune_state_dict(
        self,
        state_dict: dict[str, Tensor],
        device: torch.device | str,
        *,
        strict: bool = True,
    ) -> Any:
        """Load standalone fine-tuned weights without entering TCD checkpoints."""

        network = self._ensure_loaded(torch.device(device))
        return network.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.finish_finetuning()
        return self

    @staticmethod
    def _sample_indices(
        point_mask: Tensor,
        importance: Tensor | None,
        count: int,
    ) -> tuple[Tensor, Tensor]:
        b, _ = point_mask.shape
        output = torch.zeros((b, count), dtype=torch.long, device=point_mask.device)
        sample_valid = torch.zeros((b,), dtype=torch.bool, device=point_mask.device)
        for row in range(b):
            valid = torch.nonzero(point_mask[row], as_tuple=False).flatten()
            if importance is not None:
                score = importance[row, valid]
                positive = valid[score > 0.05]
                if len(positive):
                    valid = positive
                    score = importance[row, valid]
                    valid = valid[score.argsort(descending=True, stable=True)]
            if not len(valid):
                continue
            sample_valid[row] = True
            if len(valid) >= count:
                position = torch.linspace(
                    0, len(valid) - 1, count, device=valid.device
                ).round().long()
                selected = valid[position]
            else:
                selected = valid.repeat(math.ceil(count / len(valid)))[:count]
            output[row] = selected
        return output, sample_valid

    @staticmethod
    def _nearest_scene_point(
        translation: Tensor, xyz: Tensor, point_mask: Tensor, valid: Tensor
    ) -> Tensor:
        b, k, _ = translation.shape
        result = torch.zeros((b, k), dtype=torch.long, device=xyz.device)
        for row in range(b):
            points = torch.nonzero(point_mask[row], as_tuple=False).flatten()
            candidates = torch.nonzero(valid[row], as_tuple=False).flatten()
            if not len(points) or not len(candidates):
                continue
            distance = torch.cdist(
                translation[row, candidates].float(), xyz[row, points].float()
            )
            result[row, candidates] = points[distance.argmin(-1)]
        return result

    def forward(
        self,
        xyz: Tensor,
        point_mask: Tensor,
        *,
        importance: Tensor | None = None,
        instance_probability: Tensor | None = None,
        proposal_count: int | None = None,
        input_points: int | None = None,
    ) -> dict[str, Tensor]:
        network = self._ensure_loaded(xyz.device)
        self.finish_finetuning()
        requested_points = int(input_points or self.input_points)
        requested_proposals = int(proposal_count or self.proposal_count)
        sample_index, row_valid = self._sample_indices(
            point_mask.bool(), importance, requested_points
        )
        rows = torch.arange(xyz.shape[0], device=xyz.device)[:, None]
        sampled = xyz[rows, sample_index]
        # The pinned PointNet++ CUDA kernels only accept FP32 tensors.  Formal
        # training wraps the whole objective in autocast, so isolate the frozen
        # external network from AMP instead of disabling AMP for learned heads.
        with torch.inference_mode(), torch.autocast(
            device_type=xyz.device.type, enabled=False
        ):
            decoded = self.pred_decode(
                network({"point_clouds": sampled.float()})
            )

        b, k = xyz.shape[0], requested_proposals
        dtype = xyz.dtype
        translation = xyz.new_zeros((b, k, 3))
        rotation = xyz.new_zeros((b, k, 3, 3))
        width = xyz.new_zeros((b, k))
        depth = xyz.new_zeros((b, k))
        score = xyz.new_zeros((b, k))
        valid = torch.zeros((b, k), dtype=torch.bool, device=xyz.device)

        for row, prediction in enumerate(decoded):
            if not row_valid[row] or prediction.numel() == 0:
                continue
            prediction = prediction.to(device=xyz.device, dtype=dtype)
            ranked = prediction[:, 0].argsort(descending=True, stable=True)[:k]
            selected = prediction[ranked]
            count = len(selected)
            score[row, :count] = selected[:, 0].clamp(0.0, 1.0)
            width[row, :count] = selected[:, 1]
            depth[row, :count] = selected[:, 3]
            rotation[row, :count] = selected[:, 4:13].reshape(-1, 3, 3)
            translation[row, :count] = selected[:, 13:16]
            valid[row, :count] = torch.isfinite(selected[:, :16]).all(-1)

        attention_point_index = self._nearest_scene_point(
            translation, xyz, point_mask.bool(), valid
        )
        if instance_probability is None:
            object_logits = xyz.new_zeros((b, k, 1))
        else:
            row_index = torch.arange(b, device=xyz.device)[:, None]
            membership = instance_probability[row_index, :, attention_point_index]
            object_logits = membership.clamp_min(1e-6).log()

        return {
            "translation_world": translation,
            "rotation_matrix": rotation,
            "width_m": width,
            "depth_m": depth,
            "quality_logit": _safe_logit(score),
            "graspnet_score": score,
            "attention_point_index": attention_point_index,
            "object_logits": object_logits,
            "valid": valid,
        }
