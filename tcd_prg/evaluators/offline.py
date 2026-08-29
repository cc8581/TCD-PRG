"""Standard-only metrics for the minimal TCD-PRG training/evaluation path.

Task/global grasp and closed-loop manipulation remain evaluated by their dedicated
physical protocols. The offline evaluator reports target-instance and task-region
segmentation, which are directly comparable from native point labels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tcd_prg.config import EvaluationConfig

from .evaluator import Evaluator
from .metrics import binary_confusion


class OfflineModelEvaluator:
    def __init__(
        self,
        model_config: Any,
        bootstrap_samples: int = 1_000,
        confidence: float = 0.95,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.evaluator = Evaluator(
            bootstrap_samples, confidence, self.evaluation_config.calibration_bins
        )

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    def update(
        self,
        batch: dict[str, Any],
        output: dict[str, Any],
        loss_terms: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del loss_terms
        batch_size = batch["xyz"].shape[0]
        for row in range(batch_size):
            sample = batch["samples"][row]
            record: dict[str, Any] = {
                "scene_id": sample.observation.scene_id,
                "state_id": sample.observation.state_id,
                "task_index": sample.observation.task_index,
                "category": int(
                    sample.observation.object_category_id[
                        sample.observation.target_object
                    ]
                ),
                "task_region": sample.observation.task_region_id,
                "sequence_length": sample.state_labels.sequence_depth,
                "occlusion_level": int(
                    min(4, (1.0 - sample.state_labels.target_visible_ratio) * 5)
                ),
            }
            encoded = output.get("encoded")
            if "target_mask" in batch and encoded is not None:
                valid = self._numpy(batch["point_mask"][row]).astype(bool)
                target = self._numpy(batch["target_mask"][row]).astype(bool)
                probability = self._numpy(encoded.target_instance_probability[row])
                prediction = probability >= 0.5
                record["_confusion_standard_target"] = binary_confusion(
                    prediction, target, valid
                )
            if "region_target" in batch and output.get("region") is not None:
                valid = self._numpy(batch["region_valid"][row]).astype(bool)
                valid &= self._numpy(batch["point_mask"][row]).astype(bool)
                target = self._numpy(batch["region_target"][row]).astype(bool)
                probability = self._numpy(output["region"]["region_probability"][row])
                prediction = (
                    probability >= self.evaluation_config.region_probability_threshold
                )
                record["_confusion_standard_region_foreground"] = binary_confusion(
                    prediction, target, valid
                )
                record["_confusion_standard_region_background"] = binary_confusion(
                    ~prediction, ~target, valid
                )
            self.evaluator.add(**record)

    def summarize(self) -> dict[str, Any]:
        return self.evaluator.summarize()

    def export(self, output_dir: str, config: dict[str, Any]) -> None:
        self.evaluator.export(output_dir, config)
