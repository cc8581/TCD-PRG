"""Standard-only component metrics for training validation and offline evaluation.

This evaluator intentionally does not emit TCD-specific proxy/diagnostic metrics.
Tasks whose accepted protocol requires an external physical benchmark or executed
closed-loop trials are evaluated by dedicated entry points instead of surrogate
pose/candidate metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tcd_prg.config import EvaluationConfig, GraphConfig
from tcd_prg.losses.labels import build_verifier_labels

from .evaluator import Evaluator
from .metrics import binary_confusion
from .protocols import no_graph_constraint_relation_counts_at_k


class OfflineModelEvaluator:
    """Accumulate only benchmark-comparable metrics supported by current labels."""

    def __init__(
        self,
        model_config: Any,
        bootstrap_samples: int = 1_000,
        confidence: float = 0.95,
        graph_config: GraphConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.graph_config = graph_config or GraphConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.evaluator = Evaluator(
            bootstrap_samples, confidence, self.evaluation_config.calibration_bins
        )

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    @staticmethod
    def _probability(logit: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(logit.detach().float()).cpu().numpy()

    def update(
        self,
        batch: dict[str, Any],
        output: dict[str, Any],
        loss_terms: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # Losses are already aggregated by Trainer. They are deliberately not
        # inserted into the performance evaluator, whose output is standard-only.
        del loss_terms
        batch_size = batch["xyz"].shape[0]
        # Predicted object queries are unordered. Objective stores labels already
        # remapped through the Hungarian instance assignment.
        graph_labels = output.get("graph_labels_aligned")
        verifier_labels = (
            build_verifier_labels(batch) if output.get("verifier") is not None else None
        )

        for row in range(batch_size):
            sample = batch["samples"][row]
            record: dict[str, Any] = {
                "scene_id": sample.observation.scene_id,
                "state_id": sample.observation.state_id,
                "task_index": sample.observation.task_index,
                "category": int(
                    sample.observation.object_category_id[sample.observation.target_object]
                ),
                "task_region": sample.observation.task_region_id,
                "sequence_length": sample.state_labels.sequence_depth,
                "occlusion_level": int(
                    min(4, (1.0 - sample.state_labels.target_visible_ratio) * 5)
                ),
            }

            # Task-region segmentation: standard binary segmentation confusion.
            # Evaluator aggregates class IoUs globally and reports mIoU.
            if "region_target" in batch and "region" in output:
                valid = self._numpy(batch["region_valid"][row]).astype(bool)
                target = self._numpy(batch["region_target"][row]).astype(bool)
                probability = self._numpy(output["region"]["region_probability"][row])
                prediction = probability >= self.evaluation_config.region_probability_threshold
                record["_confusion_standard_region_foreground"] = binary_confusion(
                    prediction, target, valid
                )
                record["_confusion_standard_region_background"] = binary_confusion(
                    ~prediction, ~target, valid
                )

            # Dependency graph: PredCls-style no-graph-constraint relation retrieval.
            if output.get("graph") is not None and graph_labels is not None:
                physical_score = self._probability(
                    output["graph"].physical_edge_logits[row]
                )
                physical_target = self._numpy(
                    graph_labels["physical_edge_target"][row]
                ).astype(bool)
                physical_valid = self._numpy(
                    graph_labels["physical_edge_valid"][row]
                ).astype(bool)
                if (
                    physical_valid.ndim == 3
                    and physical_valid.shape[0] == physical_valid.shape[1]
                ):
                    physical_valid &= ~np.eye(
                        physical_valid.shape[0], dtype=bool
                    )[..., None]

                task_score = self._probability(output["graph"].task_edge_logits[row])
                task_target = self._numpy(
                    graph_labels["task_edge_target"][row]
                ).astype(bool)
                task_valid = self._numpy(
                    graph_labels["task_edge_valid"][row]
                ).astype(bool)

                for k in self.evaluation_config.relation_ranking_topk:
                    hits, total = no_graph_constraint_relation_counts_at_k(
                        physical_score, physical_target, physical_valid, k
                    )
                    record[
                        f"_relation_counts_standard_physical_relation_ng_at_{k}"
                    ] = (hits.tolist(), total.tolist())
                    if int(total.sum()) > 0:
                        record[
                            f"standard_physical_relation_ng_recall_at_{k}"
                        ] = float(hits.sum() / total.sum())

                    hits, total = no_graph_constraint_relation_counts_at_k(
                        task_score, task_target, task_valid, k
                    )
                    record[f"_relation_counts_standard_task_relation_ng_at_{k}"] = (
                        hits.tolist(), total.tolist()
                    )
                    if int(total.sum()) > 0:
                        record[f"standard_task_relation_ng_recall_at_{k}"] = float(
                            hits.sum() / total.sum()
                        )

            # Grasp verifier: candidate-level binary classification metrics.
            if output.get("verifier") is not None and verifier_labels is not None:
                valid = self._numpy(verifier_labels["overall_valid"][row]).astype(bool)
                if valid.any():
                    target = self._numpy(
                        verifier_labels["overall_target"][row]
                    ).astype(bool)
                    probability = self._probability(
                        output["verifier"]["overall_logit"][row]
                    )
                    record["_confusion_standard_verifier_overall"] = binary_confusion(
                        probability >= self.evaluation_config.verifier_probability_threshold,
                        target,
                        valid,
                    )
                    record["_binary_standard_verifier_overall"] = (
                        probability[valid],
                        target[valid],
                    )

            # Global grasp, task grasp, push and policy are intentionally absent:
            # their accepted protocols require GraspNet physical evaluation or
            # executed task/closed-loop trials. Dedicated standard evaluators
            # handle those protocols without surrogate offline metrics.
            self.evaluator.add(**record)

    def summarize(self) -> dict[str, Any]:
        return self.evaluator.summarize()

    def export(self, output_dir: str, config: dict[str, Any]) -> None:
        self.evaluator.export(output_dir, config)
