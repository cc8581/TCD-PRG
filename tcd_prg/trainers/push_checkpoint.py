"""Atomic PUSH-only best weights and restartable optimizer snapshots."""
from pathlib import Path
import os
import math
import tempfile
import copy

import torch

from tcd_prg.models.staged_checkpoint import PUSH_EVALUATOR_PROTOCOL_VERSION, PUSH_ARCHITECTURE, validate_push_checkpoint

PUSH_METRIC_PROTOCOL_VERSION = 2


def resume_compatibility(signature):
    signature = copy.deepcopy(signature)
    config = signature.get("config", {})
    # Output location and terminal frequency do not change the training objective.
    config.pop("output_dir", None)
    config.pop("logging", None)
    return signature


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same directory/volume: a failed save never replaces the published checkpoint.
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PushTrainingCheckpoint:
    def __init__(self, output, model, metadata, resume_signature, scheduler=None):
        self.output = Path(output)
        self.latest = self.output.with_name(self.output.stem + "_last.pt")
        self.model = model
        self.metadata = metadata
        self.resume_signature = resume_signature
        self.scheduler = scheduler
        self.best_state = None
        self.best_metrics = None
        self.best_step = None

    def _payload(self, state, step):
        return {**self.metadata, "model": state, "optimizer_steps": step,
                "push_architecture": PUSH_ARCHITECTURE,
                "training_stage": "push_evaluator",
                "push_evaluator_protocol_version": PUSH_EVALUATOR_PROTOCOL_VERSION,
                "push_metric_protocol_version": PUSH_METRIC_PROTOCOL_VERSION,
                "resume_signature": self.resume_signature}

    def save_latest(self, optimizer, step):
        payload = self._payload(self.model.push_evaluator.state_dict(), step)
        payload.update(optimizer=optimizer.state_dict(),
                       best_state=self.best_state, best_metrics=self.best_metrics,
                       best_step=self.best_step)
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        atomic_save(payload, self.latest)

    def consider_best(self, metrics, step):
        score = metrics["push_evaluator_ap"]
        if not math.isfinite(score):
            return
        if self.best_metrics is not None and score <= self.best_metrics["push_evaluator_ap"]:
            return
        self.best_state = {k: v.detach().cpu().clone()
                           for k, v in self.model.push_evaluator.state_dict().items()}
        self.best_metrics, self.best_step = dict(metrics), step
        self.save_best()

    def save_best(self, final_metrics=None):
        if self.best_state is None:
            raise RuntimeError("PUSH evaluator validation did not produce a finite AP")
        payload = self._payload(self.best_state, self.best_step)
        payload["validation_metrics"] = self.best_metrics
        if final_metrics is not None:
            payload["final_validation_metrics"] = final_metrics
        atomic_save(payload, self.output)

    def restore(self, path, optimizer):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_push_checkpoint(self.model, payload)
        if ("optimizer" not in payload or payload.get("resume_signature") is None or
                resume_compatibility(payload["resume_signature"]) != resume_compatibility(self.resume_signature)):
            raise RuntimeError("PUSH resume requires a matching training configuration and a _last checkpoint")
        if payload.get("push_metric_protocol_version") != PUSH_METRIC_PROTOCOL_VERSION:
            raise RuntimeError("PUSH metric protocol mismatch")
        if self.scheduler is not None:
            if payload.get("scheduler") is None or payload["scheduler"]["last_epoch"] != int(payload["optimizer_steps"]):
                raise RuntimeError("PUSH scheduler/optimizer step mismatch")
        self.model.push_evaluator.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.best_state = payload["best_state"]
        self.best_metrics = payload["best_metrics"]
        self.best_step = payload["best_step"]
        # A crash can occur after publishing best but before refreshing latest.
        # Do not overwrite that newer best with the older snapshot's selection.
        if self.output.is_file():
            published = torch.load(self.output, map_location="cpu", weights_only=False)
            metrics = published.get("validation_metrics", {})
            score = metrics.get("push_evaluator_ap", float("nan"))
            compatible = all(published.get(key) == payload.get(key) for key in
                             ("training_stage", "push_evaluator_protocol_version",
                              "push_architecture"))
            compatible = compatible and resume_compatibility(published.get("resume_signature", {})) == resume_compatibility(self.resume_signature)
            compatible = compatible and published.get("push_metric_protocol_version") == PUSH_METRIC_PROTOCOL_VERSION
            if compatible and math.isfinite(score) and (
                    self.best_metrics is None or score > self.best_metrics["push_evaluator_ap"]):
                self.best_state = published["model"]
                self.best_metrics = metrics
                self.best_step = published["optimizer_steps"]
        if self.best_state is not None:
            self.save_best()
        # The loader starts a fresh shuffled epoch. This is optimizer continuation,
        # not bitwise replay of an interrupted multi-worker augmentation stream.
        return int(payload["optimizer_steps"])
