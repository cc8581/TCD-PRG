"""Process-isolated adapter for the unmodified GAPG reference implementation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tcd_prg.baselines.base import GlobalGraspPrediction
from tcd_prg.constants import ActionType
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.paths import project_path, resolve_executable

from .base import ManipulationPolicy


@dataclass(frozen=True, slots=True)
class GAPGPaths:
    """All external source and checkpoint paths required by the GAPG baseline."""

    repository: Path
    python: Path
    graspnet_baseline: Path
    graspnet_api: Path
    grasp_checkpoint: Path
    push_checkpoint: Path
    graspnet_checkpoint: Path
    worker: Path

    def validate(self) -> None:
        required = {
            "GAPG repository": self.repository,
            "Python executable": self.python,
            "graspnet-baseline checkout": self.graspnet_baseline,
            "graspnetAPI checkout": self.graspnet_api,
            "GAPG grasp checkpoint": self.grasp_checkpoint,
            "GAPG push checkpoint": self.push_checkpoint,
            "GraspNet checkpoint": self.graspnet_checkpoint,
            "GAPG worker": self.worker,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "GAPG baseline dependencies are incomplete:\n  - " + "\n  - ".join(missing)
                + "\nRun scripts/setup_third_party.ps1 and provide the three checkpoints."
            )


class GAPGPolicyWrapper(ManipulationPolicy):
    """Run original GAPG in its Python 3.8 environment through a stable IPC contract.

    The wrapper maps the externally supplied instance mask to GAPG's binary target
    labels.  Instance identifiers are only compared for equality and never supplied
    to a learned layer as ordered numeric values.
    """

    def __init__(
        self,
        repository: str | Path,
        grasp_checkpoint: str | Path,
        push_checkpoint: str | Path,
        graspnet_checkpoint: str | Path,
        *,
        python: str | Path = "python",
        graspnet_baseline: str | Path = ".deps/graspnet-baseline",
        graspnet_api: str | Path = ".deps/graspnetAPI",
        worker: str | Path = "scripts/run_gapg_baseline_worker_py38.py",
        seed: int = 2026,
        device: str = "cuda",
    ) -> None:
        root = project_path(repository)

        def rooted(value: str | Path) -> Path:
            path = Path(value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        def project_dependency(value: str | Path) -> Path:
            path = Path(value)
            return path.resolve() if path.is_absolute() else project_path(path)

        self.paths = GAPGPaths(
            repository=root,
            python=resolve_executable(python, must_exist=False),
            graspnet_baseline=project_dependency(graspnet_baseline),
            graspnet_api=project_dependency(graspnet_api),
            grasp_checkpoint=rooted(grasp_checkpoint),
            push_checkpoint=rooted(push_checkpoint),
            graspnet_checkpoint=rooted(graspnet_checkpoint),
            worker=project_dependency(worker),
        )
        self.seed = int(seed)
        self.device = device
        self._encoded: SceneObservation | None = None

    def encode_observation(self, observation: SceneObservation) -> SceneObservation:
        observation.validate()
        if any(camera.sensor_type.lower() == "oracle" for camera in observation.camera_parameters):
            raise ValueError("Oracle cameras are forbidden for the formal GAPG baseline")
        self._encoded = observation
        return observation

    def _run(self, observation: SceneObservation, mode: str = "policy") -> dict[str, Any]:
        self.paths.validate()
        with tempfile.TemporaryDirectory(prefix="tcd_prg_gapg_") as directory:
            temporary = Path(directory)
            input_path, output_path = temporary / "observation.npz", temporary / "result.json"
            np.savez_compressed(
                input_path,
                xyz=observation.xyz.astype(np.float32),
                rgb=observation.rgb.astype(np.float32),
                instance_id=observation.instance_id.astype(np.int32),
                point_valid=(observation.point_valid if observation.point_valid is not None
                             else np.ones(len(observation.xyz), dtype=bool)),
                object_active=observation.object_active.astype(bool),
                target_object=np.asarray(observation.target_object, dtype=np.int32),
            )
            command = [
                str(self.paths.python), str(self.paths.worker),
                "--input", str(input_path), "--output", str(output_path),
                "--gapg-root", str(self.paths.repository),
                "--graspnet-root", str(self.paths.graspnet_baseline),
                "--graspnet-api-root", str(self.paths.graspnet_api),
                "--grasp-checkpoint", str(self.paths.grasp_checkpoint),
                "--push-checkpoint", str(self.paths.push_checkpoint),
                "--graspnet-checkpoint", str(self.paths.graspnet_checkpoint),
                "--seed", str(self.seed), "--device", self.device,
                "--mode", mode,
            ]
            completed = subprocess.run(
                command, cwd=self.paths.repository, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "GAPG worker failed.\nstdout:\n" + completed.stdout
                    + "\nstderr:\n" + completed.stderr
                )
            if not output_path.exists():
                raise RuntimeError("GAPG worker succeeded without writing its result")
            return json.loads(output_path.read_text(encoding="utf-8"))

    def generate_candidates(self, encoded: SceneObservation) -> dict[str, Any]:
        return self._run(encoded)

    def select_action(self, candidates: dict[str, Any]) -> dict[str, Any] | None:
        action = candidates.get("selected_action")
        return action if isinstance(action, dict) else None

    def predict_grasps(self, encoded: SceneObservation) -> list[dict[str, Any]]:
        result = self._run(encoded)
        return [item for item in result.get("candidates", [])
                if int(item["action_type"]) == int(ActionType.TASK_GRASP)]

    def predict_global_grasps(
        self, encoded: SceneObservation, track: str = "scene_only"
    ) -> list[GlobalGraspPrediction]:
        if track not in {"scene_only", "instance_assisted"}:
            raise ValueError(f"Unknown global grasp track: {track}")
        mode = "global_scene" if track == "scene_only" else "global_instance"
        result = self._run(encoded, mode=mode)
        predictions = []
        for item in result.get("candidates", []):
            pose = np.asarray(item["grasp_pose_world"], np.float32)
            predictions.append(GlobalGraspPrediction(
                object_index=int(item["acted_object"]), contact_point_world=pose[:3].copy(),
                grasp_pose_world=pose, width_m=float(item["grasp_width_m"]),
                raw_score=float(item["score"]), scene_score=float(item["score"]),
                intrinsic_score=None, certified=False,
                source="gapg_global",
            ))
        return predictions

    def reset(self) -> None:
        self._encoded = None

    def update_after_action(self, action: Any, observation: SceneObservation) -> None:
        self._encoded = observation
