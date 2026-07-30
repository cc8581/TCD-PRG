"""External-Python observation rendering for the existing GAPG environment."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .base import ObservationProvider, ObservationRequest, PointObservation
from tcd_prg.paths import project_path, resolve_executable


class ExternalPyBulletObservationProvider(ObservationProvider):
    """Invoke the Python 3.8-compatible worker in an environment with PyBullet."""

    def __init__(
        self,
        python_executable: str | Path,
        worker_script: str | Path,
        scene_root: str | Path,
        runtime_mesh_root: str | Path,
        width: int = 320,
        height: int = 200,
        temporary_root: str | Path = "runtime/tmp/render_requests",
    ) -> None:
        self.python_executable = resolve_executable(python_executable)
        self.worker_script = project_path(worker_script)
        self.scene_root = project_path(scene_root)
        self.runtime_mesh_root = project_path(runtime_mesh_root)
        self.width = int(width)
        self.height = int(height)
        self.temporary_root = project_path(temporary_root)
        for path in (
            self.worker_script,
            self.scene_root,
            self.runtime_mesh_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    def get(self, request: ObservationRequest) -> PointObservation:
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            directory_path = Path(directory)
            request_path = directory_path / "request.npz"
            output_path = directory_path / "observation.npz"
            np.savez_compressed(
                request_path,
                scene_id=np.asarray(request.scene_id, dtype=np.int32),
                state_id=np.asarray(request.state_id, dtype=np.int32),
                object_pose=np.asarray(request.object_pose, dtype=np.float32),
                object_present=np.asarray(request.object_present, dtype=bool),
                object_active=np.asarray(request.object_active, dtype=bool),
                object_asset_ids=np.asarray(request.object_asset_ids),
                object_model_ids=np.asarray(request.object_model_ids),
                object_scales=np.asarray(request.object_scales, dtype=np.float32),
                render_seed=np.asarray(request.render_seed, dtype=np.int64),
                point_count=np.asarray(request.point_count, dtype=np.int32),
                renderer_version=np.asarray(request.renderer_version),
            )
            command = [
                str(self.python_executable),
                str(self.worker_script),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
                "--scene-root",
                str(self.scene_root),
                "--runtime-mesh-root",
                str(self.runtime_mesh_root),
                "--width",
                str(self.width),
                "--height",
                str(self.height),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    "External PyBullet renderer failed\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError("External renderer returned success without an observation file")
            with np.load(output_path, allow_pickle=False) as data:
                return PointObservation(
                    data["xyz"].astype(np.float32),
                    data["rgb"].astype(np.float32),
                    data["instance_id"].astype(np.int64),
                    data["source_view"].astype(np.int16),
                )
