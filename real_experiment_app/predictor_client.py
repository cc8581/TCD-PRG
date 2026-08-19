from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .types import Prediction


class PredictorClient:
    """Persistent isolated TCD-PRG process for perception and manipulation."""

    def __init__(self, config_path: Path):
        self.temp = tempfile.TemporaryDirectory(prefix="tcd_prg_real_")
        self.request_path = Path(self.temp.name) / "scene.npz"
        command = [
            sys.executable,
            "-u",
            "-m",
            "real_experiment_app.predictor_worker",
            "--config",
            str(config_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"Model worker failed to start: {ready}")

    def _read(self):
        assert self.process.stdout is not None
        diagnostics = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "Model worker exited\n" + "".join(diagnostics[-20:])
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                diagnostics.append(line)

    def _call(self, command: str, **payload):
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps({"command": command, **payload}, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()
        response = self._read()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "model worker error"))
        return response.get("result")

    def _save_scene(self, scene) -> None:
        np.savez_compressed(
            self.request_path,
            xyz_m=scene.xyz_m,
            rgb=scene.rgb,
            instance_id=scene.instance_id,
            source_view=scene.source_view,
            camera_to_world=(
                np.stack(scene.camera_to_world).astype(np.float32)
                if scene.camera_to_world
                else np.empty((0, 4, 4), np.float32)
            ),
            category_keys=np.asarray(
                list(scene.category_by_instance.keys()), np.int64
            ),
            category_values=np.asarray(
                list(scene.category_by_instance.values()), np.int64
            ),
        )

    def perceive(self, scene):
        self._save_scene(scene)
        result = self._call("perceive", scene=str(self.request_path))
        scene.instance_id = np.asarray(result["instance_id"], np.int64)
        scene.category_by_instance = {
            int(key): int(value)
            for key, value in result["category_by_instance"].items()
        }
        return scene

    def predict(
        self,
        scene,
        target: int | None,
        category: int,
        region: int,
    ):
        self._save_scene(scene)
        result = self._call(
            "predict",
            scene=str(self.request_path),
            target=(-1 if target is None else int(target)),
            category=category,
            region=region,
        )
        return Prediction(
            result["action"], float(result["inference_seconds"])
        )

    def action_executed(self, action):
        self._call("action_executed", action=action)

    def reset(self):
        self._call("reset")

    def close(self):
        if self.process.poll() is None:
            try:
                self._call("close")
            finally:
                self.process.wait(timeout=10)
        self.temp.cleanup()
