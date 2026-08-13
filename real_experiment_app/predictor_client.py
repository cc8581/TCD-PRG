from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

from .types import Prediction


class PredictorClient:
    """Persistent isolated model process for GUI/runtime stability."""

    def __init__(self, config_path: Path):
        self.temp = tempfile.TemporaryDirectory(prefix="tcd_prg_real_")
        self.request_path = Path(self.temp.name) / "scene.npz"
        command = [sys.executable, "-u", "-m", "real_experiment_app.predictor_worker",
                   "--config", str(config_path)]
        self.process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"Model worker failed to start: {ready}")

    def _read(self):
        assert self.process.stdout is not None
        diagnostics = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Model worker exited\n" + "".join(diagnostics[-20:]))
            try: return json.loads(line)
            except json.JSONDecodeError: diagnostics.append(line)

    def _call(self, command: str, **payload):
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command":command, **payload}, ensure_ascii=False)+"\n")
        self.process.stdin.flush(); response = self._read()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "model worker error"))
        return response.get("result")

    def predict(self, scene, target:int, category:int, region:int, required:int=1):
        np.savez_compressed(self.request_path, xyz_m=scene.xyz_m, rgb=scene.rgb,
            instance_id=scene.instance_id, source_view=scene.source_view,
            category_keys=np.asarray(list(scene.category_by_instance.keys()),np.int64),
            category_values=np.asarray(list(scene.category_by_instance.values()),np.int64))
        result = self._call("predict", scene=str(self.request_path), target=target,
                            category=category, region=region, required=required)
        return Prediction(result["action"], float(result["inference_seconds"]))

    def action_executed(self, action): self._call("action_executed", action=action)
    def reset(self): self._call("reset")

    def close(self):
        if self.process.poll() is None:
            try: self._call("close")
            finally: self.process.wait(timeout=10)
        self.temp.cleanup()

