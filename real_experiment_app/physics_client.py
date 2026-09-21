"""Subprocess boundary for PyBullet, which lives in the GAPG environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from .worker_process import stop_worker_process


class PhysicsClient:
    def __init__(self, app_config):
        self.settings = dict(app_config.raw["physics"])
        self.settings["table_plane_base"] = dict(
            app_config.raw["fusion"]["table_plane_base"]
        )
        self.temp = tempfile.TemporaryDirectory(prefix="tcd_prg_physics_")
        self.scene_path = Path(self.temp.name) / "scene.npz"
        command = [
            str(self.settings["python"]), "-u", "-m", "real_experiment_app.physics_worker"
        ]
        self.process = None
        try:
            self.process = subprocess.Popen(
                command, cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if not self._read().get("ready"):
                raise RuntimeError("PyBullet worker failed to start")
        except Exception:
            self.close()
            raise

    def _read(self):
        diagnostics = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("PyBullet worker exited\n" + "".join(diagnostics[-20:]))
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                diagnostics.append(line)

    def _call(self, command, scene, candidates, target):
        np.savez_compressed(self.scene_path, xyz_m=scene.xyz_m, instance_id=scene.instance_id)
        payload = {"command": command, "scene": str(self.scene_path), "candidates": candidates,
                   "target": int(target), "settings": self.settings}
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._read)
            try:
                response = future.result(timeout=float(self.settings.get("request_timeout_s", 180)))
            except FutureTimeout as error:
                self.process.kill()
                raise TimeoutError("PyBullet仿真超时，工作进程已终止") from error
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "PyBullet error"))
        return response["result"]

    def collision_free_grasps(self, scene, candidates, target):
        results = self.assess_grasps(scene, candidates, target)
        by_index = {int(x["candidate_index"]): x for x in candidates}
        return [by_index[int(x["candidate_index"])] for x in results if x["collision_free"]]

    def assess_grasps(self, scene, candidates, target):
        """Return the final-pose collision result for every submitted grasp."""
        return self._call("grasp", scene, candidates, target)

    def simulate_pushes(self, scene, candidates, target):
        return self._call("push", scene, candidates, target)

    def close(self):
        try:
            stop_worker_process(self.process)
        finally:
            self.temp.cleanup()
