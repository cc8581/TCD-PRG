"""Host-side batch adapter for exact external FR5/AG action certification."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tcd_prg.constants import ActionType, PUSH_DISTANCE_M
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.paths import project_path, resolve_executable


class ExternalFR5AG16095Certifier:
    """Exact grasp-pose certifier; PUSH motion planning is executor-owned."""
    def __init__(self, python_executable: str | Path, worker_script: str | Path,
                 robot_root: str | Path, runtime_mesh_root: str | Path,
                 scene_root: str | Path,
                 temporary_root: str | Path = "runtime/tmp/certification") -> None:
        self.python = resolve_executable(python_executable); self.worker = project_path(worker_script)
        self.robot_root = project_path(robot_root); self.mesh_root = project_path(runtime_mesh_root)
        self.scene_root = project_path(scene_root); self.temporary_root = project_path(temporary_root)
        for path in (self.worker, self.robot_root, self.mesh_root, self.scene_root):
            if not path.exists():
                raise FileNotFoundError(path)
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self.observation: SceneObservation | None = None

    def set_observation(self, observation: SceneObservation) -> None:
        self.observation = observation

    def certify_many(self, actions: list[dict[str, Any]]) -> list[tuple[bool, str]]:
        if not actions:
            return []
        if self.observation is None:
            raise RuntimeError("set_observation must be called before certification")
        observation = self.observation
        action_type_all = np.asarray([item["action_type"] for item in actions], np.int8)
        distances = np.asarray(
            [item.get("push_distance_m", np.nan) for item in actions], dtype=np.float64
        )
        if np.any((action_type_all == int(ActionType.PUSH)) &
                  (~np.isclose(distances, PUSH_DISTANCE_M, atol=1e-6, rtol=0.0))):
            raise ValueError("Formal PUSH actions require exactly 0.15 m")
        grasp_indices = np.flatnonzero(action_type_all != int(ActionType.PUSH)).tolist()
        decisions: list[tuple[bool, str]] = [
            (False, "push_requires_executor_motion_planner")
            if kind == int(ActionType.PUSH) else (False, "unprocessed")
            for kind in action_type_all
        ]
        if not grasp_indices:
            return decisions
        grasp_actions = [actions[index] for index in grasp_indices]
        metadata = observation.metadata
        model_ids = np.asarray(metadata["object_model_id"])
        scales = np.asarray(metadata["object_scale"], np.float32)
        scene_file = self.scene_root / f"scene_{observation.scene_id:04d}" / "scene.npz"
        with np.load(scene_file, allow_pickle=False) as scene:
            support_present = scene["thin_support_block_present"]
            support_pose = scene["thin_support_block_pose"]
            support_size = scene["thin_support_block_size"]
        count = len(grasp_actions)
        action_type = np.asarray([item["action_type"] for item in grasp_actions], np.int8)
        pose = np.full((count, 7), np.nan, np.float32)
        width = np.full(count, np.nan, np.float32)
        contact = np.full((count, 3), np.nan, np.float32)
        direction = np.full((count, 3), np.nan, np.float32)
        for index, item in enumerate(grasp_actions):
            pose[index] = item["grasp_pose_world"]
            width[index] = item["grasp_width_m"]
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            request, output = Path(directory) / "request.npz", Path(directory) / "result.npz"
            np.savez_compressed(
                request, object_model_ids=model_ids, object_scales=scales,
                object_pose=observation.object_pose,
                object_present=observation.physical_active,
                support_present=support_present, support_pose=support_pose, support_size=support_size,
                action_type=action_type,
                acted_object=np.asarray(
                    [item["acted_object"] for item in grasp_actions], np.int16
                ),
                pose_world=pose, width_m=width, contact_world=contact,
                direction_world=direction,
            )
            completed = subprocess.run(
                [str(self.python), str(self.worker), "--request", str(request), "--output",
                 str(output), "--robot-root", str(self.robot_root),
                 "--runtime-mesh-root", str(self.mesh_root)],
                capture_output=True, text=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError("FR5 certification worker failed\n" + completed.stdout + completed.stderr)
            with np.load(output, allow_pickle=False) as result:
                grasp_decisions = list(zip(
                    result["success"].astype(bool).tolist(),
                    result["reasons"].astype(str).tolist(),
                ))
        for index, decision in zip(grasp_indices, grasp_decisions, strict=True):
            decisions[index] = decision
        return decisions

    def certify(self, action: dict[str, Any]) -> tuple[bool, str]:
        return self.certify_many([action])[0]
