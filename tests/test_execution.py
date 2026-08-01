from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tcd_prg.constants import PUSH_DISTANCE_M, ActionType
from tcd_prg.datasets.types import SceneObservation
from tcd_prg.execution.pybullet_certifier import ExternalFR5AG16095Certifier


def test_float32_push_distance_is_semantically_exact_015m() -> None:
    stored = float(np.float32(PUSH_DISTANCE_M))
    assert stored != PUSH_DISTANCE_M
    assert np.isclose(stored, PUSH_DISTANCE_M, atol=1e-6, rtol=0.0)


def test_exact_certifier_excludes_inactive_objects_from_request(tmp_path, monkeypatch) -> None:
    scene_dir = tmp_path / "scene_0000"
    scene_dir.mkdir()
    np.savez(
        scene_dir / "scene.npz",
        thin_support_block_present=np.empty(0, bool),
        thin_support_block_pose=np.empty((0, 7), np.float32),
        thin_support_block_size=np.empty((0, 3), np.float32),
    )
    observation = SceneObservation(
        scene_id=0, state_id=1, task_index=0,
        xyz=np.zeros((1, 3), np.float32), rgb=np.zeros((1, 3), np.float32),
        instance_id=np.array([0], np.int64), target_mask=np.array([True]),
        target_object=0, task_region_id=0, object_uuid=("0", "1"),
        object_pose=np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], np.float32),
        object_category_id=np.array([0, 1], np.int64),
        object_present=np.array([True, True]), object_active=np.array([True, False]),
        camera_parameters=(),
        metadata={"object_model_id": ("a", "b"), "object_scale": (1.0, 1.0)},
    )
    certifier = object.__new__(ExternalFR5AG16095Certifier)
    certifier.python = tmp_path / "python"
    certifier.worker = tmp_path / "worker.py"
    certifier.robot_root = tmp_path / "robot"
    certifier.mesh_root = tmp_path / "meshes"
    certifier.scene_root = tmp_path
    certifier.temporary_root = tmp_path / "requests"
    certifier.temporary_root.mkdir()
    certifier.set_observation(observation)

    captured: dict[str, np.ndarray] = {}
    original_savez = np.savez_compressed

    def capture_request(path, *args, **kwargs):
        if "object_present" in kwargs:
            captured["object_present"] = np.asarray(kwargs["object_present"]).copy()
        return original_savez(path, *args, **kwargs)

    def fake_run(command, **kwargs):
        del kwargs
        output = command[command.index("--output") + 1]
        original_savez(
            output, success=np.array([True]), reasons=np.array(["ok"], dtype="U8")
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tcd_prg.execution.pybullet_certifier.np.savez_compressed", capture_request)
    monkeypatch.setattr("tcd_prg.execution.pybullet_certifier.subprocess.run", fake_run)
    result = certifier.certify({
        "action_type": int(ActionType.PICK_REMOVE), "acted_object": 0,
        "grasp_pose_world": np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
        "grasp_width_m": 0.05,
    })
    assert result == (True, "ok")
    assert captured["object_present"].tolist() == [True, False]
