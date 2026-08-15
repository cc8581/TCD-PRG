import numpy as np
import pytest

from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.proposal_certification import (
    load_entry, object_local_key, save_entry, validate_entry,
)


def entry(count=3):
    unknown = np.full(count, int(CandidateStatus.UNKNOWN_UNTESTED), np.int8)
    return {
        "proposal_pose_object": np.zeros((count, 7), np.float32),
        "proposal_pose_world": np.zeros((count, 7), np.float32),
        "graspnet_width_m": np.zeros(count, np.float32),
        "graspnet_score": np.zeros(count, np.float32),
        "intrinsic_status": unknown.copy(), "intrinsic_valid": np.zeros(count, bool),
        "ag_width_target_m": np.zeros(count, np.float32), "ag_width_valid": np.zeros(count, bool),
        "contact_left_object": np.zeros((count, 3), np.float32),
        "contact_right_object": np.zeros((count, 3), np.float32), "contact_valid": np.zeros(count, bool),
        "force_closure_score": np.zeros(count, np.float32), "force_closure_valid": np.zeros(count, bool),
        "collision_free": np.zeros(count, bool), "collision_valid": np.zeros(count, bool),
        "approach_feasible": np.zeros(count, bool), "approach_valid": np.zeros(count, bool),
        "reachable": np.zeros(count, bool), "reachability_valid": np.zeros(count, bool),
        "scene_executable": np.zeros(count, bool), "scene_executable_valid": np.zeros(count, bool),
        "task_status": unknown.copy(),
    }


def test_unknown_is_not_implicitly_negative():
    values = entry()
    assert validate_entry(values) == 3
    assert np.all(values["intrinsic_status"] == int(CandidateStatus.UNKNOWN_UNTESTED))


def test_known_scene_requires_all_components():
    values = entry(1); values["scene_executable_valid"][0] = True
    with pytest.raises(ValueError, match="every scene component"):
        validate_entry(values)


def test_roundtrip_and_quaternion_sign_invariant_key(tmp_path):
    values = entry(1)
    path = save_entry(tmp_path / "entry.npz", values, {"scene": 3})
    loaded, metadata = load_entry(path)
    assert metadata == {"scene": 3}
    assert loaded["task_status"].tolist() == [-1]
    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert object_local_key("m", 0.01, pose) == object_local_key("m", 0.01, np.r_[pose[:3], -pose[3:]])
