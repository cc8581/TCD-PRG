"""Real subprocess/PyBullet integration checks (no mocked physics)."""

from pathlib import Path

import numpy as np
import pytest

from real_experiment_app.config import AppConfig
from real_experiment_app.physics_client import PhysicsClient
from real_experiment_app.types import FusedScene


def test_independent_pushes_use_real_worker_and_identical_initial_state():
    config = AppConfig.load(Path(__file__).parents[1] / "configs" / "real_experiment.yaml")
    if not Path(config.raw["physics"]["python"]).is_file():
        pytest.skip("configured PyBullet Python is unavailable")
    rng = np.random.default_rng(9)
    pushed = rng.uniform([0.35, -0.03, 0.02], [0.40, 0.03, 0.07], (120, 3))
    target = rng.uniform([0.43, -0.03, 0.02], [0.48, 0.03, 0.07], (120, 3))
    scene = FusedScene(
        np.vstack((pushed, target)),
        np.zeros((240, 3)),
        np.repeat([1, 2], 120),
        np.zeros(240, int),
        {},
    )
    base = {
        "acted_object": 1,
        "push_contact_world": [0.35, 0, 0.045],
        "push_direction_world": [-1, 0, 0],
        "push_distance_m": 0.15,
    }
    candidates = (dict(base, candidate_index=10), dict(base, candidate_index=11))
    client = PhysicsClient(config)
    try:
        results = client.simulate_pushes(scene, candidates, 2)
    finally:
        client.close()
    assert [item["candidate_index"] for item in results] == [10, 11]
    assert results[0]["effective"] == results[1]["effective"]
    assert np.isclose(results[0]["distance_gain_m"], results[1]["distance_gain_m"])
    assert np.isclose(results[0]["direction_projection_m"], results[1]["direction_projection_m"])
