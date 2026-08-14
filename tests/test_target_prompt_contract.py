from __future__ import annotations

import numpy as np

from tcd_prg.planners.tcd_policy import TCDPRGPolicy


def test_prompt_from_predicted_instance_is_observed_point():
    xyz = np.asarray([
        [0.0, 0.0, 0.1],
        [0.01, 0.0, 0.1],
        [0.4, 0.0, 0.1],
        [0.41, 0.0, 0.1],
    ], np.float32)
    instance = np.asarray([3, 3, 9, 9], np.int64)
    prompt = TCDPRGPolicy.target_prompt_from_instance(xyz, instance, 9)
    assert any(np.allclose(prompt, point) for point in xyz[instance == 9])


def test_prompt_from_instance_rejects_missing_query():
    xyz = np.asarray([[0.0, 0.0, 0.1]], np.float32)
    instance = np.asarray([3], np.int64)
    try:
        TCDPRGPolicy.target_prompt_from_instance(xyz, instance, 9)
    except RuntimeError as error:
        assert "no visible fused points" in str(error)
    else:
        raise AssertionError("missing query must fail closed")
