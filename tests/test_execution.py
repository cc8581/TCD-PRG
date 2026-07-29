from __future__ import annotations

import numpy as np

from tcd_prg.constants import PUSH_DISTANCE_M


def test_float32_push_distance_is_semantically_exact_015m() -> None:
    stored = float(np.float32(PUSH_DISTANCE_M))
    assert stored != PUSH_DISTANCE_M
    assert np.isclose(stored, PUSH_DISTANCE_M, atol=1e-6, rtol=0.0)
