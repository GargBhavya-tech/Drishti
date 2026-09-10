"""Shape-validation tests for the canonical Sweep struct itself, independent
of any dataset loader."""

import numpy as np
import pytest

from perception.sweep import Sweep


def _valid_kwargs(n=5):
    return dict(
        xyz=np.zeros((n, 3), dtype=np.float32),
        intensity=np.zeros((n,), dtype=np.float32),
        ring=np.full((n,), -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id="hdl32e",
    )


def test_valid_sweep_constructs():
    Sweep(**_valid_kwargs())


def test_mismatched_intensity_shape_rejected():
    kwargs = _valid_kwargs()
    kwargs["intensity"] = np.zeros((3,), dtype=np.float32)  # wrong N
    with pytest.raises(ValueError):
        Sweep(**kwargs)


def test_wrong_T_world_shape_rejected():
    kwargs = _valid_kwargs()
    kwargs["T_world"] = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError):
        Sweep(**kwargs)
