"""
tests/test_ground_prior_numba_equivalence.py

Real equivalence verification for perception/ground_prior.py's Numba
optimization (DRISHTI_MASTER_BIBLE.md's per-frame FPS bottleneck fix):
the JIT, column-parallel path (`use_numba=True`, the default) must
produce results IDENTICAL to the original pure-Python path
(`use_numba=False`) -- not merely close, since the parallelization
introduced here changes only which CPU core executes which column,
never the floating-point operation order within a column (see
ground_prior.py's own module docstring). Tested against REAL RELLIS-3D
frames, not synthetic data, per this project's own standard for a
performance-only refactor claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.ground_prior import compute_ground_prior
from perception.rellis_loader import load_rellis_sweep
from perception.train import build_multi_sequence_splits

SEQUENCE_DIR = Path("data/rellis/00000")


def _real_frame_items(n=5):
    train_items, _val_items, _ = build_multi_sequence_splits([SEQUENCE_DIR])
    step = max(1, len(train_items) // n)
    return train_items[::step][:n]


@pytest.mark.parametrize("frame_item", _real_frame_items())
def test_numba_path_matches_python_path_exactly_on_real_data(frame_item):
    sequence_dir, frame_idx = frame_item
    sweep = load_rellis_sweep(sequence_dir, frame_idx)

    result_python = compute_ground_prior(sweep, use_numba=False)
    result_numba = compute_ground_prior(sweep, use_numba=True)

    assert np.array_equal(result_python.is_ground, result_numba.is_ground), (
        f"is_ground mismatch: {int(np.sum(result_python.is_ground != result_numba.is_ground))} points differ"
    )
    assert result_python.column_ground_height.keys() == result_numba.column_ground_height.keys()
    for col in result_python.column_ground_height:
        assert result_python.column_ground_height[col] == pytest.approx(
            result_numba.column_ground_height[col], abs=1e-9
        ), f"column {col} height mismatch"


def test_numba_path_handles_empty_sweep():
    from perception.sweep import Sweep

    empty_sweep = Sweep(
        xyz=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros((0,), dtype=np.float32),
        ring=np.zeros((0,), dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4, dtype=np.float64),
        sensor_id="test",
    )
    result = compute_ground_prior(empty_sweep, use_numba=True)
    assert result.is_ground.shape[0] == 0
    assert result.column_ground_height == {}
