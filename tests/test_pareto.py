"""
tests/test_pareto.py

Ticket #58 tests, per the Build Map's own test list: "at gamma=0
deviation is ~0 and memory is maximal. Deviation increases
monotonically with gamma. The knee is identifiable."
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.pareto import cell_size_for_gamma, default_height_field, find_knee_index, sweep_gamma

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def test_cell_size_at_gamma_zero_is_c0_everywhere():
    r = np.array([0.0, 5.0, 50.0, 200.0])
    c = cell_size_for_gamma(r, gamma=0.0, c0=0.05)
    assert np.allclose(c, 0.05)


def test_cell_size_grows_with_range_for_positive_gamma():
    r = np.array([0.0, 10.0, 50.0, 100.0])
    c = cell_size_for_gamma(r, gamma=1.0, c0=0.05)
    assert np.all(np.diff(c) > 0)  # strictly increasing with range


def test_cell_size_never_goes_finer_than_c0():
    r = np.array([0.0, 1.0, 10.0])
    for gamma in [0.0, 0.5, 3.0]:
        c = cell_size_for_gamma(r, gamma, c0=0.05)
        assert np.all(c >= 0.05 - 1e-12)


def test_at_gamma_zero_deviation_is_exactly_zero_and_memory_is_maximal():
    points = sweep_gamma(GAMMAS, n_per_axis=21)
    zero_gamma = points[0]
    assert zero_gamma.gamma == 0.0
    assert zero_gamma.deviation_rms_m == pytest.approx(0.0, abs=1e-9)
    memories = [p.total_memory_bytes for p in points]
    assert zero_gamma.total_memory_bytes == max(memories)


def test_deviation_increases_monotonically_with_gamma():
    points = sweep_gamma(GAMMAS, n_per_axis=21)
    deviations = [p.deviation_rms_m for p in points]
    assert deviations == sorted(deviations)
    # Not just non-decreasing -- real growth across the swept range,
    # otherwise the "curve" carries no information.
    assert deviations[-1] > deviations[0]


def test_memory_decreases_monotonically_with_gamma():
    points = sweep_gamma(GAMMAS, n_per_axis=21)
    memories = [p.total_memory_bytes for p in points]
    assert memories == sorted(memories, reverse=True)


def test_knee_is_identifiable_and_not_an_endpoint():
    points = sweep_gamma(GAMMAS, n_per_axis=21)
    knee = find_knee_index(points)
    assert 0 < knee < len(points) - 1


def test_deviation_by_band_covers_every_configured_band():
    points = sweep_gamma([0.0, 1.0], n_per_axis=21)
    for p in points:
        assert set(p.deviation_by_band_m.keys()) == {0, 1, 2, 3}


def test_a_flat_height_field_produces_zero_deviation_at_every_gamma():
    def flat(x, y):
        return np.zeros_like(x)

    points = sweep_gamma(GAMMAS, height_fn=flat, n_per_axis=15)
    for p in points:
        assert p.deviation_rms_m == pytest.approx(0.0, abs=1e-9)


def test_default_height_field_is_deterministic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    a = default_height_field(x, y)
    b = default_height_field(x, y)
    assert np.array_equal(a, b)
