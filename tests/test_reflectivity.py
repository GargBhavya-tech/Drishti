"""
tests/test_reflectivity.py

perception/reflectivity.py: the correction formula itself, checked by
hand for a known input, plus the near-zero-range floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.reflectivity import MIN_RANGE_FOR_CORRECTION_M, REFERENCE_RANGE_M, range_corrected_intensity


def test_matches_the_formula_by_hand():
    intensity = np.array([0.5])
    r = np.array([20.0])
    expected = 0.5 * (20.0 / REFERENCE_RANGE_M) ** 2
    result = range_corrected_intensity(intensity, r)
    assert result[0] == pytest.approx(expected)


def test_at_reference_range_correction_is_a_no_op():
    intensity = np.array([0.3, 0.7])
    r = np.full(2, REFERENCE_RANGE_M)
    result = range_corrected_intensity(intensity, r)
    np.testing.assert_allclose(result, intensity)


def test_farther_range_scales_up_same_raw_intensity():
    intensity = np.array([0.1, 0.1])
    r = np.array([5.0, 50.0])
    result = range_corrected_intensity(intensity, r)
    assert result[1] > result[0]


def test_near_zero_range_is_floored_not_exploded():
    intensity = np.array([1.0])
    r = np.array([0.0])
    result = range_corrected_intensity(intensity, r)
    expected = 1.0 * (MIN_RANGE_FOR_CORRECTION_M / REFERENCE_RANGE_M) ** 2
    assert result[0] == pytest.approx(expected)
    assert np.isfinite(result[0])
