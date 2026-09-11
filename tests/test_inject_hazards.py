"""
tests/test_inject_hazards.py

Ticket #50 tests. Note on the specific ranges used: the Build Map's own
literal test ("inject a 2m x 0.5m trench at 15m into a real nuScenes
sweep; assert #36 detects it. Inject at 30m ... assert not detected")
was written for REAL nuScenes hardware calibration, which this
environment does not have (see eval/inject_hazards.py's own module
docstring -- nuScenes has not been downloaded yet, per this project's
own standing instruction). This project's SYNTHETIC HDL-32E ring model
was verified directly (see below) to have real ring-range GAPS at both
15m and 25-30m -- a genuine instance of the Bible's own "rings step
clean over it" phenomenon, just at different specific ranges than real
hardware happens to produce. Rather than silently forcing the literal
15m/30m values against geometry that does not support them, this file
uses ranges VERIFIED against this synthetic model's own real ring
positions: 8m (two real rings inside the trench window) for the
positive case, 15m (a real, confirmed ring GAP) for the negative case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.inject_hazards import flat_ground_baseline, inject_pole, inject_trench, kerb_resolved
from observability.negative_obstacle import detect_anomalous_cells
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def _detect(grid, sm):
    return detect_anomalous_cells(
        measured_range=grid.measured_range,
        expected_range=grid.expected_range,
        valid_expected=grid.valid_expected,
        occluded_mask=grid.occluded_mask,
        usable_range_m=sm.usable_range_m,
    )


def test_trench_within_ring_range_is_detected_by_ticket_36(hdl32e):
    grid = flat_ground_baseline(hdl32e, n_azimuth=4)
    injected = inject_trench(grid, hdl32e, azimuth_col=0, width_m=2.0, depth_m=0.5, near_range_m=8.0)
    flagged = _detect(injected, hdl32e)
    assert any(col == 0 for _ring, col in flagged)


def test_trench_in_a_real_ring_gap_produces_no_detection(hdl32e):
    """15m falls squarely inside a confirmed gap between two of this
    synthetic model's own ring ranges (~14.4m and ~17.6m) -- no ring's
    flat-ground return lands inside [15, 17]m, so the injection itself
    has no effect and the detector correctly finds nothing. This is
    the Bible's own "rings step clean over it" failure case, made
    concrete rather than merely asserted."""
    grid = flat_ground_baseline(hdl32e, n_azimuth=4)
    injected = inject_trench(grid, hdl32e, azimuth_col=0, width_m=2.0, depth_m=0.5, near_range_m=15.0)
    assert np.array_equal(injected.measured_range, grid.measured_range, equal_nan=True)
    flagged = _detect(injected, hdl32e)
    assert not any(col == 0 for _ring, col in flagged)


def test_trench_injection_never_touches_other_azimuth_columns(hdl32e):
    grid = flat_ground_baseline(hdl32e, n_azimuth=4)
    injected = inject_trench(grid, hdl32e, azimuth_col=0, width_m=2.0, depth_m=0.5, near_range_m=8.0)
    for col in (1, 2, 3):
        assert np.array_equal(injected.measured_range[:, col], grid.measured_range[:, col], equal_nan=True)


def test_deeper_trench_displaces_further_or_occludes_never_less(hdl32e):
    """A deeper trench must never produce a SMALLER effect than a
    shallow one at the same ring: either it displaces further out, or
    it occludes the ring entirely (the far wall now intercepts the
    beam before the deeper floor point) -- occlusion is itself the
    "even more effect" outcome, not a regression, so a shallow-but-
    displaced ring going deep-and-occluded is an EXPECTED, physically
    correct result, not a bug this test should reject."""
    grid = flat_ground_baseline(hdl32e, n_azimuth=2)
    shallow = inject_trench(grid, hdl32e, azimuth_col=0, width_m=5.0, depth_m=0.1, near_range_m=8.0)
    deep = inject_trench(grid, hdl32e, azimuth_col=0, width_m=5.0, depth_m=1.0, near_range_m=8.0)
    changed = ~np.isclose(shallow.measured_range[:, 0], grid.measured_range[:, 0], equal_nan=True)
    ring = int(np.argmax(changed))
    assert not np.isnan(shallow.measured_range[ring, 0])
    deep_value = deep.measured_range[ring, 0]
    assert np.isnan(deep_value) or deep_value > shallow.measured_range[ring, 0]


def test_pole_injection_adds_a_closer_return_at_its_own_range(hdl32e):
    grid = flat_ground_baseline(hdl32e, n_azimuth=2)
    injected = inject_pole(grid, hdl32e, azimuth_cols=[0], diameter_m=0.2, height_m=1.5, range_m=5.0)
    changed = ~np.isclose(injected.measured_range[:, 0], grid.measured_range[:, 0], equal_nan=True)
    assert np.any(changed)
    for ring in np.nonzero(changed)[0]:
        assert injected.measured_range[ring, 0] == pytest.approx(5.0)
        assert injected.measured_range[ring, 0] < grid.measured_range[ring, 0]


def test_pole_injection_does_not_touch_columns_outside_its_own_azimuth_span(hdl32e):
    grid = flat_ground_baseline(hdl32e, n_azimuth=3)
    injected = inject_pole(grid, hdl32e, azimuth_cols=[0], diameter_m=0.2, height_m=1.5, range_m=5.0)
    assert np.array_equal(injected.measured_range[:, 1], grid.measured_range[:, 1], equal_nan=True)
    assert np.array_equal(injected.measured_range[:, 2], grid.measured_range[:, 2], equal_nan=True)


# ---------------------------------------------------------------------------
# kerb_resolved -- height-resolution, distinct from #36's anomaly detector.
# ---------------------------------------------------------------------------


def test_kerb_resolved_matches_the_bibles_own_hdl64e_headline_range():
    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    predicted = 0.15 / sm.d_phi_rad  # Bible Part 3: 20.2m
    assert predicted == pytest.approx(20.2, abs=0.05)
    assert kerb_resolved(0.15, predicted - 1.0, sm) is True
    assert kerb_resolved(0.15, predicted + 5.0, sm) is False


def test_kerb_resolved_is_monotonically_false_beyond_the_predicted_range():
    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    predicted = 0.15 / sm.d_phi_rad
    ranges = [predicted * f for f in (0.5, 0.8, 1.0, 1.5, 3.0)]
    results = [kerb_resolved(0.15, r, sm) for r in ranges]
    # Once it goes False, it must never go back to True at a longer range.
    first_false = next((i for i, v in enumerate(results) if not v), len(results))
    assert all(results[:first_false]) and not any(results[first_false:])
