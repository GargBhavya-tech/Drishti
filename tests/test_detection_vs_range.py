"""
tests/test_detection_vs_range.py

Ticket #60 tests, per the Build Map's own test list: "measured
detection range tracks prediction within ~20%."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.detection_vs_range import predicted_ditch_range_m, predicted_kerb_range_m, sweep_ditch_detection, sweep_kerb_detection
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


def test_kerb_measured_range_tracks_prediction_within_20_percent(hdl64e):
    predicted = predicted_kerb_range_m(0.15, hdl64e)
    ranges = [predicted * f for f in (0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0)]
    curve = sweep_kerb_detection(0.15, hdl64e, ranges)
    assert curve.measured_range_m == pytest.approx(predicted, rel=0.2)
    # A real curve, not a constant -- detection must actually fall off
    # somewhere within the swept ladder.
    assert 0.0 in curve.detection_rate
    assert 1.0 in curve.detection_rate


def test_ditch_measured_range_tracks_prediction_within_20_percent(hdl64e):
    predicted = predicted_ditch_range_m(2.0, hdl64e)
    ranges = [predicted * f for f in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)]
    curve = sweep_ditch_detection(width_m=2.0, depth_m=0.5, sm=hdl64e, ranges_m=ranges)
    assert curve.measured_range_m == pytest.approx(predicted, rel=0.2)


def test_detection_rate_is_between_zero_and_one_at_every_range(hdl64e):
    predicted = predicted_ditch_range_m(2.0, hdl64e)
    ranges = [predicted * f for f in (0.5, 1.0, 1.5)]
    curve = sweep_ditch_detection(width_m=2.0, depth_m=0.5, sm=hdl64e, ranges_m=ranges)
    assert all(0.0 <= r <= 1.0 for r in curve.detection_rate)


def test_predicted_kerb_and_ditch_ranges_match_portability_formulas(hdl64e):
    from eval.portability import compute_row
    from sensor.vehicle_config import load_vehicle_config

    vehicle = load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")
    row = compute_row(hdl64e, vehicle)
    assert predicted_kerb_range_m(0.15, hdl64e) == pytest.approx(row.kerb_range_m)
    assert predicted_ditch_range_m(2.0, hdl64e) == pytest.approx(row.ditch_range_m)
