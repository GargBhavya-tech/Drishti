"""
tests/test_portability.py

Ticket #59 tests. Note on scope: this project's own real, measured
Ouster OS1-64 constants (Ticket #6) do NOT match the Build Map's own
placeholder table (which explicitly says those numbers are "carried
from the HDL-64E reference config" and warns "do not present the table
below as-is if [the measurement] hasn't run" -- it has, and the real
numbers differ from the placeholder, as expected). This file therefore
asserts:
- HDL-64E and HDL-32E reproduce the BIBLE'S OWN exact headline numbers
  (the two columns the Build Map's placeholder table was never a
  placeholder for).
- The Ouster OS1-64 column is real, finite, and internally consistent
  with the same formulas -- not asserted against the Build Map's
  placeholder numbers, which this project was explicitly told not to
  trust for that sensor.
- No code change was required to add the third config: the SAME
  `compute_row()` function, unchanged, produces all three rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.portability import build_portability_table, compute_row, format_table
from sensor.sensor_model import load_sensor_config
from sensor.vehicle_config import load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def test_hdl64e_reproduces_the_bibles_own_headline_numbers_exactly(vehicle):
    sm = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")
    row = compute_row(sm, vehicle)
    assert row.level0_reach_m == pytest.approx(16.6, abs=0.05)
    assert row.kerb_range_m == pytest.approx(20.2, abs=0.05)
    assert row.ditch_range_m == pytest.approx(21.6, abs=0.05)
    assert row.pedestrian_r_blind_m == pytest.approx(194.8, abs=0.1)
    assert row.safe_speed_ditch_kmh == pytest.approx(43.2, abs=0.05)


def test_hdl32e_reproduces_the_bibles_own_headline_numbers_exactly(vehicle):
    sm = load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")
    row = compute_row(sm, vehicle)
    assert row.level0_reach_m == pytest.approx(8.6, abs=0.05)
    assert row.kerb_range_m == pytest.approx(6.4, abs=0.05)
    assert row.ditch_range_m == pytest.approx(12.6, abs=0.05)
    assert row.pedestrian_r_blind_m == pytest.approx(79.2, abs=0.1)
    assert row.safe_speed_ditch_kmh == pytest.approx(32.0, abs=0.05)


def test_ouster_os1_64_column_is_real_measured_and_self_consistent(vehicle):
    sm = load_sensor_config(CONFIGS / "sensor_ouster_os1_64.yaml")
    row = compute_row(sm, vehicle)
    # Real, finite numbers -- not the HDL-64E reference placeholders,
    # and not NaN/inf from a config that failed to load real constants.
    for value in (row.level0_reach_m, row.kerb_range_m, row.ditch_range_m, row.pedestrian_r_blind_m, row.safe_speed_ditch_kmh):
        assert value > 0
        assert value == value  # not NaN
    # Self-consistency: the same underlying formulas link ditch range
    # to safe speed for THIS sensor's own measured ditch range, exactly
    # as they do for the other two sensors.
    from planning.speed_envelope import v_max_for_range

    assert row.safe_speed_ditch_kmh == pytest.approx(v_max_for_range(row.ditch_range_m, vehicle) * 3.6)


def test_level_boundary_scales_with_d_theta_only_not_d_phi(vehicle):
    """Build Map's own "Watch out": level boundaries scale with
    d_theta ONLY. Two sensors with the SAME d_theta but DIFFERENT
    d_phi must report the SAME level0_reach_m -- if they didn't,
    d_phi would have silently leaked into a d_theta-only formula."""
    from sensor.sensor_model import SensorConfig

    sm_a = SensorConfig(sensor_id="a", n_beams=64, d_theta_rad=0.003, d_phi_rad=0.005, phi_max_rad=0.1, h_m=1.7, usable_range_m=100.0)
    sm_b = SensorConfig(sensor_id="b", n_beams=64, d_theta_rad=0.003, d_phi_rad=0.020, phi_max_rad=0.1, h_m=1.7, usable_range_m=100.0)
    row_a = compute_row(sm_a, vehicle)
    row_b = compute_row(sm_b, vehicle)
    assert row_a.level0_reach_m == pytest.approx(row_b.level0_reach_m)
    # But the d_phi-dependent rows MUST differ between the two.
    assert row_a.kerb_range_m != pytest.approx(row_b.kerb_range_m)


def test_build_portability_table_uses_the_same_function_for_all_three_configs(vehicle):
    rows = build_portability_table(vehicle)
    assert set(rows.keys()) == {"hdl64e", "ouster_os1_64", "hdl32e"}
    # Every row was produced by calling compute_row() with a real,
    # independently-loaded config -- confirmed by cross-checking one
    # row directly against a standalone compute_row() call.
    sm = load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")
    standalone = compute_row(sm, vehicle)
    assert rows["hdl32e"] == standalone


def test_format_table_includes_all_three_sensors_and_every_quantity(vehicle):
    rows = build_portability_table(vehicle)
    text = format_table(rows)
    for label in ("HDL-64E", "Ouster OS1-64", "HDL-32E"):
        assert label in text
    for quantity in ("5cm level reaches", "15cm kerb", "2m ditch", "Pedestrian r_blind", "Safe speed"):
        assert quantity in text
