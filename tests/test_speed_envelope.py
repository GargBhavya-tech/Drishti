"""
tests/test_speed_envelope.py

Ticket #40 tests -- Claim 4. Pure algebra: assert against the Build
Map's own table, assert monotonicity (a shorter detection range must
never produce a higher safe speed), and assert lowering `a` lowers
v_max.

Note on the acceptance numbers: v_max(21.6) = 12.00 m/s = 43.2 km/h
matches EXACTLY with vehicle_ugv.yaml's real a=4.0/t_react=0.30 (and is
this project's own headline number, repeated throughout the Bible). The
Build Map's other two table entries (v_max(6.7)="6.24 m/s = 22.5 km/h",
v_max(12.6)="8.99 m/s = 32.0 km/h") do NOT reproduce exactly from the
same formula and config (computed here: 6.2189 m/s and 8.9114 m/s) --
each pairing isn't even fully self-consistent under simple rounding
(6.24 m/s -> 22.46 km/h, not 22.5; 8.99 m/s -> 32.36 km/h, not 32.0).
This is the same class of issue as the Bible's own Part 8 worked-example
arithmetic error (see tests/test_addressing.py) -- the FORMULA and
CONFIG are correct (proven by the exact 21.6m match, this project's most
load-bearing single number), so this file asserts the precisely
COMPUTED values for the other two ranges, not the Bible's imprecise
prose restatement of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.speed_envelope import speed_envelope, stopping_distance_m, v_max_for_range
from sensor.vehicle_config import VehicleConfig, load_vehicle_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def vehicle():
    return load_vehicle_config(CONFIGS / "vehicle_ugv.yaml")


def test_v_max_21_6_matches_the_headline_number_exactly(vehicle):
    """The project's own most-repeated number: 43.2 km/h for a 2m ditch
    at HDL-64E's 21.6m detection range."""
    v = v_max_for_range(21.6, vehicle)
    assert v == pytest.approx(12.00, abs=0.01)
    assert v * 3.6 == pytest.approx(43.2, abs=0.05)


def test_v_max_other_ranges_match_the_precisely_computed_values(vehicle):
    assert v_max_for_range(6.7, vehicle) == pytest.approx(6.2189, abs=0.01)
    assert v_max_for_range(12.6, vehicle) == pytest.approx(8.9114, abs=0.01)


def test_monotonicity_shorter_range_never_yields_higher_speed(vehicle):
    ranges = [5.0, 10.0, 21.6, 30.5, 66.8, 194.8]
    speeds = [v_max_for_range(r, vehicle) for r in ranges]
    assert speeds == sorted(speeds)  # strictly non-decreasing as range grows
    assert all(a < b for a, b in zip(speeds, speeds[1:]))  # strictly increasing, no ties in this set


def test_lowering_braking_capability_lowers_v_max():
    strong_brakes = VehicleConfig(
        max_slope_deg=25, max_step_height_m=0.20, min_clearance_m=2.50, ground_clearance_m=0.35,
        width_m=2.10, max_roughness_m=0.08, min_object_t_m=1.00, min_object_w_m=0.10,
        braking_a_ms2=6.0, t_react_s=0.30,
    )
    weak_brakes = VehicleConfig(
        max_slope_deg=25, max_step_height_m=0.20, min_clearance_m=2.50, ground_clearance_m=0.35,
        width_m=2.10, max_roughness_m=0.08, min_object_t_m=1.00, min_object_w_m=0.10,
        braking_a_ms2=2.0, t_react_s=0.30,
    )
    r = 21.6
    assert v_max_for_range(r, weak_brakes) < v_max_for_range(r, strong_brakes)


def test_stopping_distance_and_v_max_are_inverses(vehicle):
    for r in (5.0, 21.6, 66.8, 194.8):
        v = v_max_for_range(r, vehicle)
        assert stopping_distance_m(v, vehicle) == pytest.approx(r, abs=1e-6)


# ---------------------------------------------------------------------------
# speed_envelope(): the binding-hazard composite
# ---------------------------------------------------------------------------


def test_binding_hazard_is_the_minimum_over_the_active_set(vehicle):
    result = speed_envelope({"2m_ditch": 21.6, "5cm_cable": 6.7, "fence_post_presence": 66.8}, vehicle)
    assert result.binding_hazard == "5cm_cable"
    assert result.binding_range_m == 6.7
    assert result.v_max_ms == pytest.approx(v_max_for_range(6.7, vehicle))
    assert set(result.per_hazard_v_max_ms.keys()) == {"2m_ditch", "5cm_cable", "fence_post_presence"}


def test_speed_envelope_rejects_an_empty_hazard_set(vehicle):
    with pytest.raises(ValueError):
        speed_envelope({}, vehicle)


def test_v_max_kmh_matches_ms_times_3_6(vehicle):
    result = speed_envelope({"2m_ditch": 21.6}, vehicle)
    assert result.v_max_kmh == pytest.approx(result.v_max_ms * 3.6)
