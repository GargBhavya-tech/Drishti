"""
tests/test_fovea_controller.py

Ticket #47 tests, per the Build Map's own test list:
- "The floor test is the first test in the file": 10,000 Hypothesis
  cases asserting c(p) <= c_range(r), unconditionally.
- Standstill produces cell sizes no coarser than Profile A.
- The fovea's principal axis rotates with the velocity vector.
- Bible Part 13's own worked table (15 m ahead/left at 15 m/s; 100 m
  ahead), using the REAL HDL-64E config -- these numbers reproduce
  exactly (unlike the speed-envelope table, this one checks out to the
  Bible's stated precision against the real config, verified by hand
  before writing this test).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from attention.fovea_controller import c_range, c_ttc, fovea_cell_size, ttc_s, v_close
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_HDL64E = load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def hdl64e():
    return _HDL64E


# ---------------------------------------------------------------------------
# The floor test -- first in the file, per the Build Map's own instruction.
# ---------------------------------------------------------------------------


@st.composite
def velocities(draw):
    speed = draw(st.floats(min_value=0.0, max_value=40.0, allow_nan=False))
    heading = draw(st.floats(min_value=0.0, max_value=2 * math.pi, allow_nan=False))
    return (speed * math.cos(heading), speed * math.sin(heading))


@st.composite
def points(draw):
    r = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False))
    bearing = draw(st.floats(min_value=0.0, max_value=2 * math.pi, allow_nan=False))
    return (r * math.cos(bearing), r * math.sin(bearing))


@given(v=velocities(), p=points())
@settings(max_examples=10_000)
def test_floor_never_violated(v, p):
    r = math.hypot(p[0], p[1])
    floor = c_range(r, _HDL64E)
    result = fovea_cell_size(p, v, _HDL64E, profile="B", cm=None)
    assert result <= floor + 1e-9


# ---------------------------------------------------------------------------
# Standstill: Profile B must be no coarser than Profile A anywhere.
# ---------------------------------------------------------------------------


def test_standstill_produces_cell_sizes_no_coarser_than_profile_a(hdl64e):
    v = (0.0, 0.0)
    for p in [(15.0, 0.0), (60.0, 0.0), (100.0, 0.0), (0.0, 30.0), (5.0, 5.0)]:
        profile_a = fovea_cell_size(p, v, hdl64e, profile="A")
        profile_b = fovea_cell_size(p, v, hdl64e, profile="B")
        assert profile_b <= profile_a + 1e-9


# ---------------------------------------------------------------------------
# Principal axis rotates with the velocity vector.
# ---------------------------------------------------------------------------


def test_fine_region_principal_axis_rotates_with_velocity_vector():
    # Ahead-of-travel point gets a finer c_ttc than a lateral point at
    # the SAME range, and which point counts as "ahead" rotates with v.
    speed = 15.0
    r = 60.0

    v_east = (speed, 0.0)
    ahead_east = (r, 0.0)
    lateral_to_east = (0.0, r)
    c_ahead_east = c_ttc(v_east, ahead_east)
    c_lateral_east = c_ttc(v_east, lateral_to_east)
    assert c_ahead_east < c_lateral_east  # finer dead ahead than to the side

    v_north = (0.0, speed)
    ahead_north = (0.0, r)  # now "ahead" is what was lateral before
    lateral_to_north = (r, 0.0)
    c_ahead_north = c_ttc(v_north, ahead_north)
    c_lateral_north = c_ttc(v_north, lateral_to_north)
    assert c_ahead_north < c_lateral_north

    # The specific point (r, 0) is "ahead" (fine) when v points east
    # and "lateral" (coarse) when v points north -- the fine region
    # follows the velocity vector's direction, not a fixed world axis.
    assert c_ttc(v_east, (r, 0.0)) < c_ttc(v_north, (r, 0.0))


# ---------------------------------------------------------------------------
# Bible Part 13's worked example (15 m/s, tau0=1.0s, c0=5cm, gamma=1,
# v_min=2m/s), against the REAL HDL-64E config.
# ---------------------------------------------------------------------------


def test_worked_table_15m_ahead_gives_5cm(hdl64e):
    v = (15.0, 0.0)
    p = (15.0, 0.0)
    assert v_close(v, p) == pytest.approx(15.0)
    assert ttc_s(v, p) == pytest.approx(1.0)
    assert c_ttc(v, p) == pytest.approx(0.05, abs=1e-9)
    assert c_range(15.0, hdl64e) == pytest.approx(0.05, abs=1e-9)
    assert fovea_cell_size(p, v, hdl64e, profile="B") == pytest.approx(0.05, abs=1e-9)


def test_worked_table_100m_ahead_ttc_refines_beyond_the_range_floor(hdl64e):
    v = (15.0, 0.0)
    p = (100.0, 0.0)
    assert ttc_s(v, p) == pytest.approx(100.0 / 15.0)
    assert c_ttc(v, p) == pytest.approx(0.05 * (100.0 / 15.0), abs=1e-6)
    assert c_range(100.0, hdl64e) == pytest.approx(0.40, abs=1e-9)  # coarsest level, per the schedule
    result = fovea_cell_size(p, v, hdl64e, profile="B")
    assert result == pytest.approx(0.3333, abs=1e-3)  # TTC term wins, refining well below the 40cm floor
    assert result < c_range(100.0, hdl64e)


def test_worked_table_15m_left_floor_holds(hdl64e):
    v = (15.0, 0.0)  # heading east
    p = (0.0, 15.0)  # 15m directly to the side (north)
    assert v_close(v, p) == pytest.approx(0.0)
    assert ttc_s(v, p) == pytest.approx(7.5)  # v_min=2 m/s floors the closing speed
    c_ttc_value = c_ttc(v, p)
    assert c_ttc_value == pytest.approx(0.375, abs=1e-6)  # would be coarser than the floor
    result = fovea_cell_size(p, v, hdl64e, profile="B")
    assert result == pytest.approx(0.05, abs=1e-9)  # c_range's floor wins -- "floor holds"


def test_profile_a_is_c_range_alone_regardless_of_velocity(hdl64e):
    p = (60.0, 0.0)
    fast = fovea_cell_size(p, (40.0, 0.0), hdl64e, profile="A")
    stopped = fovea_cell_size(p, (0.0, 0.0), hdl64e, profile="A")
    assert fast == stopped == c_range(60.0, hdl64e)


def test_unknown_profile_raises(hdl64e):
    with pytest.raises(ValueError):
        fovea_cell_size((10.0, 0.0), (0.0, 0.0), hdl64e, profile="C")
