"""
Ticket #5 tests, against the worked table in DRISHTI_Build_Map.md.
"""

from pathlib import Path

import pytest

from sensor.sensor_model import load_sensor_config
from sensor.schedule import generate_schedule, n_levels_for_extent

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def test_hdl64e_schedule(hdl64e):
    levels = generate_schedule(hdl64e, c0=0.05, n_levels=4)
    expected = [(0, 0.05, 16.58), (1, 0.10, 33.17), (2, 0.20, 66.33), (3, 0.40, 132.7)]
    for lvl, (l, c, r) in zip(levels, expected):
        assert lvl.level == l
        assert lvl.cell_size_m == pytest.approx(c, abs=1e-6)
        assert lvl.nyquist_radius_m == pytest.approx(r, abs=0.1)


def test_hdl32e_schedule(hdl32e):
    levels = generate_schedule(hdl32e, c0=0.05, n_levels=4)
    expected = [(0, 0.05, 8.59), (1, 0.10, 17.19), (2, 0.20, 34.38), (3, 0.40, 68.75)]
    for lvl, (l, c, r) in zip(levels, expected):
        assert lvl.level == l
        assert lvl.cell_size_m == pytest.approx(c, abs=1e-6)
        assert lvl.nyquist_radius_m == pytest.approx(r, abs=0.1)


def test_no_hardcoded_radii(hdl64e, hdl32e):
    """Same c0/n_levels, different configs -> different radii. If this
    ever fails, someone hardcoded a radius instead of deriving it from
    d_theta_rad."""
    r64 = generate_schedule(hdl64e, c0=0.05, n_levels=1)[0].nyquist_radius_m
    r32 = generate_schedule(hdl32e, c0=0.05, n_levels=1)[0].nyquist_radius_m
    assert r64 != pytest.approx(r32, rel=0.01)


def test_n_levels_for_extent(hdl64e):
    # Level 3 (0.40m cells) reaches 132.7m on HDL-64E, so covering 100m
    # extent needs exactly 4 levels (0..3).
    assert n_levels_for_extent(hdl64e, extent_m=100.0, c0=0.05) == 4
