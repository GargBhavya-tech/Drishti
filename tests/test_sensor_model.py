"""
Ticket #4 tests. Every assertion here is checked against the worked-example
table in DRISHTI_Build_Map.md, tolerance 0.1 m unless noted, and was
independently verified with a standalone Python computation before this
file was written (not just against itself).
"""

import math
from pathlib import Path

import pytest

from sensor.sensor_model import load_sensor_config, s_tangential, s_radial_ground, \
    r_max_height, r_max_ditch, n_expected, r_blind

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def test_hdl64e_table(hdl64e):
    assert s_tangential(10, hdl64e) == pytest.approx(0.0302, abs=0.001)
    assert s_radial_ground(100, hdl64e) == pytest.approx(42.9, abs=0.1)
    assert r_max_height(0.15, hdl64e) == pytest.approx(20.2, abs=0.1)
    assert r_max_ditch(2.0, hdl64e) == pytest.approx(21.6, abs=0.1)
    assert r_blind(1.7, 0.5, hdl64e) == pytest.approx(194.8, abs=0.1)
    assert n_expected(100, 3.0, 0.2, hdl64e) == pytest.approx(2.68, abs=0.01)


def test_hdl32e_table(hdl32e):
    assert r_max_height(0.15, hdl32e) == pytest.approx(6.4, abs=0.1)
    assert r_max_ditch(2.0, hdl32e) == pytest.approx(12.6, abs=0.1)
    assert r_blind(1.7, 0.5, hdl32e) == pytest.approx(79.2, abs=0.1)


def test_degrees_vs_radians_guard(tmp_path):
    """A config missing d_theta_rad/d_phi_rad must fail loudly, not silently
    fall back to interpreting the _deg field as radians (the 57x bug the
    Build Map's Ticket #4 'Watch out' warns about)."""
    bad = tmp_path / "sensor_bad.yaml"
    bad.write_text(
        "sensor_id: bad\nn_beams: 64\nd_theta_deg: 0.1728\nd_phi_deg: 0.4254\nh_m: 1.73\n"
    )
    with pytest.raises(ValueError):
        load_sensor_config(bad)


def test_n_expected_falls_as_inverse_square(hdl64e):
    """Claim 3's structure test depends on N_exp falling as 1/r^2, not 1/r."""
    n_10 = n_expected(10, 1.0, 0.1, hdl64e)
    n_20 = n_expected(20, 1.0, 0.1, hdl64e)
    ratio = n_10 / n_20
    assert ratio == pytest.approx(4.0, rel=0.01)  # (20/10)^2 = 4


def test_r_max_ditch_requires_h(tmp_path):
    """h_m: null (as in the Ouster placeholder config) must raise, not
    silently compute a wrong number with h=None or h=0."""
    no_h = tmp_path / "sensor_no_h.yaml"
    no_h.write_text(
        "sensor_id: no_h\nn_beams: 64\n"
        "d_theta_rad: 0.003\nd_phi_rad: 0.007\nh_m: null\n"
    )
    sm = load_sensor_config(no_h)
    with pytest.raises(ValueError):
        r_max_ditch(2.0, sm)


def test_ouster_placeholder_config_loads_but_is_flagged():
    """The Ouster config is expected to load (Ticket #4 doesn't block on
    Ticket #6 having run) but its h_m is None, so anything needing h must
    still fail per test_r_max_ditch_requires_h above -- this test just
    confirms the config itself is well-formed enough to load."""
    sm = load_sensor_config(CONFIGS / "sensor_ouster_os1_64.yaml")
    assert sm.sensor_id == "ouster_os1_64"
    assert sm.h_m is None
