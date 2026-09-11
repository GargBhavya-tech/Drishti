"""
tests/test_path_smoothing.py

Kinodynamic path smoothing (planning/path_smoothing.py). Covers:
smooth_path's endpoint-reaching guarantee and its short-input passthrough,
curvature_radius on known straight/right-angle geometry, the friction-
lateral-speed relation's monotonicity and its infinite-radius no-limit
case, and curvature_speed_profile's end-to-end behaviour including its
integration with planning.friction's own CLASS_TO_MU table.
"""

from __future__ import annotations

import math

import pytest

from perception.taxonomy import DrishtiClass
from planning.friction import MU_DRY_REFERENCE, mu_for_class
from planning.path_smoothing import (
    G_MS2,
    curvature_radius,
    curvature_speed_profile,
    lateral_speed_limit_ms,
    smooth_path,
)


def test_smooth_path_short_inputs_pass_through_unchanged():
    assert smooth_path([]) == []
    assert smooth_path([(3, 4)]) == [(3.0, 4.0)]
    assert smooth_path([(0, 0), (5, 0)]) == [(0.0, 0.0), (5.0, 0.0)]


def test_smooth_path_reaches_true_start_and_goal():
    waypoints = [(0, 0), (2, 2), (4, 5), (6, 2), (8, 0)]
    curve = smooth_path(waypoints)
    assert curve[0] == pytest.approx(waypoints[0])
    assert curve[-1] == pytest.approx(waypoints[-1])
    # More samples than input waypoints -- it actually smoothed, not
    # just echoed the same points back.
    assert len(curve) > len(waypoints)


def test_smooth_path_on_a_straight_line_stays_on_the_line():
    # A degenerate case worth checking explicitly: Catmull-Rom through
    # perfectly collinear points must reproduce the line, not wobble
    # off it (a real risk with a naive spline implementation).
    waypoints = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    curve = smooth_path(waypoints)
    for x, y in curve:
        assert y == pytest.approx(0.0, abs=1e-9)


def test_curvature_radius_of_collinear_points_is_infinite():
    assert curvature_radius((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == math.inf
    assert curvature_radius((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)) == math.inf  # repeated point


def test_curvature_radius_of_a_known_right_triangle():
    # A right-angle turn through (0,0) -> (1,0) -> (1,1): the classic
    # circumradius for this exact triangle (legs 1 and 1, hypotenuse
    # sqrt(2)) is sqrt(2)/2 -- worth pinning down by hand once so a
    # future refactor of the circumradius formula can't silently drift.
    r = curvature_radius((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert r == pytest.approx(math.sqrt(2) / 2, abs=1e-9)


def test_lateral_speed_limit_infinite_radius_means_no_limit():
    assert lateral_speed_limit_ms(math.inf, mu=0.8) == math.inf


def test_lateral_speed_limit_matches_the_formula_by_hand():
    mu = 0.5
    r = 10.0
    expected = math.sqrt(mu * G_MS2 * r)
    assert lateral_speed_limit_ms(r, mu) == pytest.approx(expected)


def test_lateral_speed_limit_lower_mu_means_lower_speed_for_the_same_radius():
    r = 15.0
    v_dry = lateral_speed_limit_ms(r, mu=MU_DRY_REFERENCE)
    v_muddy = lateral_speed_limit_ms(r, mu=mu_for_class(DrishtiClass.CAUTION))
    assert v_muddy < v_dry


def test_lateral_speed_limit_rejects_nonpositive_mu():
    with pytest.raises(ValueError):
        lateral_speed_limit_ms(10.0, mu=0.0)


def test_curvature_speed_profile_short_input_returns_empty():
    assert curvature_speed_profile([], cell_size_m=0.32, class_at_point=lambda p: DrishtiClass.DRIVABLE) == []
    assert curvature_speed_profile([(0.0, 0.0), (1.0, 0.0)], 0.32, lambda p: DrishtiClass.DRIVABLE) == []


def test_curvature_speed_profile_tight_turn_through_mud_is_doubly_penalised():
    # A sharp turn (small radius) sampled entirely on CAUTION terrain
    # must produce a lower v_max than the SAME turn sampled on
    # DRIVABLE terrain -- the "doubly penalised" claim in the module
    # docstring, checked directly rather than just asserted in prose.
    sharp_turn = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    dry_profile = curvature_speed_profile(sharp_turn, cell_size_m=1.0, class_at_point=lambda p: DrishtiClass.DRIVABLE)
    muddy_profile = curvature_speed_profile(sharp_turn, cell_size_m=1.0, class_at_point=lambda p: DrishtiClass.CAUTION)
    assert len(dry_profile) == len(muddy_profile) == len(sharp_turn) - 2
    for dry_sample, muddy_sample in zip(dry_profile, muddy_profile):
        assert dry_sample.radius_m == pytest.approx(muddy_sample.radius_m)  # same geometry
        assert muddy_sample.v_max_ms < dry_sample.v_max_ms  # only mu differs


def test_curvature_speed_profile_straight_stretch_has_no_curvature_limit():
    straight = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    profile = curvature_speed_profile(straight, cell_size_m=0.32, class_at_point=lambda p: DrishtiClass.DRIVABLE)
    assert len(profile) == len(straight) - 2
    for sample in profile:
        assert sample.v_max_ms == math.inf
        assert sample.radius_m == math.inf
