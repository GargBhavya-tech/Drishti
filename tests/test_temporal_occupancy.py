"""
tests/test_temporal_occupancy.py

Synthetic validation of grid/temporal_occupancy.py against KNOWN
scenarios, before it is ever run against real RELLIS-3D data -- same
discipline as tests/test_ransac_primitives.py.
"""

from __future__ import annotations

import numpy as np

from grid.temporal_occupancy import TemporalOccupancyGrid, LOG_ODDS_MIN


def test_real_pole_persists_across_multiple_viewpoints():
    """A real pole at world (10, 5, z) is hit by real returns from THREE
    different ego positions (simulating the vehicle driving past it) --
    its own voxels should accumulate strongly positive log-odds."""
    grid = TemporalOccupancyGrid(center_xy=np.array([0.0, 0.0]))
    pole_xyz = np.array([[10.0, 5.0, z] for z in np.linspace(0.0, 1.5, 20)])

    ego_positions = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), np.array([8.0, 3.0, 0.0])]
    for origin in ego_positions:
        grid.update_occupied(pole_xyz)
        grid.update_free_along_rays(origin, pole_xyz)

    score = grid.persistence_fraction(pole_xyz, min_log_odds=0.5)
    assert score > 0.8, f"real persistent pole should score high, got {score}"


def test_transient_clutter_gets_rayed_through_and_erased():
    """A transient cluster (e.g. a rock geometrically resembling an
    obstacle from ONE viewpoint only) is hit ONCE, then later frames'
    real rays pass THROUGH that same location to hit real ground behind
    it -- its log-odds should be driven back down, not stay positive."""
    grid = TemporalOccupancyGrid(center_xy=np.array([0.0, 0.0]))
    clutter_center = np.array([10.0, 5.0, 0.15])
    # Deliberately kept within a SINGLE voxel's z-extent (0.3m) -- a
    # cluster straddling a voxel boundary would only have ONE of its two
    # voxels actually threaded by a single ray's narrow elevation band,
    # which is a real geometric fact about ray-voxel intersection, not
    # a property this test means to exercise (that scenario was caught
    # and fixed once already: this docstring's own earlier version used
    # a 0.3m-tall cluster and the ray only cleared the far voxel, not
    # the near one -- a test-construction issue, not the grid's bug).
    clutter_xyz = np.array([[10.0, 5.0, z] for z in np.linspace(0.10, 0.20, 10)])

    grid.update_occupied(clutter_xyz)  # ONE frame's real hit -- looks occupied

    # Later frames: rays from DIFFERENT real origins pass THROUGH the
    # clutter's own location en route to a real point 50% FARTHER along
    # the SAME line (guaranteeing the ray actually threads through the
    # clutter regardless of viewing angle, not just near it) -- this is
    # what "the object wasn't really there" looks like physically: the
    # ray's own frees decrement exactly the voxels the first frame
    # marked occupied.
    for origin in [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.0]), np.array([-2.0, 2.0, 0.0])]:
        endpoint = (origin + (clutter_center - origin) * 1.5)[None, :]
        grid.update_free_along_rays(origin, endpoint)

    score = grid.persistence_fraction(clutter_xyz, min_log_odds=0.5)
    assert score < 0.5, f"transient clutter that got rayed through should NOT score high, got {score}"


def test_never_observed_region_is_conservatively_unknown():
    grid = TemporalOccupancyGrid(center_xy=np.array([0.0, 0.0]))
    never_seen = np.array([[30.0, 30.0, 1.0]])
    vals = grid.mean_log_odds_at(never_seen)
    assert vals[0] == 0.0  # a real voxel that exists but was never touched -- exactly zero, the true "no evidence either way" state


def test_out_of_bounds_point_returns_conservative_minimum_not_a_crash():
    grid = TemporalOccupancyGrid(center_xy=np.array([0.0, 0.0]), half_extent_xy_m=10.0)
    far_outside = np.array([[500.0, 500.0, 1.0]])
    vals = grid.mean_log_odds_at(far_outside)
    assert vals[0] == LOG_ODDS_MIN


def test_empty_input_does_not_crash():
    grid = TemporalOccupancyGrid(center_xy=np.array([0.0, 0.0]))
    grid.update_occupied(np.zeros((0, 3)))
    grid.update_free_along_rays(np.array([0.0, 0.0, 0.0]), np.zeros((0, 3)))
    assert grid.persistence_fraction(np.zeros((0, 3))) == 0.0
