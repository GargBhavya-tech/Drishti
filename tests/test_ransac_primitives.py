"""
tests/test_ransac_primitives.py

Synthetic validation of perception/ransac_primitives.py, following this
project's own established discipline (Part G.16's synthetic box-split
test) of proving new geometry logic against KNOWN synthetic shapes
before ever running it on real RELLIS-3D data.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.ransac_primitives import (
    fit_vertical_plane,
    fit_vertical_cylinder,
)


def _make_vertical_wall(n_points=200, width=2.0, height=1.5, noise_std=0.005, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, width, n_points)
    v = rng.uniform(0, height, n_points)
    # A wall in the X-Z plane at y=5.0 (normal along Y -- horizontal normal, vertical plane)
    x = u
    y = np.full(n_points, 5.0) + rng.normal(0, noise_std, n_points)
    z = v
    return np.stack([x, y, z], axis=1)


def _make_sloped_ground(n_points=200, extent=2.0, slope=0.3, noise_std=0.01, rng=None):
    rng = rng or np.random.default_rng(1)
    x = rng.uniform(-extent, extent, n_points)
    y = rng.uniform(-extent, extent, n_points)
    z = slope * x + rng.normal(0, noise_std, n_points)  # tilted, but nowhere near vertical
    return np.stack([x, y, z], axis=1)


def _make_vertical_pole(n_points=60, radius=0.08, height=1.8, noise_std=0.005, rng=None):
    rng = rng or np.random.default_rng(2)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = radius + rng.normal(0, noise_std, n_points)
    x = r * np.cos(theta) + 10.0
    y = r * np.sin(theta) - 3.0
    z = rng.uniform(0, height, n_points)
    return np.stack([x, y, z], axis=1)


def _make_wide_rock(n_points=60, radius=0.4, height=0.15, noise_std=0.01, rng=None):
    """A squat, wide, roughly-circular-in-XY blob -- what a real rock
    might look like, deliberately shaped to tempt a naive circle fit."""
    rng = rng or np.random.default_rng(3)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = radius + rng.normal(0, noise_std, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(0, height, n_points)
    return np.stack([x, y, z], axis=1)


def test_vertical_wall_fits_as_a_plane():
    wall = _make_vertical_wall()
    model = fit_vertical_plane(wall)
    assert model is not None
    assert model.inlier_fraction > 0.9
    assert abs(model.normal[2]) < 0.259  # confirms the fitted plane really is vertical


def test_sloped_ground_does_not_fit_as_a_vertical_plane():
    ground = _make_sloped_ground()
    model = fit_vertical_plane(ground)
    # A slope of 0.3 (~16.7 degrees from horizontal) is nowhere near
    # vertical -- the constraint must reject it outright, not merely
    # score it lower.
    assert model is None


def test_flat_ground_does_not_fit_as_a_vertical_plane():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 2, 200)
    y = rng.uniform(-2, 2, 200)
    z = rng.normal(0, 0.01, 200)  # nearly perfectly flat
    flat = np.stack([x, y, z], axis=1)
    model = fit_vertical_plane(flat)
    assert model is None


def test_vertical_pole_fits_as_a_cylinder():
    pole = _make_vertical_pole()
    model = fit_vertical_cylinder(pole)
    assert model is not None
    assert model.inlier_fraction > 0.8
    assert 0.02 <= model.radius_m <= 0.5
    assert (model.z_max - model.z_min) >= 3.0 * model.radius_m


def test_wide_squat_rock_does_not_fit_as_a_cylinder():
    rock = _make_wide_rock()
    model = fit_vertical_cylinder(rock)
    # radius ~0.4 with height ~0.15 fails the height/radius >= 3.0 ratio
    # even though it IS circular in XY -- this is the exact false-
    # positive shape this module's own docstring names.
    assert model is None


def test_too_few_points_returns_none_not_a_crash():
    assert fit_vertical_plane(np.zeros((2, 3))) is None
    assert fit_vertical_cylinder(np.zeros((2, 3))) is None


def test_empty_input_returns_none_not_a_crash():
    assert fit_vertical_plane(np.zeros((0, 3))) is None
    assert fit_vertical_cylinder(np.zeros((0, 3))) is None
