"""
tests/test_surface_geometry.py

perception/surface_geometry.py: a hand-constructed flat plane gives a
known, checkable normal (straight up) and near-zero curvature; a
step/cliff gives a large curvature and an invalid region at the
discontinuity; azimuth wraparound is verified explicitly (the seam must
NOT be treated as a discontinuity).
"""

from __future__ import annotations

import numpy as np

from perception.range_image import RangeImage
from perception.surface_geometry import compute_surface_geometry


def _flat_ground_range_image(H=8, W=16) -> RangeImage:
    """A REAL angularly-sampled range image of a flat ground plane at
    z=-1 -- row = elevation angle (downward-looking, one per beam),
    column = azimuth angle, exactly how a real spinning LiDAR samples
    the world (perception.range_image's own convention), NOT a directly
    parameterised x/y grid. This matters: for FIXED elevation, range to
    a flat ground plane is EXACTLY constant across azimuth (basic
    trig), so this fixture's horizontal neighbours have a real,
    well-posed, genuinely-flat tangent -- a naive grid where x/y are
    set directly (an earlier version of this fixture) gives a
    degenerate all-zero vertical tangent (no elevation variation) and a
    range that is NOT physically flat, producing spurious "curvature"
    that has nothing to do with the actual 3D surface's true (zero)
    curvature.
    """
    phi = np.linspace(np.radians(-30.0), np.radians(-5.0), H)  # elevation, downward-looking
    theta = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)  # azimuth, full real wraparound
    ground_z = -1.0

    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    r = ground_z / np.sin(phi_grid)  # constant per row (elevation), by construction
    x = r * np.cos(phi_grid) * np.cos(theta_grid)
    y = r * np.cos(phi_grid) * np.sin(theta_grid)
    z = r * np.sin(phi_grid)  # == ground_z everywhere, up to floating point
    valid = np.ones((H, W), dtype=bool)
    return RangeImage(
        x=x, y=y, z=z, range=r, intensity=np.zeros((H, W)),
        valid_mask=valid, occlusion_count=np.zeros((H, W), dtype=int),
        occlusion_spread=np.zeros((H, W)), point_index=np.full((H, W), -1, dtype=np.int64),
        H=H, W=W,
    )


def test_flat_plane_gives_near_vertical_normal_and_low_curvature():
    img = _flat_ground_range_image()
    geom = compute_surface_geometry(img)
    interior = geom.geometry_valid[1:-1, :]
    assert interior.any(), "expected at least some valid interior geometry on a flat plane"
    assert np.all(np.abs(geom.normal_z[1:-1, :][interior]) > 0.9)
    # Horizontal (azimuth) range differences are EXACTLY zero for flat
    # ground at fixed elevation (real trig, not an approximation) --
    # any remaining curvature here comes only from the vertical
    # (elevation-to-elevation) term's real, small, angle-discretisation
    # non-linearity, not from a bug. A generous bound distinguishes
    # "small and real" from "something is actually broken".
    assert np.all(geom.curvature[1:-1, :][interior] < 0.5)


def test_invalid_pixels_produce_zero_not_garbage():
    img = _flat_ground_range_image()
    img.valid_mask[3, 3] = False
    geom = compute_surface_geometry(img)
    assert not geom.geometry_valid[3, 3]
    assert geom.normal_x[3, 3] == 0.0
    assert geom.normal_y[3, 3] == 0.0
    assert geom.normal_z[3, 3] == 0.0
    assert geom.curvature[3, 3] == 0.0


def test_occlusion_boundary_is_excluded_not_fabricated():
    """A sudden range jump between neighbouring columns (a cliff) must
    NOT produce a normal/curvature value as if the two points were on
    one continuous surface."""
    img = _flat_ground_range_image()
    # Punch a huge step in range/z for one column, well past the
    # MAX_NEIGHBOR_RANGE_JUMP_M threshold.
    img.z[:, 8] = -50.0
    img.range[:, 8] = np.sqrt(img.x[:, 8] ** 2 + img.y[:, 8] ** 2 + img.z[:, 8] ** 2)
    geom = compute_surface_geometry(img)
    # Columns 7 and 8 straddle the discontinuity -- neither should claim
    # a valid geometry computed across that gap.
    assert not geom.geometry_valid[2, 7]
    assert not geom.geometry_valid[2, 8]


def test_azimuth_wraparound_is_not_treated_as_a_discontinuity():
    """The last column's neighbour is column 0 (real 360-degree wrap) --
    the SAME realistic flat-ground fixture used above, whose azimuth
    values genuinely close smoothly around the seam via cos/sin (not
    just numerically close by construction accident), must still report
    valid geometry at that seam, not a fabricated edge."""
    img = _flat_ground_range_image()
    geom = compute_surface_geometry(img)
    assert geom.geometry_valid[3, img.W - 1], "the wraparound seam must not be treated as an occlusion boundary"
