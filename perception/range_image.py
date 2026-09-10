"""
perception/range_image.py

Ticket #23 -- spherical range-image projection: turn one Sweep's point
cloud into a dense (H, W) tensor in the sensor's own coordinate system.

Ticket #24 -- occlusion depth channels -- is built into the SAME pass
deliberately: Build Map Ticket #24 says "count during the same scatter
that selects the nearest return... do not let it become a second pass
over the cloud." Both tickets are one vectorised computation here.

Projects in the SENSOR frame (Sweep.xyz), not world frame -- Bible Part 7:
"the front end's job is to be shaped like the sensor."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.sweep import Sweep
from sensor.sensor_model import SensorConfig


@dataclass(frozen=True)
class RangeImage:
    x: np.ndarray  # (H, W) float64
    y: np.ndarray
    z: np.ndarray
    range: np.ndarray
    intensity: np.ndarray
    valid_mask: np.ndarray  # (H, W) bool -- True where a real return was kept
    occlusion_count: np.ndarray  # (H, W) int -- how many returns were discarded here
    occlusion_spread: np.ndarray  # (H, W) float -- deepest discarded minus kept range
    point_index: np.ndarray  # (H, W) int64 -- index into the source Sweep, -1 if none
    H: int
    W: int


def project_to_range_image(sweep: Sweep, sm: SensorConfig, W: int | None = None) -> RangeImage:
    """Spherical projection. H = sm.n_beams. W defaults to the azimuth
    resolution implied by d_theta_rad (2*pi / d_theta_rad), matching Part
    3's "a cell should match the beam footprint" reasoning applied to
    columns instead of map cells; pass W explicitly to override.

    Elevation row (v): uses `sweep.ring` directly when available (Ticket
    #23 "Watch out" #1 -- the arcsin form assumes uniform beam spacing,
    which is false for real sensors, and warps the image where rings
    aren't evenly spaced). RELLIS-3D's KITTI-format .bin files do NOT
    ship ring (perception/rellis_loader.py reports ring=-1 throughout,
    documented there) -- this function falls back to the arcsin formula
    per point whenever ring is unavailable, rather than requiring it.

    Azimuth column (u): atan2(y, x) -- NOT atan2(x, y). Reversed, the
    image mirrors (Ticket #23 "Watch out" #3).

    Keeps the NEAREST return per pixel on a many-to-one collision; the
    rest are counted, not discarded silently (occlusion_count/spread).
    """
    H = sm.n_beams
    if W is None:
        W = int(round(2 * np.pi / sm.d_theta_rad))

    xyz = sweep.xyz.astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    n = x.shape[0]

    real_point = r > 1e-6  # guards the "no return" zero-vector padding some formats ship

    u = np.floor(0.5 * (1.0 - np.arctan2(y, x) / np.pi) * W).astype(np.int64)
    u = np.clip(u, 0, W - 1)

    ring = sweep.ring.astype(np.int64)
    has_ring = ring >= 0
    v = np.where(has_ring, ring, 0)

    if not np.all(has_ring):
        phi_min = sm.phi_max_rad - (sm.n_beams - 1) * sm.d_phi_rad
        fov = sm.phi_max_rad - phi_min
        with np.errstate(invalid="ignore", divide="ignore"):
            elevation = np.arcsin(np.clip(z / np.maximum(r, 1e-9), -1.0, 1.0))
        v_fallback = np.floor((1.0 - (elevation - phi_min) / fov) * H).astype(np.int64)
        v = np.where(has_ring, v, v_fallback)

    v_in_range = (v >= 0) & (v < H)
    valid_point = real_point & v_in_range
    v = np.clip(v, 0, H - 1)

    H_out = np.zeros((H, W), dtype=np.float64)
    x_img = np.zeros((H, W), dtype=np.float64)
    y_img = np.zeros((H, W), dtype=np.float64)
    z_img = np.zeros((H, W), dtype=np.float64)
    range_img = np.zeros((H, W), dtype=np.float64)
    intensity_img = np.zeros((H, W), dtype=np.float64)
    valid_mask = np.zeros((H, W), dtype=bool)
    occlusion_count = np.zeros((H, W), dtype=np.int64)
    occlusion_spread = np.zeros((H, W), dtype=np.float64)
    point_index = np.full((H, W), -1, dtype=np.int64)

    idx = np.nonzero(valid_point)[0]
    if idx.size == 0:
        return RangeImage(
            x=x_img, y=y_img, z=z_img, range=range_img, intensity=intensity_img,
            valid_mask=valid_mask, occlusion_count=occlusion_count,
            occlusion_spread=occlusion_spread, point_index=point_index, H=H, W=W,
        )

    pix = v[idx] * W + u[idx]
    r_valid = r[idx]

    # Primary sort key = pix (grouping), secondary = range ascending
    # (nearest first within each pixel's group). np.lexsort's LAST key is
    # the primary sort key.
    order = np.lexsort((r_valid, pix))
    sorted_pix = pix[order]
    sorted_r = r_valid[order]
    sorted_src = idx[order]

    is_first = np.ones(order.shape[0], dtype=bool)
    is_first[1:] = sorted_pix[1:] != sorted_pix[:-1]

    group_start = np.flatnonzero(is_first)
    group_end = np.r_[group_start[1:], order.shape[0]]
    group_size = group_end - group_start

    touched_pix = sorted_pix[group_start]
    kept_src = sorted_src[group_start]  # nearest point's ORIGINAL index, per touched pixel
    kept_r = sorted_r[group_start]
    deepest_r = sorted_r[group_end - 1]  # groups sorted ascending -> last = farthest

    flat_x, flat_y, flat_z = x[kept_src], y[kept_src], z[kept_src]
    flat_intensity = sweep.intensity[kept_src]

    x_img.flat[touched_pix] = flat_x
    y_img.flat[touched_pix] = flat_y
    z_img.flat[touched_pix] = flat_z
    range_img.flat[touched_pix] = kept_r
    intensity_img.flat[touched_pix] = flat_intensity
    valid_mask.flat[touched_pix] = True
    occlusion_count.flat[touched_pix] = group_size - 1
    occlusion_spread.flat[touched_pix] = deepest_r - kept_r
    point_index.flat[touched_pix] = kept_src

    return RangeImage(
        x=x_img, y=y_img, z=z_img, range=range_img, intensity=intensity_img,
        valid_mask=valid_mask, occlusion_count=occlusion_count,
        occlusion_spread=occlusion_spread, point_index=point_index, H=H, W=W,
    )
