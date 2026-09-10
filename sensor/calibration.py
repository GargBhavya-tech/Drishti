"""
sensor/calibration.py

Ticket #7 — verify mount height from calibration.

`h` in every ground formula (s_radial_ground, r_max_ditch) is height
ABOVE THE GROUND PLANE, not above the ego origin or the rear axle -- a
wrong `h` is the same class of error as the extrinsic-tilt error in
Bible Part 3.6 (r_max_ditch scales as sqrt(h), so a 10% error in h is a
~5% error in every negative-obstacle detection range).

Two independent estimates, cross-checked against each other:

  1. From the dataset's own calibration record (e.g. nuScenes'
     `calibrated_sensor` table: sensor-to-ego translation z, plus ego
     height above ground).
  2. From fitting a plane to ground-proxy points across many sweeps in
     the SENSOR frame and taking the plane's offset -- this doesn't
     trust the calibration record at all, so it catches a calibration
     bug the first method can't see by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from perception.sweep import Sweep


@dataclass
class HeightEstimate:
    h_from_calibration_m: float
    h_from_ground_fit_m: float
    ground_fit_residual_m: float  # RMS residual of the plane fit -- large means "not flat" or "wrong points"
    agree_within_m: float

    @property
    def agrees(self) -> bool:
        return abs(self.h_from_calibration_m - self.h_from_ground_fit_m) <= self.agree_within_m


def h_from_calibration(sensor_to_ego_translation_z_m: float, ego_height_above_ground_m: float) -> float:
    """h = how far the sensor sits above the ground plane. Both terms are
    read from the dataset's calibration/vehicle-geometry records, never
    guessed."""
    return sensor_to_ego_translation_z_m + ego_height_above_ground_m


def _ground_proxy_mask(xyz: np.ndarray, z_band_m: float = 0.15) -> np.ndarray:
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    floor = np.percentile(xyz[:, 2], 5)
    return np.abs(xyz[:, 2] - floor) <= z_band_m


def h_from_ground_plane_fit(sweeps: Sequence[Sweep], z_band_m: float = 0.15) -> tuple[float, float]:
    """Fit z = a*x + b*y + c to ground-proxy points across many sweeps, in
    the SENSOR frame (i.e. before applying T_world -- the sensor doesn't
    move relative to the vehicle body, so pooling multiple sweeps in the
    sensor frame is valid and gives more points than one sweep alone).

    Returns (h_estimate_m, rms_residual_m). h_estimate = -c under the
    assumption the sensor is close to level; a large residual is itself
    informative (either the ground-proxy mask is picking up non-ground
    points, or the mount has non-trivial roll/pitch that this simple fit
    doesn't model -- see Bible Part 3.6 for the tilt case).
    """
    pts = []
    for sw in sweeps:
        mask = _ground_proxy_mask(sw.xyz, z_band_m=z_band_m)
        if mask.sum() > 0:
            pts.append(sw.xyz[mask])
    if not pts:
        raise ValueError("no ground-proxy points found across the given sweeps")
    all_pts = np.concatenate(pts, axis=0)

    A = np.column_stack([all_pts[:, 0], all_pts[:, 1], np.ones(all_pts.shape[0])])
    b = all_pts[:, 2]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_x, a_y, c = coeffs

    pred = A @ coeffs
    residual = b - pred
    rms = float(np.sqrt(np.mean(residual**2)))

    h_estimate = float(-c)  # ground sits at z ~= c in sensor frame; sensor is h above ground => c = -h
    return h_estimate, rms


def cross_check_mount_height(
    sensor_to_ego_translation_z_m: float,
    ego_height_above_ground_m: float,
    sweeps: Sequence[Sweep],
    agree_within_m: float = 0.10,
    z_band_m: float = 0.15,
) -> HeightEstimate:
    h_calib = h_from_calibration(sensor_to_ego_translation_z_m, ego_height_above_ground_m)
    h_fit, residual = h_from_ground_plane_fit(sweeps, z_band_m=z_band_m)
    return HeightEstimate(
        h_from_calibration_m=h_calib,
        h_from_ground_fit_m=h_fit,
        ground_fit_residual_m=residual,
        agree_within_m=agree_within_m,
    )
