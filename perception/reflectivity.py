"""
perception/reflectivity.py

Range-corrected ("calibrated") LiDAR intensity: raw received laser
power falls off with range (roughly 1/r^2 for many beam models -- see
this module's own caller, eval/validate_feature_hypotheses.py, for the
real-data check of whether this correction actually helps BEFORE it is
wired into training), so a raw intensity channel conflates "how far
away" with "how reflective" -- two physically different quantities.
Dividing out the range dependence recovers a quantity closer to true
surface reflectivity, which is a real, sensor-physics-motivated
candidate fix for the "Ground Paradox" this project's own zero-shot
nuScenes analysis diagnosed (BASELINE_COMPARISON.md): flat asphalt and
flat grass are geometrically identical, so a purely geometric input
cannot distinguish them, but they have materially different near-
infrared reflectivity.

IMPORTANT, stated plainly: whether RELLIS-3D's raw `.bin` intensity
field is itself a trustworthy, real, meaningfully-varying signal (as
opposed to sensor-firmware noise, a placeholder, or already partially
range-compensated by the driver) was flagged as an OPEN, unverified
assumption in `perception/rellis_loader.py`'s own comment ("typically
already ~0..1... verify against real data") -- never actually checked.
`eval/validate_feature_hypotheses.py` checks it empirically before this
module's output is trusted for training.
"""

from __future__ import annotations

import numpy as np

# An arbitrary but fixed reference range -- the correction only needs to
# be internally consistent (same reference for every frame/point), not
# tied to a specific real-world calibration distance, since the network
# only ever sees the corrected value, never the reference itself.
REFERENCE_RANGE_M = 10.0

# Avoids an explosive correction factor for near-zero-range points
# (e.g. sensor-housing self-returns) blowing the corrected value up
# arbitrarily -- these should already be filtered by valid_mask, but a
# floor here is a second, cheap line of defence.
MIN_RANGE_FOR_CORRECTION_M = 0.5


def range_corrected_intensity(
    intensity_raw: np.ndarray, range_m: np.ndarray, reference_range_m: float = REFERENCE_RANGE_M
) -> np.ndarray:
    """I_calibrated = I_raw * (r / r_ref)^2 -- see module docstring for
    the physical justification and the open verification question."""
    safe_range = np.maximum(range_m, MIN_RANGE_FOR_CORRECTION_M)
    return intensity_raw * (safe_range / reference_range_m) ** 2
