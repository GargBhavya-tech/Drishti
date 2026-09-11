"""
observability/sparsity.py

Tickets #38-39 -- the Sparsity Trap (Claim 3): expected return count,
kappa, r_blind, and the structure test that decides between normal
confidence, SPARSE_STRUCTURED, noise, FREE, and UNKNOWN.

Bible Part 11: past r_blind for the smallest object the vehicle must
not hit, "no returns" cannot mean "no obstacle" -- the sensor could not
have seen it, so absence proves nothing. The whole module exists to
protect one line in the decision table below:

    N_obs = 0, N_exp < 1  ->  UNKNOWN, never FREE

A model asked to implement this "simplifies" that row to "no returns ->
free", because that is what every conventional occupancy grid does.
That row IS Claim 3 -- do not let it collapse (Build Map Ticket #39's
own explicit warning).

t_min/w_min come from `vehicle_ugv.yaml`'s `min_object_t_m`/
`min_object_w_m` (the sponsor's stated safety requirement -- "the
smallest thing we refuse to hit"), never hardcoded here or anywhere
downstream -- `tests/test_vehicle_config.py` already greps the repo for
the two other watched vehicle literals; this module's own tests check
the same discipline for these two.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from sensor.sensor_model import SensorConfig, n_expected
from sensor.sensor_model import r_blind as _r_blind
from sensor.vehicle_config import VehicleConfig


class SparsityVerdict(IntEnum):
    """Bible Part 11's decision table, Ticket #39. Ordered roughly by
    increasing caution, though this is not itself the conservatism cost
    order (Ticket #41 owns that) -- just a readable enum."""

    NORMAL = 0  # kappa >= 1, N_obs > 0: resolvable and observed
    SPARSE_STRUCTURED = 1  # 0 < kappa < 1, structured (or N_obs == 1): flagged, low confidence, NOT free
    NOISE_SUPPRESSED = 2  # 0 < kappa < 1, unstructured: scattered, treated as noise
    FREE = 3  # N_obs == 0, N_exp >= 1: the sensor WOULD have seen it
    UNKNOWN = 4  # N_obs == 0, N_exp < 1: past r_blind -- Claim 3


@dataclass(frozen=True)
class SparsityResult:
    n_obs: int
    n_exp: float
    kappa: float  # 0.0 when n_obs == 0 (kappa is only meaningful for n_obs > 0; verdict already encodes the n_obs==0 cases)
    verdict: SparsityVerdict


def n_exp_for_min_object(r: float, vehicle: VehicleConfig, sm: SensorConfig) -> float:
    """N_exp(r) for the smallest object this vehicle must not hit.
    t_min/w_min from the vehicle config, never a literal here."""
    return n_expected(r, vehicle.min_object_t_m, vehicle.min_object_w_m, sm)


def r_blind_for_min_object(vehicle: VehicleConfig, sm: SensorConfig) -> float:
    """The range beyond which the smallest object this vehicle must not
    hit drops below one expected return -- past this, absence of
    returns proves nothing (Claim 3)."""
    return _r_blind(vehicle.min_object_t_m, vehicle.min_object_w_m, sm)


def is_structured(ranges: np.ndarray, ring_indices: np.ndarray, vehicle: VehicleConfig) -> bool:
    """Real thin objects return points that are co-located: tightly
    clustered in range AND contiguous in ring index (Bible Part 11).
    Noise returns are scattered in both.

    Range-spread threshold: `min_object_w_m` -- a real thin vertical
    feature (the object this whole module sizes itself to) has close to
    zero range-depth other than measurement noise at any viewing angle,
    so its own configured width is a physically-grounded scale for "how
    much range spread is still plausible for that object", rather than
    an arbitrary fraction picked to look right. Bible Principle 5's own
    caveat still applies: this is a reasoned default, not yet calibrated
    against labelled real sparse-return data (no such ground truth
    exists in this build) -- calibrate it if/when that data does.
    """
    if ranges.size == 0:
        return False
    range_spread = float(np.max(ranges) - np.min(ranges))
    if range_spread > vehicle.min_object_w_m:
        return False

    sorted_rings = np.unique(ring_indices)
    if sorted_rings.size <= 1:
        return True
    return bool(np.all(np.diff(sorted_rings) == 1))


def classify_sparsity(
    n_obs: int,
    r: float,
    vehicle: VehicleConfig,
    sm: SensorConfig,
    ranges: Optional[np.ndarray] = None,
    ring_indices: Optional[np.ndarray] = None,
) -> SparsityResult:
    """The full Ticket #39 decision table for one cell/direction at
    range r. `ranges`/`ring_indices` (the cell's own observed returns)
    are only consulted when 0 < kappa < 1 and n_obs > 1 -- the structure
    test. Omitting them in that regime defaults to NOT structured
    (the more cautious of the two non-NORMAL sparse verdicts is
    SPARSE_STRUCTURED, so failing to prove structure, rather than
    assuming it, is the conservative direction -- Bible Principle 6).
    """
    n_exp = n_exp_for_min_object(r, vehicle, sm)

    if n_obs == 0:
        verdict = SparsityVerdict.FREE if n_exp >= 1.0 else SparsityVerdict.UNKNOWN
        return SparsityResult(n_obs=0, n_exp=n_exp, kappa=0.0, verdict=verdict)

    kappa = n_obs / n_exp if n_exp > 0 else float("inf")

    if kappa >= 1.0:
        return SparsityResult(n_obs=n_obs, n_exp=n_exp, kappa=kappa, verdict=SparsityVerdict.NORMAL)

    # 0 < kappa < 1.
    if n_obs == 1:
        # A single point can never be proven unstructured -- the
        # cautious verdict applies regardless of what ranges/ring_indices
        # say (Ticket #39's own explicit rule).
        return SparsityResult(n_obs=n_obs, n_exp=n_exp, kappa=kappa, verdict=SparsityVerdict.SPARSE_STRUCTURED)

    structured = (
        is_structured(np.asarray(ranges), np.asarray(ring_indices), vehicle)
        if ranges is not None and ring_indices is not None
        else False
    )
    verdict = SparsityVerdict.SPARSE_STRUCTURED if structured else SparsityVerdict.NOISE_SUPPRESSED
    return SparsityResult(n_obs=n_obs, n_exp=n_exp, kappa=kappa, verdict=verdict)
