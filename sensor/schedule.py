"""
sensor/schedule.py

Ticket #5 — Resolution schedule generator.

Given a SensorConfig and a base cell size c0, emit the power-of-two
resolution levels the clipmap (Ticket #11 onward) will use, and the
sensor-Nyquist outer radius at which each level's cell size stops being
finer than what the sensor can actually fill.

The Nyquist radius r_l is NOT the array extent (N * c_l) — see Bible
Part 5 / Build Map Ticket #5 "Watch out". Conflating the two is the
single most common way a model asked to "implement variable resolution"
gets this wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sensor.sensor_model import SensorConfig


@dataclass(frozen=True)
class Level:
    level: int
    cell_size_m: float
    nyquist_radius_m: float


def generate_schedule(sm: SensorConfig, c0: float = 0.05, n_levels: int = 4) -> List[Level]:
    """Level l has cell size c_l = c0 * 2^l and sensor-Nyquist outer radius
    r_l = c_l / d_theta — the range at which the tangential point spacing
    s_tangential(r) first exceeds c_l, i.e. where the sensor can no longer
    fill a cell of that size with more than about one point per azimuth
    step. Purely a function of d_theta; independent of d_phi and h_m.
    """
    levels: List[Level] = []
    for l in range(n_levels):
        c_l = c0 * (2 ** l)
        r_l = c_l / sm.d_theta_rad
        levels.append(Level(level=l, cell_size_m=c_l, nyquist_radius_m=r_l))
    return levels


def n_levels_for_extent(sm: SensorConfig, extent_m: float, c0: float = 0.05, max_levels: int = 8) -> int:
    """How many levels are needed so the coarsest level's Nyquist radius
    covers the requested extent. Does not by itself determine the clipmap
    array size (N * c_l) — that's a separate, deliberate choice (Ticket #11),
    not derived from this function.
    """
    for l in range(max_levels):
        c_l = c0 * (2 ** l)
        r_l = c_l / sm.d_theta_rad
        if r_l >= extent_m:
            return l + 1
    return max_levels
