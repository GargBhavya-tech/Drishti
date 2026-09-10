"""
perception/sweep.py

The canonical Sweep struct every dataset loader (#2, #3, and any future
loader) must return. Defined once, here, so nothing downstream needs to
know which dataset a Sweep came from.

Spec is exact per DRISHTI_Build_Map.md, Ticket #2 / #4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sweep:
    xyz: np.ndarray         # (N, 3) float32, SENSOR frame
    intensity: np.ndarray   # (N,)   float32, 0..1 normalised
    ring: np.ndarray        # (N,)   int16, beam index; -1 if unavailable
    timestamp: float        # seconds
    T_world: np.ndarray     # (4, 4) float64, sensor -> world
    sensor_id: str          # 'hdl32e' | 'hdl64e' | 'ouster_os1_64'

    def __post_init__(self) -> None:
        n = self.xyz.shape[0]
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz must be (N,3), got {self.xyz.shape}")
        if self.intensity.shape != (n,):
            raise ValueError(f"intensity must be (N,), got {self.intensity.shape}")
        if self.ring.shape != (n,):
            raise ValueError(f"ring must be (N,), got {self.ring.shape}")
        if self.T_world.shape != (4, 4):
            raise ValueError(f"T_world must be (4,4), got {self.T_world.shape}")
