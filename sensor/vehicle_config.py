"""
sensor/vehicle_config.py

Ticket #9 — loader for configs/vehicle_ugv.yaml. Pure loading, no logic;
every consumer imports the resulting VehicleConfig rather than reading
the yaml itself, so there is exactly one place a platform parameter is
declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VehicleConfig:
    max_slope_deg: float
    max_step_height_m: float
    min_clearance_m: float
    ground_clearance_m: float
    width_m: float
    max_roughness_m: float
    min_object_t_m: float
    min_object_w_m: float
    braking_a_ms2: float
    t_react_s: float


def load_vehicle_config(path: str | Path) -> VehicleConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return VehicleConfig(**raw)
