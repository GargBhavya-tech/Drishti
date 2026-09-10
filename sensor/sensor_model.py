"""
sensor/sensor_model.py

Ticket #4 — SensorModel: the seven formulas.

Every function here is a pure function of a SensorConfig. No globals, no
module-level constants, no hardcoded sensor numbers. If a sensor number
needs to change, it changes in a configs/sensor_*.yaml file, never here.

See DRISHTI_Project_Bible_v3.md Part 2 for the derivation of each formula
and DRISHTI_Build_Map.md Ticket #4 for the worked-example test table these
functions are checked against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class SensorConfig:
    """Immutable sensor geometry. Load with `load_sensor_config`, never
    construct with literals scattered through the codebase."""

    sensor_id: str
    n_beams: int
    d_theta_rad: float          # horizontal angular resolution
    d_phi_rad: float            # vertical beam spacing
    phi_max_rad: float          # top-beam elevation above horizontal
    h_m: Optional[float]        # mount height above ground plane
    usable_range_m: Optional[float]

    def __post_init__(self) -> None:
        if self.d_theta_rad is not None and self.d_theta_rad <= 0:
            raise ValueError(f"d_theta_rad must be positive, got {self.d_theta_rad}")
        if self.d_phi_rad is not None and self.d_phi_rad <= 0:
            raise ValueError(f"d_phi_rad must be positive, got {self.d_phi_rad}")


def load_sensor_config(path: str | Path) -> SensorConfig:
    """Load a SensorConfig from a configs/sensor_*.yaml file.

    Deliberately strict about units: the yaml must carry `d_theta_rad` and
    `d_phi_rad` directly (not just the `_deg` variants) so there is no
    degrees/radians conversion silently happening at load time in a place
    someone forgets to check. See Build Map Ticket #4 "Watch out".
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    for required in ("d_theta_rad", "d_phi_rad"):
        if raw.get(required) is None:
            raise ValueError(
                f"{path} is missing '{required}' — every SensorConfig must carry "
                f"radians directly, not just the _deg field. If this is a "
                f"work-in-progress config (e.g. sensor_ouster_os1_64.yaml before "
                f"Ticket #6 has measured it), that is expected: do not compute "
                f"anything from it until the TODO is resolved."
            )

    return SensorConfig(
        sensor_id=raw["sensor_id"],
        n_beams=raw["n_beams"],
        d_theta_rad=float(raw["d_theta_rad"]),
        d_phi_rad=float(raw["d_phi_rad"]),
        phi_max_rad=math.radians(raw.get("phi_max_deg", 0.0)) if raw.get("phi_max_deg") is not None else 0.0,
        h_m=raw.get("h_m"),
        usable_range_m=raw.get("usable_range_m"),
    )


# ---------------------------------------------------------------------------
# The seven formulas. Bible Part 2.
# ---------------------------------------------------------------------------

def s_tangential(r: float, sm: SensorConfig) -> float:
    """Tangential (along-arc) point spacing at range r. s_t(r) = r * d_theta."""
    return r * sm.d_theta_rad


def s_radial_ground(r: float, sm: SensorConfig) -> float:
    """Radial ground-plane spacing between adjacent beam rings at range r.
    s_r(r) ~= r^2 * d_phi / h. Requires h_m to be set."""
    _require_h(sm)
    return r**2 * sm.d_phi_rad / sm.h_m


def s_vertical(r: float, sm: SensorConfig) -> float:
    """Vertical spacing between adjacent beam rings at range r. s_v(r) = r * d_phi."""
    return r * sm.d_phi_rad


def r_max_height(t: float, sm: SensorConfig) -> float:
    """Furthest range at which a feature of height t is still resolved by
    at least one extra beam (height-resolving range). r_max(t) = t / d_phi."""
    return t / sm.d_phi_rad


def r_max_ditch(w: float, sm: SensorConfig) -> float:
    """Furthest range at which a gap of width w can be straddled by the
    beam spacing (negative-obstacle detection range).
    r_max(w) = sqrt(w * h / d_phi). Requires h_m to be set."""
    _require_h(sm)
    return math.sqrt(w * sm.h_m / sm.d_phi_rad)


def n_expected(r: float, t: float, w: float, sm: SensorConfig) -> float:
    """Expected number of returns on an object of height t, width w, at
    range r. N_exp(r; t, w) = t * w / (r^2 * d_phi * d_theta)."""
    return (t * w) / (r**2 * sm.d_phi_rad * sm.d_theta_rad)


def r_blind(t: float, w: float, sm: SensorConfig) -> float:
    """Range beyond which an object of height t, width w has N_exp < 1 and
    must be reported UNKNOWN rather than FREE (Bible Part 11, Claim 3).
    r_blind(t, w) = sqrt(t * w / (d_phi * d_theta))."""
    return math.sqrt((t * w) / (sm.d_phi_rad * sm.d_theta_rad))


def _require_h(sm: SensorConfig) -> None:
    if sm.h_m is None:
        raise ValueError(
            f"SensorConfig '{sm.sensor_id}' has no h_m set — this formula needs "
            f"mount height above the ground plane. For sensor_ouster_os1_64.yaml "
            f"this is expected until Ticket #7-equivalent calibration runs."
        )
