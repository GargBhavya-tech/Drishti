"""
tests/test_range_image.py

Tickets #23 (spherical projection) + #24 (occlusion depth) tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from perception.range_image import project_to_range_image
from perception.rellis_loader import load_rellis_sweep
from perception.sweep import Sweep
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RELLIS_SEQ_DIR = Path(__file__).resolve().parents[1] / "data" / "rellis" / "00004"


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def _synthetic_sweep_with_ring(n_beams=32, points_per_beam=20) -> Sweep:
    """A clean nuScenes-shaped sweep: ring known exactly, points spread
    around a full circle at a fixed range per beam."""
    rings = []
    xyz = []
    for beam in range(n_beams):
        elevation = math.radians(-25 + beam * (30 / n_beams))  # arbitrary spread
        for k in range(points_per_beam):
            az = 2 * math.pi * k / points_per_beam
            r = 10.0
            x = r * math.cos(elevation) * math.cos(az)
            y = r * math.cos(elevation) * math.sin(az)
            z = r * math.sin(elevation)
            xyz.append([x, y, z])
            rings.append(beam)
    xyz = np.array(xyz, dtype=np.float32)
    n = xyz.shape[0]
    return Sweep(
        xyz=xyz,
        intensity=np.full(n, 0.5, dtype=np.float32),
        ring=np.array(rings, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="hdl32e",
    )


def test_v_from_ring_matches_ring_exactly(hdl32e):
    sweep = _synthetic_sweep_with_ring(n_beams=32, points_per_beam=50)
    img = project_to_range_image(sweep, hdl32e, W=1080)

    # For every touched pixel, the row it landed in must equal the
    # source point's own ring (Ticket #23's own required assertion).
    touched = np.nonzero(img.valid_mask)
    for row, col in zip(*touched):
        src_idx = img.point_index[row, col]
        assert sweep.ring[src_idx] == row


def test_round_trip_within_one_cell(hdl32e):
    sweep = _synthetic_sweep_with_ring(n_beams=32, points_per_beam=200)
    img = project_to_range_image(sweep, hdl32e, W=1080)

    kept = img.point_index[img.valid_mask]
    assert kept.size > 0
    # Round-trip: the pixel's own recorded x/y/z must equal the source
    # point's x/y/z exactly (it IS that point, not a resampled value).
    rows, cols = np.nonzero(img.valid_mask)
    for row, col in zip(rows[:50], cols[:50]):  # sample, full sweep is slow in pure Python
        src_idx = img.point_index[row, col]
        assert img.x[row, col] == pytest.approx(sweep.xyz[src_idx, 0])
        assert img.y[row, col] == pytest.approx(sweep.xyz[src_idx, 1])
        assert img.z[row, col] == pytest.approx(sweep.xyz[src_idx, 2])


def test_valid_mask_sum_close_to_n_at_low_density(hdl32e):
    """Few collisions expected at 32 beams with generous azimuth
    resolution and modest point density."""
    sweep = _synthetic_sweep_with_ring(n_beams=32, points_per_beam=50)
    img = project_to_range_image(sweep, hdl32e, W=1080)
    n = sweep.xyz.shape[0]
    assert img.valid_mask.sum() >= 0.95 * n


def test_atan2_argument_order_not_swapped(hdl32e):
    """Regression guard for Ticket #23 Watch-out #3: atan2(y,x), not
    atan2(x,y) -- a swap mirrors the image but still 'looks' plausible."""
    W = 1080
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],   # +x axis
            [0.0, 1.0, 0.0],   # +y axis
            [0.0, -1.0, 0.0],  # -y axis
        ],
        dtype=np.float32,
    )
    sweep = Sweep(
        xyz=xyz,
        intensity=np.full(3, 0.5, dtype=np.float32),
        ring=np.array([0, 0, 0], dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="hdl32e",
    )
    sm = load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")
    img = project_to_range_image(sweep, sm, W=W)

    def col_of(x, y):
        return int(np.floor(0.5 * (1.0 - np.arctan2(y, x) / np.pi) * W))

    assert col_of(1.0, 0.0) == W // 2
    assert col_of(0.0, 1.0) == W // 4
    assert col_of(0.0, -1.0) == 3 * W // 4


def test_two_returns_one_bin_20m_apart_occlusion(hdl32e):
    """Ticket #24's own test: two returns in one angular bin 20 m apart
    -> occlusion_count == 1 (one discarded), spread ~= 20."""
    az = 0.0
    xyz = np.array(
        [
            [10.0 * math.cos(az), 10.0 * math.sin(az), 0.0],  # near, kept
            [30.0 * math.cos(az), 30.0 * math.sin(az), 0.0],  # far, discarded
        ],
        dtype=np.float32,
    )
    sweep = Sweep(
        xyz=xyz,
        intensity=np.array([0.5, 0.5], dtype=np.float32),
        ring=np.array([5, 5], dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="hdl32e",
    )
    img = project_to_range_image(sweep, hdl32e, W=1080)

    rows, cols = np.nonzero(img.valid_mask)
    assert len(rows) == 1  # both points collided into one pixel
    row, col = rows[0], cols[0]
    assert img.occlusion_count[row, col] == 1
    assert img.occlusion_spread[row, col] == pytest.approx(20.0, abs=0.01)
    assert img.range[row, col] == pytest.approx(10.0, abs=0.01)  # nearest kept


def test_ring_unavailable_falls_back_to_arcsin(hdl32e):
    """RELLIS-shaped sweep (ring = -1 throughout) must still project,
    using the arcsin elevation formula instead of requiring ring."""
    n = 200
    rng = np.random.default_rng(3)
    az = rng.uniform(0, 2 * math.pi, n)
    el = rng.uniform(math.radians(-25), math.radians(2), n)
    r = 15.0
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    sweep = Sweep(
        xyz=xyz,
        intensity=np.full(n, 0.5, dtype=np.float32),
        ring=np.full(n, -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="ouster_os1_64",
    )
    img = project_to_range_image(sweep, hdl32e, W=1080)
    assert img.valid_mask.sum() > 0


@pytest.mark.skipif(not RELLIS_SEQ_DIR.exists(), reason="real RELLIS-3D sequence 00004 not present locally")
def test_real_rellis_sweep_projects_without_crashing():
    """Sanity check against real data: produces a plausible-density
    'panorama', not garbage."""
    ouster_cfg_path = CONFIGS / "sensor_ouster_os1_64.yaml"
    if not ouster_cfg_path.exists():
        pytest.skip("sensor_ouster_os1_64.yaml not present")
    sm = load_sensor_config(ouster_cfg_path)
    sweep = load_rellis_sweep(RELLIS_SEQ_DIR, frame_idx=1000)
    img = project_to_range_image(sweep, sm)

    assert img.H == sm.n_beams
    occupancy = img.valid_mask.mean()
    assert 0.1 < occupancy < 1.0  # a real panorama, not empty and not impossible
