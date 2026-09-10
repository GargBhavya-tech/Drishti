"""
tests/test_input_tensor.py

Ticket #27 tests: shape, normalisation (except valid_mask), and the
saved-stats round trip (loaded at inference, not recomputed).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from perception.ground_prior import compute_ground_prior
from perception.input_tensor import (
    N_CHANNELS,
    VALID_MASK_CHANNEL,
    _raw_channels,
    assemble_input_tensor,
    compute_channel_stats,
    ground_prior_channel_from_points,
    load_stats,
    save_stats,
)
from perception.range_image import project_to_range_image
from perception.rellis_loader import load_rellis_sweep
from perception.sweep import Sweep
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RELLIS_SEQ_DIR = Path(__file__).resolve().parents[1] / "data" / "rellis" / "00004"


def _synthetic_sweep(n=300, seed=0) -> Sweep:
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2 * math.pi, n)
    el = rng.uniform(math.radians(-20), math.radians(5), n)
    r = rng.uniform(3.0, 30.0, n)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return Sweep(
        xyz=xyz,
        intensity=rng.uniform(0, 1, n).astype(np.float32),
        ring=np.full(n, -1, dtype=np.int16),
        timestamp=0.0,
        T_world=np.eye(4),
        sensor_id="test",
    )


@pytest.fixture
def hdl32e():
    return load_sensor_config(CONFIGS / "sensor_hdl32e.yaml")


def test_tensor_shape_is_9_H_W(hdl32e):
    sweep = _synthetic_sweep()
    img = project_to_range_image(sweep, hdl32e, W=360)
    ground = compute_ground_prior(sweep, n_azimuth_bins=360)
    raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
    stats = compute_channel_stats([raw])
    tensor = assemble_input_tensor(img, ground, stats)
    assert tensor.shape == (N_CHANNELS, img.H, img.W)
    assert tensor.shape == (9, 32, 360)


def test_normalised_channels_are_roughly_zero_mean_unit_std(hdl32e):
    sweep = _synthetic_sweep(n=2000, seed=1)
    img = project_to_range_image(sweep, hdl32e, W=720)
    ground = compute_ground_prior(sweep, n_azimuth_bins=720)
    raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
    stats = compute_channel_stats([raw])
    tensor = assemble_input_tensor(img, ground, stats)

    valid = img.valid_mask
    for c in range(N_CHANNELS):
        if c == VALID_MASK_CHANNEL:
            continue
        vals = tensor[c][valid]
        if vals.size == 0:
            continue
        assert abs(vals.mean()) < 0.15, f"channel {c} mean not ~0: {vals.mean()}"
        assert abs(vals.std() - 1.0) < 0.15, f"channel {c} std not ~1: {vals.std()}"


def test_valid_mask_channel_stays_binary_never_normalised(hdl32e):
    sweep = _synthetic_sweep()
    img = project_to_range_image(sweep, hdl32e, W=360)
    ground = compute_ground_prior(sweep, n_azimuth_bins=360)
    raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
    stats = compute_channel_stats([raw])
    tensor = assemble_input_tensor(img, ground, stats)

    unique_vals = set(np.unique(tensor[VALID_MASK_CHANNEL]).tolist())
    assert unique_vals <= {0.0, 1.0}


def test_saved_stats_loaded_at_inference_match_original(tmp_path, hdl32e):
    """Ticket #27: 'Assert the saved stats file is loaded at inference,
    not recomputed.' Stats computed from one (training) frame, saved,
    loaded back, and applied to a DIFFERENT (held-out) frame must give
    bit-identical normalisation to applying the original in-memory stats
    object directly -- nothing gets silently recomputed on load."""
    train_sweep = _synthetic_sweep(n=1000, seed=10)
    train_img = project_to_range_image(train_sweep, hdl32e, W=720)
    train_ground = compute_ground_prior(train_sweep, n_azimuth_bins=720)
    train_raw = _raw_channels(train_img, ground_prior_channel_from_points(train_img, train_ground))
    stats = compute_channel_stats([train_raw])

    stats_path = tmp_path / "channel_stats.json"
    save_stats(stats, stats_path)
    loaded_stats = load_stats(stats_path)
    assert loaded_stats == stats

    held_out_sweep = _synthetic_sweep(n=500, seed=99)
    held_out_img = project_to_range_image(held_out_sweep, hdl32e, W=720)
    held_out_ground = compute_ground_prior(held_out_sweep, n_azimuth_bins=720)

    tensor_with_original = assemble_input_tensor(held_out_img, held_out_ground, stats)
    tensor_with_loaded = assemble_input_tensor(held_out_img, held_out_ground, loaded_stats)
    np.testing.assert_array_equal(tensor_with_original, tensor_with_loaded)


@pytest.mark.skipif(not RELLIS_SEQ_DIR.exists(), reason="real RELLIS-3D sequence 00004 not present locally")
def test_real_rellis_frame_assembles_end_to_end():
    ouster_cfg_path = CONFIGS / "sensor_ouster_os1_64.yaml"
    if not ouster_cfg_path.exists():
        pytest.skip("sensor_ouster_os1_64.yaml not present")
    sm = load_sensor_config(ouster_cfg_path)
    sweep = load_rellis_sweep(RELLIS_SEQ_DIR, frame_idx=1000)

    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
    raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
    stats = compute_channel_stats([raw])
    tensor = assemble_input_tensor(img, ground, stats)

    assert tensor.shape == (9, sm.n_beams, img.W)
    assert np.isfinite(tensor).all()
