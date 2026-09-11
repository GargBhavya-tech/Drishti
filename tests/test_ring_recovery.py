"""
tests/test_ring_recovery.py

Tests for perception/ring_recovery.py, supporting Ticket #6 against
RELLIS-3D's Ouster OS1-64 stream (which ships no per-point ring field).

Two tiers, matching tests/test_rellis_loader.py's own pattern:

1. Synthetic-fixture tests of the pure position-arithmetic logic -- run
   always, no real data needed.
2. Real-data tests confirming the actual claim this module's docstring
   makes (position-in-group is a fixed physical beam property) against
   the real downloaded dataset -- skipped if it isn't present locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.ring_recovery import aggregate_beam_elevation_table, recover_ring_from_scan_order
from perception.rellis_loader import load_rellis_sweep

RELLIS_ROOT = Path(__file__).resolve().parents[1] / "data" / "rellis"
SEQUENCES_AVAILABLE = [s for s in ("00000", "00001", "00002", "00003", "00004") if (RELLIS_ROOT / s).exists()]


def test_ring_is_position_modulo_n_beams():
    ring = recover_ring_from_scan_order(n_points=192, n_beams=64)  # 3 full azimuth ticks
    expected = np.tile(np.arange(64), 3)
    np.testing.assert_array_equal(ring, expected)


def test_ring_range_is_0_to_n_beams_minus_1():
    ring = recover_ring_from_scan_order(n_points=64 * 10, n_beams=64)
    assert ring.min() == 0
    assert ring.max() == 63


def test_non_multiple_of_n_beams_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        recover_ring_from_scan_order(n_points=100, n_beams=64)


def test_aggregate_table_averages_across_frames_and_flags_undersampled_beams():
    n_beams = 4
    elevation_rad = 1.0
    # A unit vector at exactly elevation_rad: x=cos, y=0, z=sin, so
    # r == 1 exactly and arcsin(z/r) == elevation_rad exactly.
    point_at_elevation = [np.cos(elevation_rad), 0.0, np.sin(elevation_rad)]

    # Frame 1: beam 0 has one point at elevation 1.0 rad, others empty.
    frame1 = np.zeros((n_beams, 3))
    frame1[0] = point_at_elevation
    # Frame 2: beam 0 has another point at the same physical beam angle
    # (repeated measurements of the same beam agree, as real data does --
    # see this module's own docstring).
    frame2 = np.zeros((n_beams, 3))
    frame2[0] = point_at_elevation

    table = aggregate_beam_elevation_table([frame1, frame2], n_beams=n_beams, min_valid_per_beam=2)
    assert table[0] == pytest.approx(1.0, abs=1e-6)
    # Beams 1-3 never had a single real point in either frame (all zero,
    # i.e. r <= 1e-6) -- must be NaN (explicit gap), not silently 0.
    assert np.isnan(table[1:]).all()


def test_aggregate_table_respects_min_valid_per_beam_threshold():
    n_beams = 2
    frame = np.zeros((n_beams, 3))
    frame[0] = [2.0, 0.0, 0.0]  # exactly one real point at beam 0
    table = aggregate_beam_elevation_table([frame], n_beams=n_beams, min_valid_per_beam=2)
    assert np.isnan(table[0])  # only 1 valid point, threshold is 2


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_real_frame_point_count_is_a_multiple_of_64():
    """The core assumption this whole module depends on."""
    seq = SEQUENCES_AVAILABLE[0]
    sweep = load_rellis_sweep(RELLIS_ROOT / seq, frame_idx=0)
    assert sweep.xyz.shape[0] % 64 == 0


@pytest.mark.skipif(len(SEQUENCES_AVAILABLE) < 2, reason="need at least 2 local RELLIS-3D sequences for this cross-check")
def test_recovered_ring_matches_a_fixed_per_position_elevation_on_real_data():
    """The machine-checked form of this module's central claim: the
    elevation angle at a fixed ring position is the same real physical
    beam angle regardless of which frame or sequence it came from --
    confirmed in this build to agree to ~1e-6 degrees; this test uses a
    looser (still tight) tolerance to stay robust to future data."""
    seq_a, seq_b = SEQUENCES_AVAILABLE[0], SEQUENCES_AVAILABLE[1]
    sweep_a = load_rellis_sweep(RELLIS_ROOT / seq_a, frame_idx=0)
    sweep_b = load_rellis_sweep(RELLIS_ROOT / seq_b, frame_idx=0)

    table_a = aggregate_beam_elevation_table([sweep_a.xyz], n_beams=64, min_valid_per_beam=3)
    table_b = aggregate_beam_elevation_table([sweep_b.xyz], n_beams=64, min_valid_per_beam=3)

    both_present = ~np.isnan(table_a) & ~np.isnan(table_b)
    assert both_present.sum() > 20, "too few beams had data in both frames to compare"
    max_diff_deg = np.degrees(np.max(np.abs(table_a[both_present] - table_b[both_present])))
    assert max_diff_deg < 0.05, f"beam elevation disagrees by {max_diff_deg} deg across sequences -- position-in-group may not be a fixed beam after all"


@pytest.mark.skipif(not SEQUENCES_AVAILABLE, reason="no local RELLIS-3D sequences present")
def test_recovered_ring_is_monotonically_decreasing_in_elevation():
    """Ring 0 = highest elevation, ring 63 = lowest, matching
    perception.range_image's own row-assignment convention when a real
    ring field is available."""
    seq = SEQUENCES_AVAILABLE[0]
    sweep = load_rellis_sweep(RELLIS_ROOT / seq, frame_idx=0)
    table = aggregate_beam_elevation_table([sweep.xyz], n_beams=64, min_valid_per_beam=1)
    present = np.nonzero(~np.isnan(table))[0]
    values = table[present]
    assert np.all(np.diff(values) < 0), "elevation must strictly decrease as ring index increases"
