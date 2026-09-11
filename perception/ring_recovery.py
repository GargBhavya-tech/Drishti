"""
perception/ring_recovery.py

Supports Ticket #6 (THE GATE) for RELLIS-3D's Ouster OS1-64 stream.

`perception/rellis_loader.py` reports `ring = -1` throughout, because
RELLIS-3D's KITTI-format `.bin` files carry no per-point ring field (see
that module's own docstring). But the gate (`eval/point_distribution.py`)
needs points grouped by ring to measure within-ring spacing at all --
"measuring nearest-neighbour over the whole cloud finds the neighbour in
the adjacent ring, not the adjacent azimuth sample" (Build Map Ticket
#6's own explicit warning). Without ring, the gate cannot run on this
dataset.

**Finding, confirmed against real downloaded data (2026-09-11):** RELLIS-
3D's Ouster `.bin` dumps are NOT a sparse point list. Every frame checked
across sequences 00000/00002/00004 (9 frames spot-checked, later 155
frames aggregated) has EXACTLY 131,072 = 2048 x 64 points -- the
sensor's full (azimuth_tick, beam) raster, flattened row-major, WITH
invalid/no-return points included as an explicit (0, 0, 0) placeholder
rather than omitted. This means the array position modulo 64 directly
recovers ring, with no statistical recovery (clustering, elevation
histograms, etc.) needed at all:

    ring = index % 64

Verified, not assumed: the elevation angle at a fixed position-within-
group-of-64 is IDENTICAL to float32 precision (~1e-6 degrees observed)
across different frames AND different sequences -- i.e. position-in-
group is a fixed physical property of the sensor (which beam fired),
not an artifact of any one frame's geometry. See
`tests/test_ring_recovery.py::test_recovered_ring_matches_a_fixed_per_position_elevation_on_real_data`
for the machine-checked form of this claim (skipped if the real dataset
isn't present locally).

This is deliberately narrow: it depends on RELLIS-3D's specific raw dump
preserving native scan order, which is a property of THIS dataset's
KITTI-format export, not a general LiDAR fact. It is not a substitute
for a real `ring` field shipped by the format; it is what to do when one
isn't shipped but the native order still is.
"""

from __future__ import annotations

import numpy as np


def recover_ring_from_scan_order(n_points: int, n_beams: int = 64) -> np.ndarray:
    """RELLIS-3D Ouster OS1-64 `.bin` dumps only: `ring = index % n_beams`,
    per this module's docstring. Raises rather than guessing if the point
    count isn't an exact multiple of `n_beams` -- that would mean this
    frame's file is not the fixed (azimuth_tick, beam) raster this
    function assumes (e.g. a partial/corrupted file, or a different
    dataset entirely), and silently returning a wrong ring assignment
    would poison every downstream measurement (Bible Part 3's "if
    measured spacing does not sit on those curves, your sensor constants
    are wrong and every number downstream is wrong" -- the same
    discipline applies to what feeds that measurement).

    Ring 0 = the highest-elevation beam (confirmed in this build's own
    measurement: position 0 sits at ~+17 degrees, position 63 at ~-16.4
    degrees, monotonically decreasing) -- matching the convention
    `perception.range_image`'s row assignment already uses when a real
    `ring` field IS available (row index increases as elevation
    decreases).
    """
    if n_points % n_beams != 0:
        raise ValueError(
            f"n_points={n_points} is not a multiple of n_beams={n_beams} -- "
            f"this is not the fixed (azimuth_tick, beam) raster "
            f"recover_ring_from_scan_order assumes for RELLIS-3D's Ouster "
            f"OS1-64 .bin dumps (see module docstring). Do not guess a ring "
            f"assignment for a file shaped like this."
        )
    return (np.arange(n_points, dtype=np.int64) % n_beams).astype(np.int16)


def aggregate_beam_elevation_table(
    xyz_frames, n_beams: int = 64, min_valid_per_beam: int = 5
) -> np.ndarray:
    """Real per-beam elevation angle (radians), averaged over every
    supplied frame's real (non-zero-range) points at that beam position --
    a single frame under-samples the top few beams (they rarely return
    anything off-road; observed as low as 0 valid points at ring 0-2 in
    one frame), so this is meant to be called across MANY frames (ideally
    spanning multiple sequences, to also confirm the table doesn't drift
    with terrain) to get a complete table.

    `xyz_frames`: iterable of (N, 3) sensor-frame point arrays, each with
    N a multiple of `n_beams` (i.e. raw RELLIS Ouster frames, not a
    ground-filtered or otherwise reduced subset -- ring position is only
    meaningful against the original raster).

    Returns (n_beams,) float64 radians, NaN at any beam with fewer than
    `min_valid_per_beam` real returns across ALL supplied frames combined
    -- an explicit gap rather than a number from too little evidence.
    """
    sums = np.zeros(n_beams, dtype=np.float64)
    counts = np.zeros(n_beams, dtype=np.int64)

    for xyz in xyz_frames:
        xyz = np.asarray(xyz, dtype=np.float64)
        n = xyz.shape[0]
        ring = recover_ring_from_scan_order(n, n_beams=n_beams)
        r = np.linalg.norm(xyz, axis=1)
        valid = r > 1e-6
        if not np.any(valid):
            continue
        elevation = np.arcsin(np.clip(xyz[valid, 2] / r[valid], -1.0, 1.0))
        ring_valid = ring[valid]
        # Vectorised per-beam sum/count via bincount rather than a
        # Python loop over n_beams per frame -- matches this codebase's
        # own "no Python loop over points" discipline (Ticket #18).
        sums += np.bincount(ring_valid, weights=elevation, minlength=n_beams)
        counts += np.bincount(ring_valid, minlength=n_beams)

    table = np.full(n_beams, np.nan, dtype=np.float64)
    enough = counts >= min_valid_per_beam
    table[enough] = sums[enough] / counts[enough]
    return table
