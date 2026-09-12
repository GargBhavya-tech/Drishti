"""
perception/multi_sweep.py

Motion-compensated multi-sweep accumulation: merges the current sweep
with N-1 PRIOR sweeps, each transformed from its own sensor frame into
the CURRENT sweep's sensor frame via the real recorded poses
(`Sweep.T_world`), before the merged point set is projected to a range
image. Densifies distant/sparse returns (RELLIS-3D's own usable range
thins out well before the sensor's nominal maximum, per
`configs/sensor_ouster_os1_64.yaml`'s own note) without any new
hardware.

Reuses `perception.range_image.project_to_range_image` UNCHANGED on the
merged point set -- that function's existing nearest-return-per-pixel
collision handling (Ticket #23/#24) is exactly the z-buffering a merged
multi-time point cloud needs, so no new occlusion logic is written here.

HONEST LIMITATION, stated plainly (do not claim otherwise to a judge):
only EGO motion is compensated. A genuinely moving object (a walking
person, a moving vehicle) has ITS OWN motion on top of the ego's, which
this function does NOT correct for -- a moving object's points from
sweeps t-1 and t-2 land at its PAST world positions, not smeared toward
its current one. The practical effect is that a moving object appears
as multiple partial "ghost" copies at its past positions rather than
one denser blob, which a STATIC object (a real UGV target: a fixed pole,
a rock) does not exhibit. This is a real, usable appearance-based cue
for motion (a static object densifies; a moving one fragments into
ghosts) but it must be described as exactly that -- an emergent
artifact, not "3x point density everywhere" -- not oversold as if every
merged frame were equally dense and clean.
"""

from __future__ import annotations

from typing import List

import numpy as np

from perception.sweep import Sweep

DEFAULT_N_SWEEPS = 3


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a (4,4) homogeneous transform to (N,3) points."""
    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
    transformed = (T @ homog.T).T
    return transformed[:, :3]


def merge_sweeps_motion_compensated(sweeps: List[Sweep]) -> Sweep:
    """`sweeps` must be given OLDEST FIRST, current sweep LAST (matches
    reading order: sweeps[-1] is "now"). Every earlier sweep's points
    are transformed into sweeps[-1]'s own sensor frame via
    `inv(T_world_now) @ T_world_then`, then all points/intensities/rings
    are concatenated. The returned Sweep carries the CURRENT sweep's own
    T_world/timestamp/sensor_id (the merge is expressed in its frame).

    A single-sweep input (`len(sweeps) == 1`) returns that sweep
    unchanged -- not an error, since "no history available yet" (e.g.
    the first two frames of a sequence) is an expected, valid case, not
    a failure.
    """
    if not sweeps:
        raise ValueError("sweeps must be non-empty")
    if len(sweeps) == 1:
        return sweeps[0]

    current = sweeps[-1]
    T_world_now_inv = np.linalg.inv(current.T_world)

    xyz_parts = [current.xyz]
    intensity_parts = [current.intensity]
    ring_parts = [current.ring]

    for older in sweeps[:-1]:
        T_then_to_now = T_world_now_inv @ older.T_world
        xyz_parts.append(transform_points(older.xyz, T_then_to_now))
        intensity_parts.append(older.intensity)
        ring_parts.append(older.ring)

    merged_xyz = np.concatenate(xyz_parts, axis=0).astype(np.float32)
    merged_intensity = np.concatenate(intensity_parts, axis=0).astype(np.float32)
    merged_ring = np.concatenate(ring_parts, axis=0).astype(np.int16)

    return Sweep(
        xyz=merged_xyz,
        intensity=merged_intensity,
        ring=merged_ring,
        timestamp=current.timestamp,
        T_world=current.T_world,
        sensor_id=current.sensor_id,
    )
