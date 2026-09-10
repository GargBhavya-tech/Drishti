"""
perception/rellis_loader.py

Ticket #3 — RELLIS-3D loader -> canonical Sweep.

RELLIS-3D deliberately ships its point clouds in "SemanticKITTI format" so
existing KITTI-style tooling works with minimal change (confirmed against
the dataset repo's own directory listing: `os1_cloud_node_kitti_bin/`,
`os1_cloud_node_semantickitti_label_id/`, `calib.txt`, `poses.txt`).
This loader assumes that format. What is NOT yet confirmed against an
actually-downloaded sequence (Ticket #3's download hasn't run in this
environment) is called out below -- treat those as "probably right,
verify against the first real file you open" rather than settled.

Expected layout for one sequence directory:

    <seq>/
      os1_cloud_node_kitti_bin/00000.bin, 00001.bin, ...   (Ouster OS1-64)
      os1_cloud_node_semantickitti_label_id/00000.label, ...
      vel_cloud_node_kitti_bin/...                          (Velodyne Ultra Puck, 32ch)
      vel_cloud_node_semantickitti_label_id/...
      calib.txt
      poses.txt

Watch out (same class of bug as the nuScenes loader, different shape):

1. KITTI-format .bin files are (x, y, z, intensity) float32 -- FOUR
   fields, not five like nuScenes. Do not reuse the nuScenes reshape
   constant here; that is exactly the kind of copy-paste bug a model
   produces when asked to "write another dataset loader like the last one".
2. `poses.txt` in the standard KITTI/SemanticKITTI convention holds one
   line per scan, 12 space-separated floats = the first 3 rows of a 4x4
   pose matrix in row-major order (the last row is implicitly
   [0, 0, 0, 1]). This loader assumes that convention. VERIFY against the
   real file once #3's download lands -- if the row count doesn't match
   the frame count 1:1, this assumption is wrong for this dataset and the
   loader needs the actual format substituted in.
3. RELLIS-3D ships two sensors per sequence (os1 = 64ch Ouster,
   vel = 32ch Velodyne Ultra Puck). `load_rellis_sweep` defaults to the
   Ouster stream (`sensor_id='ouster_os1_64'`) since that's the config
   this build targets -- pass `sensor_stream='vel'` for the Velodyne one.
4. `ring` is not present in the KITTI-format .bin -- unlike nuScenes,
   there is no per-point beam index shipped. Ticket #6 (the gate) will
   need to recover ring assignment some other way (e.g. from elevation
   angle order within an azimuth sweep) if per-ring statistics are needed
   for this dataset; this loader reports `ring = -1` throughout, which is
   the documented "unavailable" sentinel in the Sweep spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.sweep import Sweep

KITTI_POINT_FIELDS = 4  # x, y, z, intensity


def _read_poses(poses_path: Path) -> np.ndarray:
    """Return (n_frames, 4, 4) array of sensor-to-world poses, KITTI
    convention: 12 floats/line = first 3 rows of the 4x4, last row implicit."""
    rows = np.loadtxt(poses_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    n = rows.shape[0]
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, :3, :4] = rows.reshape(n, 3, 4)
    return poses


def load_rellis_sweep(
    sequence_dir: str | Path,
    frame_idx: int,
    sensor_stream: str = "os1",
    frame_period_s: float = 0.1,
) -> Sweep:
    """Load one frame from a RELLIS-3D sequence directory.

    sensor_stream: 'os1' (64-channel Ouster, default) or 'vel'
                   (32-channel Velodyne Ultra Puck).
    frame_period_s: used only to synthesise a timestamp, since RELLIS-3D's
                    per-frame timestamps are not confirmed in this loader --
                    replace with real timestamps once #3 confirms the format.
    """
    sequence_dir = Path(sequence_dir)
    prefix = {"os1": "os1_cloud_node_kitti_bin", "vel": "vel_cloud_node_kitti_bin"}[sensor_stream]
    sensor_id = {"os1": "ouster_os1_64", "vel": "velodyne_ultra_puck_32"}[sensor_stream]

    bin_path = sequence_dir / prefix / f"{frame_idx:06d}.bin"
    raw = np.fromfile(str(bin_path), dtype=np.float32)
    if raw.size % KITTI_POINT_FIELDS != 0:
        raise ValueError(
            f"{bin_path}: point buffer size {raw.size} is not a multiple of "
            f"{KITTI_POINT_FIELDS} -- this file is not (x,y,z,intensity) "
            f"float32 as the KITTI-format convention assumes. RELLIS-3D's "
            f"actual layout needs re-checking against the downloaded data."
        )
    points = raw.reshape(-1, KITTI_POINT_FIELDS)

    xyz = points[:, 0:3].astype(np.float32)
    intensity_raw = points[:, 3].astype(np.float32)
    # KITTI-format intensity is typically already ~0..1; do not blindly
    # apply the nuScenes /255 normalisation. Verify against real data --
    # if intensity_raw.max() is > ~1.5 the /255 form is needed instead.
    intensity = intensity_raw

    ring = np.full(xyz.shape[0], -1, dtype=np.int16)  # not shipped, see docstring

    poses_path = sequence_dir / "poses.txt"
    if poses_path.exists():
        poses = _read_poses(poses_path)
        if frame_idx >= poses.shape[0]:
            raise IndexError(
                f"frame_idx {frame_idx} >= {poses.shape[0]} poses in {poses_path}"
            )
        T_world = poses[frame_idx]
    else:
        # poses.txt not present (e.g. running against a partial/manual download) --
        # fall back to identity rather than fail, but this Sweep's T_world is
        # then NOT meaningful for anything that needs world coordinates.
        T_world = np.eye(4, dtype=np.float64)

    return Sweep(
        xyz=xyz,
        intensity=intensity,
        ring=ring,
        timestamp=frame_idx * frame_period_s,
        T_world=T_world,
        sensor_id=sensor_id,
    )
