"""
perception/nuscenes_loader.py

Ticket #2 — nuScenes-mini loader -> canonical Sweep.

Watch out (from the Build Map, restated here so it's next to the code
that has to get it right):

1. nuScenes LiDAR .bin files are (x, y, z, intensity, ring_index) float32
   -- five fields, not four. Reading it as four silently misparses every
   point (everything after the first point is offset by one field).
2. nuScenes intensity is 0-255, not 0-1. This loader normalises it.
3. Pose composition order is easy to invert. The correct order is
   T_world = T_ego_world @ T_sensor_ego -- transform sensor points into
   the ego frame first, then the ego frame into the world frame. Get this
   backwards and a sweep transformed to world will look locally plausible
   (it's still a rigid transform of a real point cloud) but sit in the
   wrong global place -- the kind of bug that survives a casual eyeball
   check of one frame and only shows up once you compare frames.
"""

from __future__ import annotations

import numpy as np
from pyquaternion import Quaternion

from perception.sweep import Sweep

NUSCENES_POINT_FIELDS = 5  # x, y, z, intensity, ring_index


def _pose_to_matrix(translation, rotation_quat_wxyz) -> np.ndarray:
    """nuScenes stores rotation as a quaternion in (w, x, y, z) order.
    Build the 4x4 homogeneous transform it represents."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(rotation_quat_wxyz).rotation_matrix
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def load_nuscenes_sweep(nusc, sample_token: str) -> Sweep:
    """Load one LIDAR_TOP sweep for a nuScenes sample, returning a
    canonical Sweep in the SENSOR frame with T_world filled in.

    `nusc` is a nuscenes.nuscenes.NuScenes instance (from nuscenes-devkit).
    """
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)

    # --- raw points ---------------------------------------------------
    pcl_path = nusc.get_sample_data_path(sd_token)
    raw = np.fromfile(str(pcl_path), dtype=np.float32)
    if raw.size % NUSCENES_POINT_FIELDS != 0:
        raise ValueError(
            f"{pcl_path}: point buffer size {raw.size} is not a multiple of "
            f"{NUSCENES_POINT_FIELDS} -- this file is not "
            f"(x,y,z,intensity,ring) float32 as expected."
        )
    points = raw.reshape(-1, NUSCENES_POINT_FIELDS)

    xyz = points[:, 0:3].astype(np.float32)
    intensity = (points[:, 3] / 255.0).astype(np.float32)   # 0..255 -> 0..1
    ring = points[:, 4].astype(np.int16)

    # --- pose composition: T_world = T_ego_world @ T_sensor_ego --------
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T_sensor_ego = _pose_to_matrix(calib["translation"], calib["rotation"])

    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    T_ego_world = _pose_to_matrix(ego_pose["translation"], ego_pose["rotation"])

    T_world = T_ego_world @ T_sensor_ego

    timestamp_s = sd["timestamp"] / 1e6  # nuScenes timestamps are microseconds

    return Sweep(
        xyz=xyz,
        intensity=intensity,
        ring=ring,
        timestamp=timestamp_s,
        T_world=T_world,
        sensor_id="hdl32e",
    )
