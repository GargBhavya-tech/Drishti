"""
Ticket #3 tests. Two tiers:

1. Synthetic-fixture tests that check the parsing logic (.bin reshape,
   poses.txt parsing) against a small hand-built fake sequence -- these
   run always, no download needed, and catch the loader's own bugs.
2. Real-data tests against an actual downloaded RELLIS-3D sequence --
   skipped unless RELLIS_SEQ_DIR is set, since the download (Ticket #3's
   detachable background step) has not necessarily landed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from perception.rellis_loader import load_rellis_sweep

REAL_SEQ_DIR = os.environ.get("RELLIS_SEQ_DIR")


@pytest.fixture
def fake_sequence(tmp_path):
    """A minimal synthetic RELLIS-3D-shaped sequence: 2 frames, 3 points
    each, identity-ish poses, to exercise the loader's parsing without
    needing the real 7GB download."""
    seq = tmp_path / "00004"
    bin_dir = seq / "os1_cloud_node_kitti_bin"
    bin_dir.mkdir(parents=True)

    for frame_idx in range(2):
        pts = np.array(
            [
                [1.0 + frame_idx, 2.0, 0.5, 0.9],
                [3.0, 4.0, 0.2, 0.4],
                [5.0, -1.0, 1.1, 0.1],
            ],
            dtype=np.float32,
        )
        pts.tofile(bin_dir / f"{frame_idx:06d}.bin")

    # poses.txt: 2 lines, 12 floats each (identity pose, translated by frame_idx in x)
    lines = []
    for frame_idx in range(2):
        row = np.eye(4, dtype=np.float64)[:3, :4]
        row[0, 3] = float(frame_idx)
        lines.append(" ".join(f"{v:.6f}" for v in row.flatten()))
    (seq / "poses.txt").write_text("\n".join(lines) + "\n")

    return seq


def test_fake_sequence_loads(fake_sequence):
    sweep = load_rellis_sweep(fake_sequence, frame_idx=0)
    assert sweep.xyz.shape == (3, 3)
    assert sweep.sensor_id == "ouster_os1_64"
    assert np.all(sweep.ring == -1)  # not shipped in KITTI-format .bin, see loader docstring


def test_pose_parsed_correctly(fake_sequence):
    sweep = load_rellis_sweep(fake_sequence, frame_idx=1)
    # frame_idx=1's synthetic pose translates x by 1.0
    assert sweep.T_world[0, 3] == pytest.approx(1.0)
    assert np.allclose(sweep.T_world[3], [0, 0, 0, 1])


def test_intensity_not_divided_by_255(fake_sequence):
    """Regression guard: this loader must NOT reuse the nuScenes /255
    normalisation -- KITTI-format intensity is already ~0..1."""
    sweep = load_rellis_sweep(fake_sequence, frame_idx=0)
    assert sweep.intensity[0] == pytest.approx(0.9)  # not 0.9/255


def test_missing_poses_falls_back_to_identity(tmp_path):
    seq = tmp_path / "00004_nopose"
    bin_dir = seq / "os1_cloud_node_kitti_bin"
    bin_dir.mkdir(parents=True)
    np.zeros((1, 4), dtype=np.float32).tofile(bin_dir / "000000.bin")
    sweep = load_rellis_sweep(seq, frame_idx=0)
    assert np.allclose(sweep.T_world, np.eye(4))


@pytest.mark.skipif(not REAL_SEQ_DIR, reason="RELLIS_SEQ_DIR not set -- Ticket #3 download not landed")
def test_real_sequence_00004():
    sweep = load_rellis_sweep(REAL_SEQ_DIR, frame_idx=0)
    assert sweep.xyz.shape[0] > 0
    # No N-count assertion here yet -- Ticket #6 (the gate) is what
    # establishes the expected point count and spacing for this sensor,
    # once its d_theta/d_phi are actually measured rather than assumed.
