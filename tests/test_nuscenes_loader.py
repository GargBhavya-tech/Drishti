"""
Ticket #2 tests. Needs real nuScenes-mini data on disk -- set
NUSCENES_DATAROOT to the extracted v1.0-mini folder. Skipped (not failed)
if that's not set, so `pytest -q` from a fresh clone still passes cleanly
for the parts of the build that don't need a dataset.
"""

import os

import numpy as np
import pytest

pytest.importorskip("nuscenes")

from nuscenes.nuscenes import NuScenes

from perception.nuscenes_loader import load_nuscenes_sweep

DATAROOT = os.environ.get("NUSCENES_DATAROOT")

pytestmark = pytest.mark.skipif(
    not DATAROOT, reason="NUSCENES_DATAROOT not set -- see README for how to get nuScenes-mini"
)


@pytest.fixture(scope="module")
def nusc():
    return NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)


@pytest.fixture(scope="module")
def sample_tokens(nusc):
    scene = nusc.scene[0]
    tokens = []
    sample_token = scene["first_sample_token"]
    while sample_token and len(tokens) < 3:
        tokens.append(sample_token)
        sample_token = nusc.get("sample", sample_token)["next"]
    return tokens


def test_three_sweeps_load_and_look_right(nusc, sample_tokens):
    assert len(sample_tokens) == 3
    for tok in sample_tokens:
        sweep = load_nuscenes_sweep(nusc, tok)
        assert sweep.xyz.shape[1] == 3
        # nuScenes HDL-32E sweeps are typically in the tens of thousands of points.
        assert 15_000 < sweep.xyz.shape[0] < 60_000, (
            f"N={sweep.xyz.shape[0]} is outside the expected nuScenes range -- "
            f"check the 5-field reshape didn't silently misparse the buffer."
        )


def test_ring_spans_32_beams(nusc, sample_tokens):
    sweep = load_nuscenes_sweep(nusc, sample_tokens[0])
    assert sweep.ring.min() == 0
    assert sweep.ring.max() == 31


def test_world_roundtrip(nusc, sample_tokens):
    sweep = load_nuscenes_sweep(nusc, sample_tokens[0])
    xyz_h = np.hstack([sweep.xyz, np.ones((sweep.xyz.shape[0], 1))]).astype(np.float64)
    world = (sweep.T_world @ xyz_h.T).T
    back = (np.linalg.inv(sweep.T_world) @ world.T).T
    residual = np.abs(back[:, :3] - sweep.xyz).max()
    assert residual < 1e-6


def test_bird_eye_view_looks_like_a_disc(nusc, sample_tokens, tmp_path):
    """Not a numeric assertion -- per the ticket, this is the 'plot it and
    look at it' check. Saves a PNG so it can actually be looked at rather
    than only asserted on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = load_nuscenes_sweep(nusc, sample_tokens[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sweep.xyz[:, 0], sweep.xyz[:, 1], s=0.5)
    ax.set_aspect("equal")
    ax.set_title("Ticket #2 checkpoint: bird's-eye view, one nuScenes sweep")
    out = tmp_path / "sweep_bev.png"
    fig.savefig(out, dpi=120)
    assert out.exists()
