"""
tests/test_cache_inference.py

Ticket #31 tests, against the real local checkpoint and RELLIS-3D data.
Skipped if either isn't present (matching every other real-artifact
test's pattern in this repo). This is genuinely slow (loading a 70MB
checkpoint + real CPU inference, ~0.5s/frame) -- kept to 2 frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.cache_inference import cache_inference_for_sequence, load_trained_model, per_point_ground_truth
from perception.rellis_loader import load_rellis_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoints_multi_remote" / "checkpoint.pt"
SEQUENCE_DIR = REPO_ROOT / "data" / "rellis" / "00004"
SENSOR_CONFIG = REPO_ROOT / "configs" / "sensor_ouster_os1_64.yaml"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and SEQUENCE_DIR.exists()),
    reason="real trained checkpoint or local RELLIS-3D sequence not present",
)


def test_checkpoint_loads_with_key_remapping():
    """Run #1/#2's checkpoints predate this session's circular-padding
    fix (which renamed some conv parameters) -- must still load."""
    model, epoch = load_trained_model(CHECKPOINT)
    assert epoch == 19
    total_params = sum(p.numel() for p in model.parameters())
    assert 4_000_000 <= total_params <= 9_000_000


def test_cached_labels_align_frame_for_frame_with_sweeps(tmp_path):
    """Ticket #31's own required test: len(labels) == len(points), for
    every frame."""
    manifest = cache_inference_for_sequence(
        checkpoint_path=CHECKPOINT,
        sequence_dir=SEQUENCE_DIR,
        sensor_config_path=SENSOR_CONFIG,
        out_dir=tmp_path,
        frame_indices=[1000, 1001],
        device="cpu",
    )
    assert len(manifest["frames"]) == 2
    for f in manifest["frames"]:
        sweep = load_rellis_sweep(SEQUENCE_DIR, f["frame_idx"])
        labels = np.load(f["npy_path"])
        assert labels.shape[0] == sweep.xyz.shape[0]
        assert f["n_points"] == sweep.xyz.shape[0]


def test_manifest_is_explicit_that_this_is_not_live_inference(tmp_path):
    """Build Map's own warning: 'Presenting cached labels as live
    inference is exactly the overclaim that loses credibility' --
    the manifest must say so itself, not rely on a human remembering."""
    manifest = cache_inference_for_sequence(
        checkpoint_path=CHECKPOINT,
        sequence_dir=SEQUENCE_DIR,
        sensor_config_path=SENSOR_CONFIG,
        out_dir=tmp_path,
        frame_indices=[1000],
        device="cpu",
    )
    assert "not live" in manifest["note"].lower() or "offline" in manifest["note"].lower()

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved == manifest


def test_predicted_labels_are_valid_drishti_class_ids(tmp_path):
    from perception.taxonomy import DrishtiClass

    manifest = cache_inference_for_sequence(
        checkpoint_path=CHECKPOINT,
        sequence_dir=SEQUENCE_DIR,
        sensor_config_path=SENSOR_CONFIG,
        out_dir=tmp_path,
        frame_indices=[1000],
        device="cpu",
    )
    labels = np.load(manifest["frames"][0]["npy_path"])
    valid_ids = set(int(c) for c in DrishtiClass)
    assert set(np.unique(labels).tolist()).issubset(valid_ids)


def test_per_point_ground_truth_matches_sweep_length():
    gt = per_point_ground_truth(SEQUENCE_DIR, frame_idx=1000)
    sweep = load_rellis_sweep(SEQUENCE_DIR, frame_idx=1000)
    assert gt.shape[0] == sweep.xyz.shape[0]
