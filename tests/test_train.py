"""
tests/test_train.py

Ticket #30 smoke tests: the training loop mechanics (not a real multi-
hour run -- that's the actual GPU job). Verifies split_frames' contiguous
val cut, and Ticket #30's own explicit requirement: "Training resumes
correctly from a killed session."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.train import load_checkpoint_if_exists, split_frames, train
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_split_is_contiguous_last_fraction_not_shuffled():
    train_idx, val_idx = split_frames(100, val_fraction=0.15)
    assert train_idx == list(range(0, 85))
    assert val_idx == list(range(85, 100))
    # Contiguity: no interleaving -- the whole point is avoiding
    # near-duplicate adjacent frames leaking across the split.
    assert max(train_idx) < min(val_idx)


def _make_fake_sequence(root: Path, n_frames: int = 6) -> Path:
    """A tiny fake RELLIS-3D-shaped sequence: real label IDs (void, grass,
    vehicle) so multiple DRISHTI classes are actually present, matching
    the format perception.rellis_loader expects."""
    seq = root / "00004"
    bin_dir = seq / "os1_cloud_node_kitti_bin"
    label_dir = seq / "os1_cloud_node_semantickitti_label_id"
    bin_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    label_ids = [0, 3, 3, 8]  # void, grass, grass, vehicle -- real RELLIS IDs
    poses_lines = []
    for frame_idx in range(n_frames):
        n_pts = 40
        az = rng.uniform(0, 2 * np.pi, n_pts)
        el = rng.uniform(np.radians(-20), np.radians(2), n_pts)
        r = rng.uniform(3.0, 20.0, n_pts)
        x = (r * np.cos(el) * np.cos(az)).astype(np.float32)
        y = (r * np.cos(el) * np.sin(az)).astype(np.float32)
        z = (r * np.sin(el)).astype(np.float32)
        intensity = rng.uniform(0, 1, n_pts).astype(np.float32)
        pts = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
        pts.tofile(bin_dir / f"{frame_idx:06d}.bin")

        labels = rng.choice(label_ids, size=n_pts).astype(np.uint32)
        labels.tofile(label_dir / f"{frame_idx:06d}.label")

        row = np.eye(4, dtype=np.float64)[:3, :4]
        row[0, 3] = float(frame_idx)
        poses_lines.append(" ".join(f"{v:.6f}" for v in row.flatten()))
    (seq / "poses.txt").write_text("\n".join(poses_lines) + "\n")
    return seq


def test_training_smoke_run_two_epochs(tmp_path):
    seq_dir = _make_fake_sequence(tmp_path)
    out_dir = tmp_path / "checkpoints"

    train(
        sequence_dir=str(seq_dir),
        sensor_config_path=str(CONFIGS / "sensor_hdl32e.yaml"),
        out_dir=str(out_dir),
        epochs=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        max_stats_frames=3,
    )

    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "channel_stats.json").exists()
    assert (out_dir / "val_metrics_epoch0.json").exists()
    assert (out_dir / "val_metrics_epoch1.json").exists()


def test_resumes_correctly_from_a_killed_session(tmp_path):
    """Ticket #30's own explicit test: simulate a kill after epoch 0 by
    only running 1 epoch, then call train() again asking for 2 epochs --
    it must pick up at epoch 1, not restart from epoch 0."""
    """The realistic scenario: a SINGLE logical training run targeting
    epochs=2 gets killed after epoch 0's checkpoint lands but before
    epoch 1 finishes, then is invoked again with the SAME target. (Not
    "resume with a larger epoch count than originally planned" -- that's
    a different, incompatible scenario: OneCycleLR's schedule is tied to
    a fixed total step count decided at construction time, and
    load_state_dict restores that total from the checkpoint, so a
    resumed run cannot silently extend its own horizon.)
    """
    import perception.train as train_module

    seq_dir = _make_fake_sequence(tmp_path)
    out_dir = tmp_path / "checkpoints"

    # Simulate a crash: let epoch 0 complete and checkpoint normally
    # (save_checkpoint is called twice per epoch -- ckpt_path, then the
    # per-epoch copy), then blow up right as epoch 1 would start.
    original_save = train_module.save_checkpoint
    call_count = {"n": 0}

    def crash_after_epoch_0(*args, **kwargs):
        call_count["n"] += 1
        original_save(*args, **kwargs)
        if call_count["n"] == 2:  # both epoch-0 checkpoints have landed
            raise RuntimeError("simulated disconnect")

    train_module.save_checkpoint = crash_after_epoch_0
    try:
        with pytest.raises(RuntimeError, match="simulated disconnect"):
            train(
                sequence_dir=str(seq_dir),
                sensor_config_path=str(CONFIGS / "sensor_hdl32e.yaml"),
                out_dir=str(out_dir),
                epochs=2,
                batch_size=2,
                num_workers=0,
                device="cpu",
                max_stats_frames=3,
            )
    finally:
        train_module.save_checkpoint = original_save

    # The crash lands right after epoch 0's checkpoint saves but BEFORE
    # its validation/metrics step -- so what must be true here is that
    # the resumable checkpoint exists and epoch 1 hasn't happened, not
    # that epoch 0's metrics file was written yet.
    assert (out_dir / "checkpoint.pt").exists()
    assert not (out_dir / "val_metrics_epoch1.json").exists()

    import torch

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    resume_epoch = load_checkpoint_if_exists(
        out_dir / "checkpoint.pt", model, optimizer, scheduler=None, scaler=None, device="cpu"
    )
    assert resume_epoch == 1, "should resume at epoch 1, not restart from 0"

    # Re-invoke with the SAME target (epochs=2) -- must pick up at epoch
    # 1, not redo epoch 0's training. Note: epoch 0's val_metrics file was
    # never written (the crash lands before that step in the crashed
    # run), so it correctly stays absent here too -- the resumed run
    # starts at range(1, 2), it never revisits epoch 0. What must hold is
    # that the checkpointed WEIGHTS survived (the core guarantee) and
    # that training genuinely continues and completes from where it left
    # off, not from scratch.
    train(
        sequence_dir=str(seq_dir),
        sensor_config_path=str(CONFIGS / "sensor_hdl32e.yaml"),
        out_dir=str(out_dir),
        epochs=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        max_stats_frames=3,
    )
    assert (out_dir / "val_metrics_epoch1.json").exists()  # produced by the resumed run
    final_epoch = load_checkpoint_if_exists(
        out_dir / "checkpoint.pt", FusionSegNet(n_classes=N_CLASSES_DEFAULT),
        torch.optim.AdamW(FusionSegNet(n_classes=N_CLASSES_DEFAULT).parameters()),
        scheduler=None, scaler=None, device="cpu",
    )
    assert final_epoch == 2, "checkpoint should now reflect epoch 1 completed (resume point = 2)"
