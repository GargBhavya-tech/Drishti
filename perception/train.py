"""
perception/train.py

Ticket #30 -- train FusionSegNet on RELLIS-3D sequence 00004 (preferred
over nuScenes-mini per this ticket's own spec, since #3 has landed).

AdamW + OneCycleLR + mixed precision. Checkpoints every epoch to
`--out-dir` and resumes automatically if a checkpoint is already there --
Ticket #30 "Watch out": "Colab disconnects. ... Do not run a 6-hour job
that loses everything at hour 5." (This build runs on a college GPU
server over SSH instead of Colab -- same risk, same fix: checkpoint to
local disk every epoch, resumable.)

Val split: the LAST ~15% of frames by index, not a random/shuffled split
-- Ticket #30's own warning: "RELLIS-3D ships one continuous route per
sequence... do not shuffle-split a continuous drive or you leak
near-duplicate adjacent frames into val."

Reports per-class IoU (not just mean) and compares against a majority-
class baseline, and states the training-set size explicitly -- all
required by Ticket #30's own "Test"/"Done when".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from perception.ground_prior import compute_ground_prior
from perception.input_tensor import (
    ChannelStats,
    assemble_input_tensor,
    compute_channel_stats,
    load_stats,
    save_stats,
    _raw_channels,
    ground_prior_channel_from_points,
)
from perception.losses import DrishtiSegLoss
from perception.range_image import project_to_range_image
from perception.rellis_loader import load_rellis_labels, load_rellis_sweep
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import rellis_label_ids_to_drishti
from sensor.sensor_model import SensorConfig, load_sensor_config

VAL_FRACTION = 0.15  # last 15% of frames by index, contiguous -- Ticket #30's own spec


def split_frames(n_frames: int, val_fraction: float = VAL_FRACTION):
    n_val = max(1, int(round(n_frames * val_fraction)))
    n_train = n_frames - n_val
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_frames))
    return train_idx, val_idx


def _load_frame(sequence_dir: Path, frame_idx: int, sm: SensorConfig):
    sweep = load_rellis_sweep(sequence_dir, frame_idx)
    raw_labels = load_rellis_labels(sequence_dir, frame_idx)
    drishti_labels = rellis_label_ids_to_drishti(raw_labels)

    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

    target = np.zeros((img.H, img.W), dtype=np.int64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    target[touched] = drishti_labels[src]

    return img, ground, target


class RellisSegDataset(Dataset):
    """One item = one frame: (input_tensor (9,H,W), target (H,W) int64,
    valid_mask (H,W) bool)."""

    def __init__(self, sequence_dir: Path, frame_indices, sm: SensorConfig, stats: ChannelStats):
        self.sequence_dir = Path(sequence_dir)
        self.frame_indices = list(frame_indices)
        self.sm = sm
        self.stats = stats

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, i: int):
        frame_idx = self.frame_indices[i]
        img, ground, target = _load_frame(self.sequence_dir, frame_idx, self.sm)
        tensor = assemble_input_tensor(img, ground, self.stats)
        return (
            torch.from_numpy(tensor).float(),
            torch.from_numpy(target).long(),
            torch.from_numpy(img.valid_mask).bool(),
        )


def compute_class_pixel_counts(sequence_dir: Path, frame_indices, sm: SensorConfig, n_classes: int, sample_every: int = 1):
    """Ticket #30 'Watch out': check rare classes aren't collapsing to
    zero IoU before burning hours -- pixel counts per class over (a
    sample of) the training split, printed before training starts."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for frame_idx in frame_indices[::sample_every]:
        img, _, target = _load_frame(sequence_dir, frame_idx, sm)
        valid_target = target[img.valid_mask]
        for c in range(n_classes):
            counts[c] += int((valid_target == c).sum())
    return counts


def confusion_matrix_update(cm: np.ndarray, pred: np.ndarray, target: np.ndarray, valid: np.ndarray, n_classes: int):
    p = pred[valid]
    t = target[valid]
    idx = t * n_classes + p
    binc = np.bincount(idx, minlength=n_classes * n_classes)
    cm += binc.reshape(n_classes, n_classes)


def per_class_iou(cm: np.ndarray):
    n = cm.shape[0]
    ious = np.full(n, np.nan)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    return ious


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, scaler) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        },
        path,
    )


def load_checkpoint_if_exists(path: Path, model, optimizer, scheduler, scaler, device) -> int:
    """Returns the epoch to RESUME from (0 if no checkpoint)."""
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1


def train(
    sequence_dir: str,
    sensor_config_path: str,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 3e-4,
    num_workers: int = 2,
    device: Optional[str] = None,
    max_stats_frames: int = 30,
) -> None:
    if batch_size < 2:
        # ASPP's global-average-pool branch (perception/segnet.py) collapses
        # spatial size to 1x1; with batch_size=1 that leaves exactly one
        # value per channel, which BatchNorm2d cannot compute training-mode
        # statistics from (raises "Expected more than 1 value per channel").
        # Not specific to this project's changes -- inherent to any
        # batch_size=1 + BatchNorm + global-pool combination. Caught here
        # with a clear message instead of letting it surface as a cryptic
        # PyTorch error partway through the first training step.
        raise ValueError(
            f"batch_size must be >= 2 (got {batch_size}) -- ASPP's global-pool "
            f"branch + BatchNorm2d cannot train on a single sample per batch."
        )
    sequence_dir = Path(sequence_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sm = load_sensor_config(sensor_config_path)

    bin_dir = sequence_dir / "os1_cloud_node_kitti_bin"
    n_frames = len(list(bin_dir.glob("*.bin")))
    train_idx, val_idx = split_frames(n_frames)
    print(f"Training on {len(train_idx)} frames from RELLIS-3D {sequence_dir.name} "
          f"(held out {len(val_idx)} frames, last {VAL_FRACTION:.0%} by index)")

    stats_path = out_dir / "channel_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"Loaded channel stats from {stats_path} (not recomputed)")
    else:
        raw_stacks = []
        for frame_idx in train_idx[:: max(1, len(train_idx) // max_stats_frames)][:max_stats_frames]:
            img, ground, _ = _load_frame(sequence_dir, frame_idx, sm)
            raw_stacks.append(_raw_channels(img, ground_prior_channel_from_points(img, ground)))
        stats = compute_channel_stats(raw_stacks)
        save_stats(stats, stats_path)
        print(f"Computed channel stats from {len(raw_stacks)} frames, saved to {stats_path}")

    class_counts = compute_class_pixel_counts(
        sequence_dir, train_idx, sm, N_CLASSES_DEFAULT, sample_every=max(1, len(train_idx) // 50)
    )
    print("Per-class pixel counts (sampled train frames):")
    zero_classes = []
    for c, count in enumerate(class_counts):
        print(f"  class {c}: {count}")
        if count == 0:
            zero_classes.append(c)
    if zero_classes:
        print(f"WARNING: classes {zero_classes} have ZERO pixels in the sampled training data -- "
              f"they cannot learn anything and will report IoU=NaN. Check before burning GPU hours.")

    train_ds = RellisSegDataset(sequence_dir, train_idx, sm, stats)
    val_ds = RellisSegDataset(sequence_dir, val_idx, sm, stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    # torch.amp.GradScaler (device-agnostic) doesn't exist before torch
    # ~2.3; torch.cuda.amp.GradScaler is deprecated in newer torch but is
    # what the actual training machine (torch 2.0.1) has. Try new, fall
    # back to old -- confirmed this matters: the newer-only form crashes
    # on the real GPU server's installed torch version.
    try:
        scaler = torch.amp.GradScaler(device if device != "cpu" else "cpu", enabled=(device == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    loss_fn = DrishtiSegLoss()

    ckpt_path = out_dir / "checkpoint.pt"
    start_epoch = load_checkpoint_if_exists(ckpt_path, model, optimizer, scheduler, scaler, device)
    if start_epoch > 0:
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    majority_class = int(np.argmax(class_counts))
    print(f"Majority-class baseline: class {majority_class} "
          f"({class_counts[majority_class] / max(1, class_counts.sum()):.1%} of sampled pixels)")

    epoch_durations = []
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for x, target, valid in train_loader:
            x, target, valid = x.to(device), target.to(device), valid.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device if device != "cpu" else "cpu", enabled=(device == "cuda")):
                main_logits, aux_logits = model(x)
                loss = loss_fn(main_logits, target, valid, aux_logits=aux_logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        elapsed = time.time() - t0
        epoch_durations.append(elapsed)
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
        epochs_left = epochs - (epoch + 1)
        eta_s = avg_epoch_time * epochs_left
        print(
            f"Epoch {epoch}/{epochs - 1}: train loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches) "
            f"-- avg {avg_epoch_time:.1f}s/epoch, ETA {eta_s / 60:.1f} min ({epochs_left} epochs left)"
        )

        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler)
        save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", epoch, model, optimizer, scheduler, scaler)

        cm = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
        model.eval()
        with torch.no_grad():
            for x, target, valid in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                confusion_matrix_update(cm, pred, target.numpy()[0], valid.numpy()[0], N_CLASSES_DEFAULT)

        ious = per_class_iou(cm)
        miou = float(np.nanmean(ious))
        print(f"Epoch {epoch}: val mIoU={miou:.4f}")
        for c, iou in enumerate(ious):
            print(f"  class {c} IoU: {iou if not np.isnan(iou) else 'n/a (no pixels)'}")

        with open(out_dir / f"val_metrics_epoch{epoch}.json", "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "miou": miou,
                    "per_class_iou": [None if np.isnan(v) else float(v) for v in ious],
                    "train_set_size": len(train_idx),
                    "val_set_size": len(val_idx),
                    "majority_class_baseline_class": majority_class,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket #30 -- train FusionSegNet on RELLIS-3D")
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train(
        sequence_dir=args.sequence_dir,
        sensor_config_path=args.sensor_config,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
    )
