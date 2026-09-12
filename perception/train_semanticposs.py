"""
perception/train_semanticposs.py

Fine-tune FusionSegNet on real SemanticPOSS data (Hesai Pandar40P,
Peking University campus). Same pattern as perception/train_nuscenes.py
(Bible Part G.7): a deliberately SEPARATE script from perception/train.py
rather than a --dataset flag bolted onto it, importing and reusing
train.py's own checkpoint/loss/optimizer machinery rather than touching
its training loop while checkpoints_multi_v3 (the RELLIS retrain) is
running.

Dataset size: 6 sequences, ~2,988 real frames total (488-500 per
sequence) -- an order of magnitude more than nuScenes-mini's 404
lidarseg keyframes, but still far short of RELLIS-3D's 11,522. Runs a
from-scratch-CAPABLE recipe by default (more data than nuScenes-mini),
but still defaults to --init-from-checkpoint for a faster, more
comparable result against the other two real-data experiments in the
Bible (RELLIS Run #2/_v2/_v3, nuScenes fine-tune).

Safe to run ALONGSIDE checkpoints_multi_v3 for the same reason the
nuScenes fine-tune was (see that script's own docstring): `_v3` runs at
~6% GPU utilization, CPU/disk-bound. This job's own footprint (batch
size 2-4, 1-2 workers) is small enough to share the GPU without memory
contention; CPU/disk I/O is the real shared resource, kept modest here
by a conservative default worker count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from perception.frame_cache import FrameCache

from perception.input_tensor import (
    N_CHANNELS,
    _raw_channels,
    compute_channel_stats,
    ground_prior_channel_from_points,
    load_stats,
    save_stats,
)
from perception.losses import DrishtiSegLoss
from perception.segnet import (
    STEM_CONV_STATE_DICT_KEY,
    FusionSegNet,
    N_CLASSES_DEFAULT,
    expand_stem_conv_for_checkpoint,
    remap_legacy_conv_keys,
)
from perception.semanticposs_seg_dataset import (
    SemanticPossSegDataset,
    _load_semanticposs_frame,
    build_semanticposs_splits,
)
from perception.train import (
    compute_class_weights,
    confusion_matrix_update,
    load_checkpoint_if_exists,
    per_class_iou,
    save_checkpoint,
)
from sensor.sensor_model import load_sensor_config


def compute_class_pixel_counts_semanticposs(items, sm, n_classes: int, sample_every: int = 1):
    """SemanticPOSS analogue of perception.train.compute_class_pixel_counts."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for sequence_dir, frame_id in items[::sample_every]:
        img, _, target = _load_semanticposs_frame(sequence_dir, frame_id, sm)
        valid_target = target[img.valid_mask]
        for c in range(n_classes):
            counts[c] += int((valid_target == c).sum())
    return counts


def train_semanticposs(
    sequence_dirs,
    sensor_config_path: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2e-4,
    num_workers: int = 2,
    device: Optional[str] = None,
    max_stats_frames: int = 30,
    init_from_checkpoint: Optional[str] = None,
    use_class_weights: bool = True,
    focal_gamma: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> None:
    if batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (got {batch_size}) -- ASPP's global-pool "
            f"branch + BatchNorm2d cannot train on a single sample per batch."
        )
    sequence_dirs = [Path(p) for p in sequence_dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sm = load_sensor_config(sensor_config_path)

    train_items, val_items, per_seq_counts = build_semanticposs_splits(sequence_dirs)
    seq_names = ", ".join(f"{d.name} ({n} frames)" for d, n in per_seq_counts)
    print(
        f"Training on {len(train_items)} SemanticPOSS frames [{seq_names}] "
        f"(held out {len(val_items)} frames total, last 15% by frame-id, per sequence)"
    )

    stats_path = out_dir / "channel_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"Loaded channel stats from {stats_path} (not recomputed)")
    else:
        raw_stacks = []
        step = max(1, len(train_items) // max_stats_frames)
        for sequence_dir_i, frame_id in train_items[::step][:max_stats_frames]:
            img, ground, _ = _load_semanticposs_frame(sequence_dir_i, frame_id, sm)
            raw_stacks.append(_raw_channels(img, ground_prior_channel_from_points(img, ground)))
        stats = compute_channel_stats(raw_stacks)
        save_stats(stats, stats_path)
        print(f"Computed channel stats from {len(raw_stacks)} frames, saved to {stats_path}")

    class_counts = compute_class_pixel_counts_semanticposs(
        train_items, sm, N_CLASSES_DEFAULT, sample_every=max(1, len(train_items) // 100)
    )
    print("Per-class pixel counts (sampled SemanticPOSS train frames):")
    zero_classes = []
    for c, count in enumerate(class_counts):
        print(f"  class {c}: {count}")
        if count == 0:
            zero_classes.append(c)
    if zero_classes:
        print(
            f"WARNING: classes {zero_classes} have ZERO pixels in the sampled training data -- "
            f"they cannot learn anything and will report IoU=NaN. Check before burning GPU hours."
        )

    train_cache = FrameCache(Path(cache_dir) / "semanticposs_train") if cache_dir else None
    val_cache = FrameCache(Path(cache_dir) / "semanticposs_val") if cache_dir else None
    if cache_dir:
        print(
            f"Frame caching ENABLED at {cache_dir} -- SemanticPOSS's full train set is real disk "
            f"weight (~2,540 frames); FrameCache's own disk-budget check will refuse rather than "
            f"overrun if this run's server doesn't have enough free space (see "
            f"perception/frame_cache.py's own docstring)."
        )
    train_ds = SemanticPossSegDataset(train_items, sm, stats, is_train=True, cache=train_cache)
    val_ds = SemanticPossSegDataset(val_items, sm, stats, is_train=False, cache=val_cache)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)

    ckpt_path = out_dir / "checkpoint.pt"
    if init_from_checkpoint and not ckpt_path.exists():
        init_ckpt = torch.load(init_from_checkpoint, map_location=device)
        init_state = init_ckpt["model_state"]
        init_state = remap_legacy_conv_keys(init_state, set(model.state_dict().keys()))
        old_in_channels = init_state[STEM_CONV_STATE_DICT_KEY].shape[1]
        if old_in_channels != N_CHANNELS:
            print(
                f"Checkpoint stem conv has {old_in_channels} input channels; current "
                f"input_tensor.N_CHANNELS is {N_CHANNELS} -- expanding the stem conv "
                f"(old channels' weights preserved, new channels zero-initialised)."
            )
            init_state = expand_stem_conv_for_checkpoint(init_state, N_CHANNELS)
        model.load_state_dict(init_state)
        print(f"Initialised model weights from {init_from_checkpoint} (fresh optimizer/scheduler/epoch count)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    try:
        scaler = torch.amp.GradScaler(device if device != "cpu" else "cpu", enabled=(device == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    class_weight = compute_class_weights(class_counts, N_CLASSES_DEFAULT).to(device) if use_class_weights else None
    if use_class_weights:
        print("Class weights (secondary CE term only, mean-normalised inverse-sqrt-frequency):")
        for c, w in enumerate(class_weight.tolist()):
            print(f"  class {c}: {w:.3f}")
    loss_fn = DrishtiSegLoss(class_weight=class_weight, focal_gamma=focal_gamma).to(device)

    start_epoch = load_checkpoint_if_exists(ckpt_path, model, optimizer, scheduler, scaler, device)
    if start_epoch > 0:
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    majority_class = int(np.argmax(class_counts))
    print(
        f"Majority-class baseline: class {majority_class} "
        f"({class_counts[majority_class] / max(1, class_counts.sum()):.1%} of sampled pixels)"
    )

    epoch_durations = []
    training_log_path = out_dir / "training_log.jsonl"
    best_miou = -1.0
    best_epoch = -1

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
            f"[SemanticPOSS] Epoch {epoch}/{epochs - 1}: train loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches) "
            f"-- avg {avg_epoch_time:.1f}s/epoch, ETA {eta_s / 60:.1f} min ({epochs_left} epochs left)"
        )

        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler)
        save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", epoch, model, optimizer, scheduler, scaler)

        cm = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
        model.eval()
        val_running_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            for x, target, valid in val_loader:
                x_d, target_d, valid_d = x.to(device), target.to(device), valid.to(device)
                logits = model(x_d)
                val_loss = loss_fn(logits, target_d, valid_d)
                val_running_loss += val_loss.item()
                val_n_batches += 1
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                confusion_matrix_update(cm, pred, target.numpy()[0], valid.numpy()[0], N_CLASSES_DEFAULT)

        val_loss_avg = val_running_loss / max(1, val_n_batches)
        ious = per_class_iou(cm)
        miou = float(np.nanmean(ious))
        current_lr = scheduler.get_last_lr()[0]
        print(f"[SemanticPOSS] Epoch {epoch}: val loss={val_loss_avg:.4f} val mIoU={miou:.4f} lr={current_lr:.2e}")
        for c, iou in enumerate(ious):
            print(f"  class {c} IoU: {iou if not np.isnan(iou) else 'n/a (no pixels)'}")

        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", epoch, model, optimizer, scheduler, scaler)

        with open(training_log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "val_loss": val_loss_avg,
                        "val_miou": miou,
                        "per_class_iou": [None if np.isnan(v) else float(v) for v in ious],
                        "lr": current_lr,
                        "epoch_time_s": elapsed,
                        "is_best": is_best,
                        "best_epoch_so_far": best_epoch,
                        "best_miou_so_far": best_miou,
                    }
                )
                + "\n"
            )

    print(f"SemanticPOSS training complete. Best val mIoU {best_miou:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/fine-tune FusionSegNet on real SemanticPOSS data")
    parser.add_argument("--sequence-dir", nargs="+", default=[f"data/semanticposs/extracted/dataset/sequences/{i:02d}" for i in range(6)])
    parser.add_argument("--sensor-config", default="configs/sensor_pandar40p.yaml")
    parser.add_argument("--out-dir", default="checkpoints_semanticposs")
    parser.add_argument("--init-from-checkpoint", default="checkpoints_multi_v2/best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Enable on-disk frame caching under this directory (see perception/frame_cache.py). "
             "Off by default -- only helps for multi-epoch runs.",
    )
    args = parser.parse_args()

    train_semanticposs(
        sequence_dirs=args.sequence_dir,
        sensor_config_path=args.sensor_config,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        init_from_checkpoint=args.init_from_checkpoint,
        use_class_weights=not args.no_class_weights,
        focal_gamma=args.focal_gamma,
        cache_dir=args.cache_dir,
    )
