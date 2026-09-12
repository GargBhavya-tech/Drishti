"""
perception/train_nuscenes.py

Fine-tune FusionSegNet on nuScenes-mini's real lidarseg-labeled urban
scenes. NOT a from-scratch training path -- see
perception.nuscenes_seg_dataset's own docstring for why (404 lidarseg
keyframes vs. RELLIS-3D's 11,522 training frames).

Built specifically to test the Ground Paradox fix (see
DRISHTI_MASTER_BIBLE.md Part D.5/G.1): RELLIS-3D's zero-shot transfer to
nuScenes predicts VEGETATION for nuScenes' flat asphalt, because
RELLIS-3D's own ground truth is ~85% grass/soil. Fine-tuning on a small
amount of REAL labeled asphalt/driveable-surface data tests that
diagnosis directly; re-measuring zero-shot mIoU again cannot.

Deliberately a SEPARATE script from perception/train.py rather than a
--dataset flag bolted onto it: checkpoints_multi_v3 (the RELLIS-3D
CutMix+focal-loss retrain) is running as this is written, and editing
perception/train.py's own training loop is the highest-risk change
possible right now. This script IMPORTS and REUSES train.py's building
blocks (save_checkpoint, load_checkpoint_if_exists, compute_class_
weights, confusion_matrix_update, per_class_iou) rather than
duplicating or risking them, but keeps its own training loop and CLI
entirely separate -- nothing here can affect the RELLIS run's behaviour,
its checkpoint files, or its resumability.

Safe to run ALONGSIDE checkpoints_multi_v3 (checked before launch, not
assumed): `nvidia-smi` during that run shows 856MiB/11.3GiB GPU memory
used and 6% utilisation -- it spends most of its wall-clock time in
CPU-bound data loading (eval/out/checkpoint_latency.json:
load_and_assemble ~447ms vs. model_forward ~385ms per frame, CPU),
leaving 10GB+ of GPU memory and effectively all GPU compute idle. This
job's own footprint (404 frames, batch_size 2, --num-workers 1) is far
smaller than the RELLIS job's (11,522 frames, batch_size 4, 2 workers)
and shares the GPU without contending for memory; the real shared
resource is CPU/disk I/O, kept low here deliberately with a single
worker and a small batch size rather than matching the RELLIS job's own
settings.
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
from perception.nuscenes_seg_dataset import (
    NuscenesSegDataset,
    _load_nuscenes_frame,
    build_nuscenes_scene_splits,
)
from perception.segnet import (
    STEM_CONV_STATE_DICT_KEY,
    FusionSegNet,
    N_CLASSES_DEFAULT,
    expand_stem_conv_for_checkpoint,
    remap_legacy_conv_keys,
)
from perception.taxonomy import build_nuscenes_lidarseg_lut
from perception.train import (
    compute_class_weights,
    confusion_matrix_update,
    load_checkpoint_if_exists,
    per_class_iou,
    save_checkpoint,
)
from sensor.sensor_model import load_sensor_config


def compute_class_pixel_counts_nuscenes(nusc, items, sm, lut, n_classes: int, sample_every: int = 1):
    """nuScenes analogue of perception.train.compute_class_pixel_counts
    -- same "check rare classes aren't zero before burning GPU hours"
    purpose, applied to a much smaller dataset where sample_every=1
    (checking every item) is cheap enough to just do."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for sample_token in items[::sample_every]:
        img, _, target = _load_nuscenes_frame(nusc, sample_token, sm, lut)
        valid_target = target[img.valid_mask]
        for c in range(n_classes):
            counts[c] += int((valid_target == c).sum())
    return counts


def train_nuscenes(
    dataroot: str,
    version: str,
    sensor_config_path: str,
    out_dir: str,
    epochs: int = 8,
    batch_size: int = 2,
    lr: float = 1e-4,
    num_workers: int = 1,
    device: Optional[str] = None,
    max_stats_frames: int = 30,
    init_from_checkpoint: Optional[str] = None,
    use_class_weights: bool = True,
    focal_gamma: Optional[float] = None,
    val_scene_fraction: float = 0.2,
    cache_dir: Optional[str] = None,
) -> None:
    """`init_from_checkpoint`: same semantics as perception.train.train's
    own parameter -- load ONLY model weights (fresh optimizer/scheduler/
    scaler/epoch count) as a fine-tuning starting point. Defaults to
    `checkpoints_multi_v2/best.pt` at the CLI level (a FINISHED RELLIS
    checkpoint) rather than `checkpoints_multi_v3` (still training as
    this is written) -- point this at `_v3`'s own best.pt once it
    finishes, for the strongest available starting point."""
    if batch_size < 2:
        # Same ASPP global-pool + BatchNorm2d constraint as
        # perception.train.train -- restated here rather than imported
        # as a shared check because it's a one-line guard, not shared
        # state that could drift between the two training paths.
        raise ValueError(
            f"batch_size must be >= 2 (got {batch_size}) -- ASPP's global-pool "
            f"branch + BatchNorm2d cannot train on a single sample per batch."
        )
    from nuscenes.nuscenes import NuScenes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sm = load_sensor_config(sensor_config_path)

    print(f"Loading nuScenes {version} from {dataroot} ...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    lut = build_nuscenes_lidarseg_lut(nusc)

    train_items, val_items, per_scene_counts = build_nuscenes_scene_splits(nusc, val_scene_fraction)
    n_val_scenes = max(1, int(round(len(per_scene_counts) * val_scene_fraction)))
    print(
        f"Training on {len(train_items)} nuScenes keyframes from "
        f"{len(per_scene_counts) - n_val_scenes} scenes (held out {len(val_items)} "
        f"frames from {n_val_scenes} WHOLE scenes, not a shuffled split)"
    )

    stats_path = out_dir / "channel_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"Loaded channel stats from {stats_path} (not recomputed)")
    else:
        raw_stacks = []
        step = max(1, len(train_items) // max_stats_frames)
        for sample_token in train_items[::step][:max_stats_frames]:
            img, ground, _ = _load_nuscenes_frame(nusc, sample_token, sm, lut)
            raw_stacks.append(_raw_channels(img, ground_prior_channel_from_points(img, ground)))
        stats = compute_channel_stats(raw_stacks)
        save_stats(stats, stats_path)
        print(f"Computed channel stats from {len(raw_stacks)} frames, saved to {stats_path}")

    class_counts = compute_class_pixel_counts_nuscenes(nusc, train_items, sm, lut, N_CLASSES_DEFAULT)
    print("Per-class pixel counts (ALL nuScenes train frames -- small dataset, checked exhaustively):")
    zero_classes = []
    for c, count in enumerate(class_counts):
        print(f"  class {c}: {count}")
        if count == 0:
            zero_classes.append(c)
    if zero_classes:
        print(
            f"WARNING: classes {zero_classes} have ZERO pixels in the training data -- "
            f"they cannot learn anything and will report IoU=NaN. Check before burning GPU hours."
        )

    train_cache = FrameCache(Path(cache_dir) / "nuscenes_train") if cache_dir else None
    val_cache = FrameCache(Path(cache_dir) / "nuscenes_val") if cache_dir else None
    if cache_dir:
        print(
            f"Frame caching ENABLED at {cache_dir} -- first epoch pays the full compute cost, "
            f"epoch 2+ reads from disk (see perception/frame_cache.py's own docstring)."
        )
    train_ds = NuscenesSegDataset(nusc, train_items, sm, stats, lut, is_train=True, cache=train_cache)
    val_ds = NuscenesSegDataset(nusc, val_items, sm, stats, lut, is_train=False, cache=val_cache)
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
        f"({class_counts[majority_class] / max(1, class_counts.sum()):.1%} of pixels)"
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
            f"[nuScenes] Epoch {epoch}/{epochs - 1}: train loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches) "
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
        print(f"[nuScenes] Epoch {epoch}: val loss={val_loss_avg:.4f} val mIoU={miou:.4f} lr={current_lr:.2e}")
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

    print(f"nuScenes fine-tune complete. Best val mIoU {best_miou:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FusionSegNet on nuScenes-mini (real lidarseg labels)")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sensor-config", default="configs/sensor_hdl32e.yaml")
    parser.add_argument("--out-dir", default="checkpoints_nuscenes_ft")
    parser.add_argument("--init-from-checkpoint", default="checkpoints_multi_v2/best.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--val-scene-fraction", type=float, default=0.2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Enable on-disk frame caching under this directory (see perception/frame_cache.py). "
             "Off by default -- only helps for multi-epoch runs; the first epoch pays the full "
             "compute cost regardless. nuScenes-mini's small size (~730MB estimated raw-stack cache) "
             "fits comfortably even on a nearly-full disk.",
    )
    args = parser.parse_args()

    train_nuscenes(
        dataroot=args.dataroot,
        version=args.version,
        sensor_config_path=args.sensor_config,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        init_from_checkpoint=args.init_from_checkpoint,
        use_class_weights=not args.no_class_weights,
        focal_gamma=args.focal_gamma,
        val_scene_fraction=args.val_scene_fraction,
        cache_dir=args.cache_dir,
    )
