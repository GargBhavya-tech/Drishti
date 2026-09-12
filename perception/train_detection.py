"""
perception/train_detection.py

Trains perception/detection_head.py on REAL nuScenes-mini 3D box
annotations (18,538 boxes across 404 samples, confirmed present on the
server this session) -- the answer to "static obstacles get identified
and classified, but not as discrete instances" (PS gap #2, this
session's deep-research report on lightweight range-view detection).

Backbone handling, deliberately conservative: the shared FusionSegNet
backbone is loaded from an EXISTING, already-trained segmentation
checkpoint (default: checkpoints_joint/best.pt, the multi-domain model)
and fine-tuned at a LOW learning rate alongside the new detection head,
rather than frozen outright or trained from scratch. This is the
practical middle ground between the report's own "L2-SP regularization
prevents catastrophic forgetting" finding (Olber et al. 2025) and this
project's small nuScenes box-training set (404 frames) -- a frozen
backbone risks the detection head having to compensate for features
tuned for a different task; an unconstrained fine-tune on only 404
frames risks forgetting the segmentation quality already achieved on
far larger RELLIS-3D/SemanticPOSS training sets. `--backbone-lr-scale`
implements this: the backbone's own learning rate is this fraction of
the detection head's, defaulting to 0.1 (backbone drifts slowly; head
learns fast) rather than a hard freeze or a hard L2-SP penalty term
(a real, un-taken option for a v2 -- named here rather than silently
assumed unnecessary).

Only nuScenes has box supervision (RELLIS-3D and SemanticPOSS ship none
-- see perception/nuscenes_boxes.py's own docstring), so this script,
unlike train_joint.py, trains on exactly one domain.
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

from perception.detection_head import DetectionHead
from perception.detection_loss import DetectionLoss
from perception.frame_cache import FrameCache
from perception.input_tensor import ChannelStats, load_stats
from perception.nuscenes_detection_dataset import NuscenesDetectionDataset, compute_real_pos_weight
from perception.nuscenes_seg_dataset import build_nuscenes_scene_splits
from perception.segnet import (
    STEM_CONV_STATE_DICT_KEY,
    FusionSegNet,
    N_CLASSES_DEFAULT,
    expand_stem_conv_for_checkpoint,
    remap_legacy_conv_keys,
)
from perception.input_tensor import N_CHANNELS
from sensor.sensor_model import load_sensor_config


def train_detection(
    dataroot: str,
    version: str,
    sensor_config_path: str,
    seg_checkpoint_path: str,
    seg_stats_path: str,
    out_dir: str,
    epochs: int = 15,
    batch_size: int = 4,
    head_lr: float = 3e-4,
    backbone_lr_scale: float = 0.1,
    num_workers: int = 1,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    if batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (got {batch_size}) -- ASPP's global-pool "
            f"branch + BatchNorm2d cannot train on a single sample per batch."
        )
    from nuscenes.nuscenes import NuScenes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sm = load_sensor_config(sensor_config_path)
    stats = load_stats(seg_stats_path)

    print(f"Loading nuScenes {version} from {dataroot} ...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    train_items, val_items, per_scene_counts = build_nuscenes_scene_splits(nusc)
    print(f"nuScenes-mini: {len(train_items)} train / {len(val_items)} val frames (same scene split as the segmentation fine-tune)")

    print("Computing REAL pos_weight from training data (this scans every train frame's real boxes once)...")
    pos_weight = compute_real_pos_weight(nusc, train_items, sm, stats)
    print(f"Measured pos_weight (neg:pos pixel ratio): {pos_weight:.1f}")

    train_cache = FrameCache(Path(cache_dir) / "detection_train") if cache_dir else None
    val_cache = FrameCache(Path(cache_dir) / "detection_val") if cache_dir else None
    if cache_dir:
        print(
            f"Frame caching ENABLED at {cache_dir} -- this dataset's per-frame cost includes "
            f"real 3D box-target projection (perception/nuscenes_boxes.py), the most expensive "
            f"CPU work of any dataset in this project; caching helps this run the most."
        )
    train_ds = NuscenesDetectionDataset(nusc, train_items, sm, stats, cache=train_cache)
    val_ds = NuscenesDetectionDataset(nusc, val_items, sm, stats, cache=val_cache)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    # --- shared backbone, loaded from an EXISTING segmentation checkpoint ---
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)
    seg_ckpt = torch.load(seg_checkpoint_path, map_location=device)
    seg_state = remap_legacy_conv_keys(seg_ckpt["model_state"], set(model.state_dict().keys()))
    old_in_channels = seg_state[STEM_CONV_STATE_DICT_KEY].shape[1]
    if old_in_channels != N_CHANNELS:
        seg_state = expand_stem_conv_for_checkpoint(seg_state, N_CHANNELS)
    model.load_state_dict(seg_state)
    print(f"Loaded shared backbone from {seg_checkpoint_path}")

    head = DetectionHead().to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": head_lr * backbone_lr_scale},
            {"params": head.parameters(), "lr": head_lr},
        ]
    )
    # OneCycleLR -- every OTHER training script in this project
    # (perception/train.py, train_nuscenes.py, train_semanticposs.py,
    # train_joint.py) uses this; this script's first version did not,
    # which is a real, plausible contributor to its slow-to-converge
    # first run (eval/checkpoint_detection_decode.py: even an
    # extremely permissive 0.1 objectness threshold only recovered
    # ~35% of real objects) -- fixed here rather than only "training
    # longer" on the same flat-LR schedule. `max_lr` is a per-group
    # list matching the optimizer's own per-group LRs above, so the
    # backbone and head keep their intended relative learning rates
    # throughout the whole schedule, not just at step 0.
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[head_lr * backbone_lr_scale, head_lr],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    loss_fn = DetectionLoss(pos_weight=pos_weight).to(device)

    training_log_path = out_dir / "training_log.jsonl"
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        head.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for x, objectness_target, regression_target in train_loader:
            x = x.to(device)
            objectness_target = objectness_target.to(device)
            regression_target = regression_target.to(device)

            optimizer.zero_grad()
            _seg_out, _aux, decoder_features = model(x, return_features=True)
            pred = head(decoder_features)
            loss = loss_fn(pred, objectness_target, regression_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"[DETECTION] Epoch {epoch}/{epochs - 1}: train loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches)")

        model.eval()
        head.eval()
        val_running_loss = 0.0
        val_n_batches = 0
        total_positive_pixels = 0
        total_predicted_peaks = 0
        with torch.no_grad():
            for x, objectness_target, regression_target in val_loader:
                x = x.to(device)
                objectness_target = objectness_target.to(device)
                regression_target = regression_target.to(device)
                seg_out, decoder_features = model(x, return_features=True)
                pred = head(decoder_features)
                val_loss = loss_fn(pred, objectness_target, regression_target)
                val_running_loss += val_loss.item()
                val_n_batches += 1
                total_positive_pixels += int((objectness_target > 0.5).sum().item())
                total_predicted_peaks += int((torch.sigmoid(pred[:, 0]) > 0.5).sum().item())

        val_loss_avg = val_running_loss / max(1, val_n_batches)
        print(
            f"[DETECTION] Epoch {epoch}: val loss={val_loss_avg:.4f} "
            f"(real positive pixels in val: {total_positive_pixels}, "
            f"predicted-positive pixels: {total_predicted_peaks} -- raw pixel counts, "
            f"not yet decoded into discrete instances; see perception/detection_decode.py for that step)"
        )

        is_best = val_loss_avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "backbone_state": model.state_dict(), "head_state": head.state_dict()},
                out_dir / "best.pt",
            )

        with open(training_log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "val_loss": val_loss_avg,
                        "real_positive_pixels_val": total_positive_pixels,
                        "predicted_positive_pixels_val": total_predicted_peaks,
                        "is_best": is_best,
                        "best_epoch_so_far": best_epoch,
                        "best_val_loss_so_far": best_val_loss,
                    }
                )
                + "\n"
            )

    print(f"Detection head training complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the lightweight detection head on real nuScenes-mini 3D boxes")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sensor-config", default="configs/sensor_hdl32e.yaml")
    parser.add_argument("--seg-checkpoint", default="checkpoints_joint/best.pt")
    parser.add_argument("--seg-stats", default="checkpoints_joint/channel_stats.json")
    parser.add_argument("--out-dir", default="checkpoints_detection")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--cache-dir", default=None,
        help="Enable on-disk frame caching under this directory (see perception/frame_cache.py). "
             "Especially valuable here -- this dataset's per-frame cost includes real 3D "
             "box-target projection, not just segmentation projection.",
    )
    args = parser.parse_args()

    train_detection(
        dataroot=args.dataroot,
        version=args.version,
        sensor_config_path=args.sensor_config,
        seg_checkpoint_path=args.seg_checkpoint,
        seg_stats_path=args.seg_stats,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr_scale=args.backbone_lr_scale,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
