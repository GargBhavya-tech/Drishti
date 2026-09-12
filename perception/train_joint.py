"""
perception/train_joint.py

Joint multi-dataset training: ONE FusionSegNet trained on RELLIS-3D +
nuScenes-mini + SemanticPOSS TOGETHER, in the same batches (via
perception.joint_seg_dataset.build_joint_train_dataset). Every other
training script this session (perception/train.py,
perception/train_nuscenes.py, perception/train_semanticposs.py) trains
on exactly one dataset, then reports how well that checkpoint transfers
or fine-tunes elsewhere -- this is the genuinely different experiment:
does training on all three domains AT ONCE, from the start of each
epoch, produce a model that's better across all three than any of the
three single-dataset fine-tunes was on its own domain? That question
cannot be answered by comparing the three separate runs already in
DRISHTI_MASTER_BIBLE.md Part G.6-G.8; it needs this script.

Reuses each dataset's own train/val split machinery UNCHANGED (RELLIS-3D:
perception.train.build_multi_sequence_splits + RellisSegDataset;
nuScenes: perception.nuscenes_seg_dataset.build_nuscenes_scene_splits +
NuscenesSegDataset; SemanticPOSS: perception.semanticposs_seg_dataset.
build_semanticposs_splits + SemanticPossSegDataset) -- only the TRAIN
items get concatenated (and resized to one common shape, see
perception.joint_seg_dataset's own docstring). VALIDATION deliberately
stays THREE SEPARATE per-domain passes, each at that domain's own
native resolution -- a single blended val mIoU across three different
class distributions and sensor resolutions would hide exactly the kind
of per-domain regression this experiment needs to be able to see (e.g.
if joint training helps nuScenes DRIVABLE but hurts RELLIS VEGETATION,
a single averaged number would show a small net change and hide both
real effects). "best checkpoint" selection uses the unweighted MEAN of
the three domains' own mIoU as one scalar criterion, but every epoch's
full per-domain, per-class breakdown is still logged -- the mean is a
convenience for picking best.pt, never the only number reported.

Channel-normalisation stats and class weights are computed ONCE from a
sample spanning all three domains' train items (not per-domain) -- the
network sees one shared input distribution regardless of which domain
a given batch element came from, which is the correct setup for a
single shared BatchNorm to learn from consistently.

Domain size imbalance, stated rather than hidden: RELLIS-3D contributes
~11,522 train frames, SemanticPOSS ~2,540, nuScenes-mini only ~324 --
concatenation with shuffling means RELLIS-3D dominates raw sample
counts roughly 4.5:36:1 respectively. No domain-balancing (e.g.
oversampling the smaller domains) is applied in this first version;
if nuScenes-specific metrics don't move, that imbalance -- not a
failure of joint training itself -- is the first thing to suspect.
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
from perception.joint_seg_dataset import build_joint_train_dataset
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
from perception.semanticposs_seg_dataset import (
    SemanticPossSegDataset,
    _load_semanticposs_frame,
    build_semanticposs_splits,
)
from perception.taxonomy import build_nuscenes_lidarseg_lut
from perception.train import (
    RellisSegDataset,
    _load_frame,
    build_multi_sequence_splits,
    compute_class_weights,
    confusion_matrix_update,
    load_checkpoint_if_exists,
    per_class_iou,
    save_checkpoint,
)
from sensor.sensor_model import load_sensor_config


def _sample_stats_stack(loader_fn, items, sm, max_frames):
    """Shared helper: run `loader_fn` (a per-domain _load_*_frame
    function) over a stride-sampled slice of `items`, returning
    _raw_channels() stacks for perception.input_tensor.compute_channel_stats."""
    step = max(1, len(items) // max_frames)
    stacks = []
    for item in items[::step][:max_frames]:
        img, ground, _ = loader_fn(*item, sm) if isinstance(item, tuple) else loader_fn(item, sm)
        stacks.append(_raw_channels(img, ground_prior_channel_from_points(img, ground)))
    return stacks


def train_joint(
    rellis_sequence_dirs,
    rellis_sensor_config_path: str,
    nuscenes_dataroot: str,
    nuscenes_version: str,
    nuscenes_sensor_config_path: str,
    semanticposs_sequence_dirs,
    semanticposs_sensor_config_path: str,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2e-4,
    num_workers: int = 2,
    device: Optional[str] = None,
    max_stats_frames_per_domain: int = 15,
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
    from nuscenes.nuscenes import NuScenes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- per-domain sensor configs and splits --------------------------
    rellis_sm = load_sensor_config(rellis_sensor_config_path)
    rellis_dirs = [Path(p) for p in rellis_sequence_dirs]
    rellis_train_items, rellis_val_items, rellis_counts = build_multi_sequence_splits(rellis_dirs)
    print(f"RELLIS-3D: {len(rellis_train_items)} train / {len(rellis_val_items)} val frames")

    print(f"Loading nuScenes {nuscenes_version} from {nuscenes_dataroot} ...")
    nusc = NuScenes(version=nuscenes_version, dataroot=nuscenes_dataroot, verbose=False)
    nuscenes_sm = load_sensor_config(nuscenes_sensor_config_path)
    nuscenes_lut = build_nuscenes_lidarseg_lut(nusc)
    nuscenes_train_items, nuscenes_val_items, nuscenes_scene_counts = build_nuscenes_scene_splits(nusc)
    print(f"nuScenes-mini: {len(nuscenes_train_items)} train / {len(nuscenes_val_items)} val frames")

    semanticposs_sm = load_sensor_config(semanticposs_sensor_config_path)
    semanticposs_dirs = [Path(p) for p in semanticposs_sequence_dirs]
    semanticposs_train_items, semanticposs_val_items, semanticposs_counts = build_semanticposs_splits(semanticposs_dirs)
    print(f"SemanticPOSS: {len(semanticposs_train_items)} train / {len(semanticposs_val_items)} val frames")

    total_train = len(rellis_train_items) + len(nuscenes_train_items) + len(semanticposs_train_items)
    print(
        f"JOINT train set: {total_train} frames "
        f"(RELLIS {len(rellis_train_items)/total_train:.1%}, "
        f"nuScenes {len(nuscenes_train_items)/total_train:.1%}, "
        f"SemanticPOSS {len(semanticposs_train_items)/total_train:.1%}) -- see module docstring on imbalance"
    )

    # --- shared channel stats, computed across ALL THREE domains --------
    stats_path = out_dir / "channel_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"Loaded channel stats from {stats_path} (not recomputed)")
    else:
        raw_stacks = []
        raw_stacks += _sample_stats_stack(_load_frame, rellis_train_items, rellis_sm, max_stats_frames_per_domain)
        raw_stacks += _sample_stats_stack(
            lambda tok, sm: _load_nuscenes_frame(nusc, tok, sm, nuscenes_lut),
            nuscenes_train_items, nuscenes_sm, max_stats_frames_per_domain,
        )
        raw_stacks += _sample_stats_stack(_load_semanticposs_frame, semanticposs_train_items, semanticposs_sm, max_stats_frames_per_domain)
        stats = compute_channel_stats(raw_stacks)
        save_stats(stats, stats_path)
        print(f"Computed SHARED channel stats from {len(raw_stacks)} frames across all 3 domains, saved to {stats_path}")

    # --- shared class weights, summed across ALL THREE domains ----------
    class_counts = np.zeros(N_CLASSES_DEFAULT, dtype=np.int64)
    for sequence_dir, frame_idx in rellis_train_items[:: max(1, len(rellis_train_items) // 40)]:
        img, _, target = _load_frame(sequence_dir, frame_idx, rellis_sm)
        vt = target[img.valid_mask]
        for c in range(N_CLASSES_DEFAULT):
            class_counts[c] += int((vt == c).sum())
    for tok in nuscenes_train_items[:: max(1, len(nuscenes_train_items) // 40)]:
        img, _, target = _load_nuscenes_frame(nusc, tok, nuscenes_sm, nuscenes_lut)
        vt = target[img.valid_mask]
        for c in range(N_CLASSES_DEFAULT):
            class_counts[c] += int((vt == c).sum())
    for sequence_dir, frame_id in semanticposs_train_items[:: max(1, len(semanticposs_train_items) // 40)]:
        img, _, target = _load_semanticposs_frame(sequence_dir, frame_id, semanticposs_sm)
        vt = target[img.valid_mask]
        for c in range(N_CLASSES_DEFAULT):
            class_counts[c] += int((vt == c).sum())
    print("Per-class pixel counts (sampled across all 3 domains' train frames):")
    for c, count in enumerate(class_counts):
        print(f"  class {c}: {count}")

    # --- datasets --------------------------------------------------------
    # RELLIS-3D is deliberately NEVER cached, even when cache_dir is
    # given: 11,522 frames at the real 64x2048x13 raw-stack size is an
    # ESTIMATED ~78GB uncompressed -- this project's own server sat at
    # 31GB free / 92% used when this was measured. Caching nuScenes and
    # SemanticPOSS (far smaller: ~730MB and low tens of GB respectively)
    # is real and safe; caching RELLIS-3D here would not be an
    # optimisation, it would be a disk-filling mistake on a SHARED
    # server. See perception/frame_cache.py's own docstring.
    if cache_dir:
        nuscenes_train_cache = FrameCache(Path(cache_dir) / "joint_nuscenes_train")
        nuscenes_val_cache = FrameCache(Path(cache_dir) / "joint_nuscenes_val")
        semanticposs_train_cache = FrameCache(Path(cache_dir) / "joint_semanticposs_train")
        semanticposs_val_cache = FrameCache(Path(cache_dir) / "joint_semanticposs_val")
        print(
            f"Frame caching ENABLED at {cache_dir} for nuScenes and SemanticPOSS ONLY -- "
            f"RELLIS-3D is never cached here (estimated ~78GB uncompressed, not disk-feasible "
            f"on this project's server; see this script's own comment)."
        )
    else:
        nuscenes_train_cache = nuscenes_val_cache = semanticposs_train_cache = semanticposs_val_cache = None

    rellis_train_ds = RellisSegDataset(rellis_train_items, rellis_sm, stats, is_train=True)
    nuscenes_train_ds = NuscenesSegDataset(
        nusc, nuscenes_train_items, nuscenes_sm, stats, nuscenes_lut, is_train=True, cache=nuscenes_train_cache
    )
    semanticposs_train_ds = SemanticPossSegDataset(
        semanticposs_train_items, semanticposs_sm, stats, is_train=True, cache=semanticposs_train_cache
    )
    joint_train_ds = build_joint_train_dataset(rellis_train_ds, nuscenes_train_ds, semanticposs_train_ds)
    train_loader = DataLoader(joint_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    rellis_val_ds = RellisSegDataset(rellis_val_items, rellis_sm, stats, is_train=False)
    nuscenes_val_ds = NuscenesSegDataset(
        nusc, nuscenes_val_items, nuscenes_sm, stats, nuscenes_lut, is_train=False, cache=nuscenes_val_cache
    )
    semanticposs_val_ds = SemanticPossSegDataset(
        semanticposs_val_items, semanticposs_sm, stats, is_train=False, cache=semanticposs_val_cache
    )
    val_loaders = {
        "rellis": DataLoader(rellis_val_ds, batch_size=1, shuffle=False, num_workers=num_workers),
        "nuscenes": DataLoader(nuscenes_val_ds, batch_size=1, shuffle=False, num_workers=num_workers),
        "semanticposs": DataLoader(semanticposs_val_ds, batch_size=1, shuffle=False, num_workers=num_workers),
    }

    # --- model / optimizer / warm start ----------------------------------
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    try:
        scaler = torch.amp.GradScaler(device if device != "cpu" else "cpu", enabled=(device == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    class_weight = compute_class_weights(class_counts, N_CLASSES_DEFAULT).to(device) if use_class_weights else None
    if use_class_weights:
        print("Class weights (secondary CE term only, mean-normalised inverse-sqrt-frequency, ALL 3 domains combined):")
        for c, w in enumerate(class_weight.tolist()):
            print(f"  class {c}: {w:.3f}")
    loss_fn = DrishtiSegLoss(class_weight=class_weight, focal_gamma=focal_gamma).to(device)

    start_epoch = load_checkpoint_if_exists(ckpt_path, model, optimizer, scheduler, scaler, device)
    if start_epoch > 0:
        print(f"Resuming from checkpoint at epoch {start_epoch}")

    epoch_durations = []
    training_log_path = out_dir / "training_log.jsonl"
    best_mean_miou = -1.0
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
            f"[JOINT] Epoch {epoch}/{epochs - 1}: train loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches) "
            f"-- avg {avg_epoch_time:.1f}s/epoch, ETA {eta_s / 60:.1f} min ({epochs_left} epochs left)"
        )

        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler)
        save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", epoch, model, optimizer, scheduler, scaler)

        # --- THREE SEPARATE per-domain val passes -- see module docstring ---
        model.eval()
        domain_results = {}
        with torch.no_grad():
            for domain_name, loader in val_loaders.items():
                cm = np.zeros((N_CLASSES_DEFAULT, N_CLASSES_DEFAULT), dtype=np.int64)
                for x, target, valid in loader:
                    x_d, target_d, valid_d = x.to(device), target.to(device), valid.to(device)
                    logits = model(x_d)
                    pred = logits.argmax(dim=1).cpu().numpy()[0]
                    confusion_matrix_update(cm, pred, target.numpy()[0], valid.numpy()[0], N_CLASSES_DEFAULT)
                ious = per_class_iou(cm)
                miou = float(np.nanmean(ious))
                domain_results[domain_name] = {
                    "miou": miou,
                    "per_class_iou": [None if np.isnan(v) else float(v) for v in ious],
                }
                print(f"[JOINT] Epoch {epoch} -- {domain_name} val mIoU={miou:.4f}")
                for c, iou in enumerate(ious):
                    print(f"    class {c} IoU: {iou if not np.isnan(iou) else 'n/a (no pixels)'}")

        mean_miou = float(np.mean([domain_results[d]["miou"] for d in domain_results]))
        current_lr = scheduler.get_last_lr()[0]
        print(f"[JOINT] Epoch {epoch}: MEAN val mIoU across 3 domains = {mean_miou:.4f}  lr={current_lr:.2e}")

        is_best = mean_miou > best_mean_miou
        if is_best:
            best_mean_miou = mean_miou
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", epoch, model, optimizer, scheduler, scaler)

        with open(training_log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "mean_val_miou": mean_miou,
                        "domain_results": domain_results,
                        "lr": current_lr,
                        "epoch_time_s": elapsed,
                        "is_best": is_best,
                        "best_epoch_so_far": best_epoch,
                        "best_mean_miou_so_far": best_mean_miou,
                    }
                )
                + "\n"
            )

    print(f"Joint training complete. Best MEAN val mIoU (3 domains) {best_mean_miou:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint multi-dataset training: RELLIS-3D + nuScenes-mini + SemanticPOSS")
    parser.add_argument("--rellis-sequence-dir", nargs="+", default=[f"data/rellis/{i:05d}" for i in range(5)])
    parser.add_argument("--rellis-sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--nuscenes-dataroot", default="data/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--nuscenes-sensor-config", default="configs/sensor_hdl32e.yaml")
    parser.add_argument("--semanticposs-sequence-dir", nargs="+", default=[f"data/semanticposs/extracted/dataset/sequences/{i:02d}" for i in range(6)])
    parser.add_argument("--semanticposs-sensor-config", default="configs/sensor_pandar40p.yaml")
    parser.add_argument("--out-dir", default="checkpoints_joint")
    parser.add_argument("--init-from-checkpoint", default="checkpoints_multi_v3/best.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Enable on-disk frame caching under this directory for nuScenes and SemanticPOSS "
             "ONLY (see perception/frame_cache.py). RELLIS-3D is never cached (estimated ~78GB "
             "uncompressed -- not disk-feasible on this project's server).",
    )
    args = parser.parse_args()

    train_joint(
        rellis_sequence_dirs=args.rellis_sequence_dir,
        rellis_sensor_config_path=args.rellis_sensor_config,
        nuscenes_dataroot=args.nuscenes_dataroot,
        nuscenes_version=args.nuscenes_version,
        nuscenes_sensor_config_path=args.nuscenes_sensor_config,
        semanticposs_sequence_dirs=args.semanticposs_sequence_dir,
        semanticposs_sensor_config_path=args.semanticposs_sensor_config,
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
