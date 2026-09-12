"""
perception/train.py

Ticket #30 -- train FusionSegNet on RELLIS-3D (preferred over
nuScenes-mini per this ticket's own spec, since #3 has landed).

Supports one or more sequence directories (--sequence-dir accepts
multiple paths). Each sequence is split train/val independently (last
15% by index, per sequence -- see Val split note below) and the splits
are concatenated, so held-out frames always come from the tail of
their own sequence's route rather than leaking across sequences.

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
import random
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from perception.cutmix import RareClusterSet, load_clusters, paste_rare_cluster
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import (
    N_CHANNELS,
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
from perception.segnet import (
    STEM_CONV_STATE_DICT_KEY,
    FusionSegNet,
    N_CLASSES_DEFAULT,
    expand_stem_conv_for_checkpoint,
    remap_legacy_conv_keys,
)
from perception.taxonomy import rellis_label_ids_to_drishti
from sensor.sensor_model import SensorConfig, load_sensor_config

VAL_FRACTION = 0.15  # last 15% of frames by index, contiguous -- Ticket #30's own spec


def split_frames(n_frames: int, val_fraction: float = VAL_FRACTION):
    n_val = max(1, int(round(n_frames * val_fraction)))
    n_train = n_frames - n_val
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_frames))
    return train_idx, val_idx


def build_multi_sequence_splits(sequence_dirs: list[Path], bin_dirname: str = "os1_cloud_node_kitti_bin"):
    """Per-sequence train/val split (see module docstring), concatenated
    into (seq_dir, frame_idx) item lists across all sequences."""
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    per_seq_counts: list[tuple[Path, int]] = []
    for seq_dir in sequence_dirs:
        n_frames = len(list((seq_dir / bin_dirname).glob("*.bin")))
        train_idx, val_idx = split_frames(n_frames)
        train_items.extend((seq_dir, i) for i in train_idx)
        val_items.extend((seq_dir, i) for i in val_idx)
        per_seq_counts.append((seq_dir, n_frames))
    return train_items, val_items, per_seq_counts


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


def _load_frame_with_cutmix(
    sequence_dir: Path,
    frame_idx: int,
    sm: SensorConfig,
    clusters: "RareClusterSet",
    target_class: int,
    rng: np.random.Generator,
):
    """Same as `_load_frame`, but pastes a real harvested rare-class
    cluster (`perception.cutmix.paste_rare_cluster`) into the raw sweep
    BEFORE projection, so the pasted points go through the exact same
    occlusion-aware `project_to_range_image` z-buffering as every real
    point -- see `perception/cutmix.py`'s own module docstring for why
    this must happen pre-projection, not as an image-space patch paste.
    """
    sweep = load_rellis_sweep(sequence_dir, frame_idx)
    raw_labels = load_rellis_labels(sequence_dir, frame_idx)
    drishti_labels = rellis_label_ids_to_drishti(raw_labels)

    sweep, drishti_labels = paste_rare_cluster(sweep, drishti_labels, clusters, target_class, rng)

    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

    target = np.zeros((img.H, img.W), dtype=np.int64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    target[touched] = drishti_labels[src]

    return img, ground, target


# Augmentation constants -- chosen to stay physically plausible (a real
# sensor's own range noise is a few percent, not tens), not tuned
# against any specific metric.
AUG_JITTER_PROB = 0.5
AUG_JITTER_RANGE = (0.95, 1.05)  # multiplicative, applied to x/y/z/range TOGETHER (radial)
AUG_ROLL_PROB = 0.5  # circular azimuth shift -- the network's own circular padding makes this "free"
AUG_FLIP_PROB = 0.5  # azimuth mirror -- LiDAR has no inherent left/right asymmetry

# Beam-dropout: simulates a sparser sensor (32/21/16 effective beams from
# this project's real 64-beam Ouster) by decimating ROWS of the already-
# projected (H, W) tensor -- H IS the beam/elevation axis by construction
# (perception.range_image.project_to_range_image), so dropping every
# k-th row's worth of rows is equivalent to the network never having
# seen those beams at all, not merely zeroed pixels it could otherwise
# infer from context. EVENLY-SPACED decimation (not a random subset) is
# used deliberately -- it is what a real coarser-beam-spacing sensor
# actually looks like (fewer beams spread evenly across the same
# vertical FOV), not an arbitrary corruption.
AUG_BEAM_DROPOUT_PROB = 0.3
BEAM_DROPOUT_STRIDES = (2, 3, 4)  # ~32, ~21, ~16 effective beams from 64

AUG_CUTMIX_PROB = 0.3  # see perception/cutmix.py; targets class 4's confirmed 0.051% prevalence
CUTMIX_TARGET_CLASS = 4  # DrishtiClass.STATIC_OBSTACLE


def _apply_radial_jitter(img, rng: random.Random):
    """Scale x, y, z, AND range by the SAME random factor -- a radial
    jitter along each point's own bearing, keeping range consistent
    with sqrt(x^2+y^2+z^2) (Ticket #23's own invariant). Applied to the
    RangeImage BEFORE normalisation (`assemble_input_tensor`), not to
    the already-normalised tensor -- multiplying a zero-centred
    normalised value by a scalar does not correspond to "+-5% of real
    distance" the way multiplying the RAW metre value does."""
    factor = rng.uniform(*AUG_JITTER_RANGE)
    return dataclass_replace(img, x=img.x * factor, y=img.y * factor, z=img.z * factor, range=img.range * factor)


def _apply_spatial_augmentation(tensor: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, rng: random.Random):
    """Circular roll and/or mirror flip along the AZIMUTH (W) axis,
    applied to `tensor`, `target`, AND `valid_mask` TOGETHER -- a
    pixel and its label/validity must move as one unit. (A prior draft
    of this augmentation rolled/flipped only tensor+target and left
    valid_mask untouched, which would silently apply each mask bit to
    the WRONG post-shift pixel -- caught before this was ever run.)
    """
    if rng.random() < AUG_ROLL_PROB:
        shift = rng.randint(0, tensor.shape[-1] - 1)
        tensor = torch.roll(tensor, shift, dims=2)
        target = torch.roll(target, shift, dims=1)
        valid_mask = torch.roll(valid_mask, shift, dims=1)
    if rng.random() < AUG_FLIP_PROB:
        tensor = torch.flip(tensor, dims=[2])
        target = torch.flip(target, dims=[1])
        valid_mask = torch.flip(valid_mask, dims=[1])
    return tensor, target, valid_mask


def _apply_beam_dropout(tensor: torch.Tensor, valid_mask: torch.Tensor, rng: random.Random):
    """Zeroes an evenly-spaced subset of BEAM ROWS in `tensor` and marks
    those same rows invalid in `valid_mask` -- see AUG_BEAM_DROPOUT_PROB's
    own comment for why this simulates a real coarser-beam sensor rather
    than an arbitrary corruption. `target` is untouched: the loss already
    only scores pixels where `valid_mask` is True (the SAME mechanism
    that already excludes non-projected pixels), so dropped rows are
    excluded from the loss for free, not by editing labels."""
    H = tensor.shape[1]
    stride = rng.choice(BEAM_DROPOUT_STRIDES)
    offset = rng.randint(0, stride - 1)
    keep = torch.zeros(H, dtype=torch.bool)
    keep[offset::stride] = True
    drop_rows = ~keep

    tensor = tensor.clone()
    tensor[:, drop_rows, :] = 0.0
    valid_mask = valid_mask.clone()
    valid_mask[drop_rows, :] = False
    return tensor, valid_mask


class RellisSegDataset(Dataset):
    """One item = one frame: (input_tensor (9,H,W), target (H,W) int64,
    valid_mask (H,W) bool). `items` is a list of (sequence_dir, frame_idx)
    pairs, so a single dataset can span multiple RELLIS-3D sequences.

    `is_train=True` enables augmentation (radial jitter + circular
    roll + azimuth mirror); the validation dataset must be constructed
    with `is_train=False` (the default) so held-out metrics measure
    the model on UNMODIFIED frames, matching the ticket's own "no
    shuffle-split, no leakage" discipline extended to "no augmentation
    leakage into the number you report."""

    def __init__(
        self,
        items: list,
        sm: SensorConfig,
        stats: ChannelStats,
        is_train: bool = False,
        cutmix_clusters: Optional[RareClusterSet] = None,
    ):
        self.items = list(items)
        self.sm = sm
        self.stats = stats
        self.is_train = is_train
        self.cutmix_clusters = cutmix_clusters
        self._rng = random.Random()
        self._np_rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        sequence_dir, frame_idx = self.items[i]

        use_cutmix = (
            self.is_train
            and self.cutmix_clusters is not None
            and self.cutmix_clusters.clusters
            and self._rng.random() < AUG_CUTMIX_PROB
        )
        if use_cutmix:
            img, ground, target = _load_frame_with_cutmix(
                sequence_dir, frame_idx, self.sm, self.cutmix_clusters, CUTMIX_TARGET_CLASS, self._np_rng
            )
        else:
            img, ground, target = _load_frame(sequence_dir, frame_idx, self.sm)

        if self.is_train and self._rng.random() < AUG_JITTER_PROB:
            img = _apply_radial_jitter(img, self._rng)

        tensor_np = assemble_input_tensor(img, ground, self.stats)
        tensor = torch.from_numpy(tensor_np).float()
        target_t = torch.from_numpy(target).long()
        valid_t = torch.from_numpy(img.valid_mask).bool()

        if self.is_train:
            tensor, target_t, valid_t = _apply_spatial_augmentation(tensor, target_t, valid_t, self._rng)
            if self._rng.random() < AUG_BEAM_DROPOUT_PROB:
                tensor, valid_t = _apply_beam_dropout(tensor, valid_t, self._rng)

        return tensor, target_t, valid_t


def compute_class_weights(class_counts: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-SQRT-frequency weights (dampens extremes vs. plain
    inverse -- a class 461x rarer than the majority would otherwise
    get a 461x weight, overcorrecting hard enough to destabilise the
    classes that were already learning well), MEAN-normalised (not
    sum-normalised) so the overall CE loss magnitude stays comparable
    to the unweighted case -- normalising to sum=1 would shrink the
    secondary CE term by roughly `n_classes`, silently changing how
    much it contributes relative to the primary Lovász term's own
    fixed weighting (`DrishtiSegLoss.ce_weight`).

    Classes with ZERO pixels get weight 0, not an arbitrary large
    number from 1/sqrt(0) -- classes 8/9 (NEGATIVE_OBSTACLE, OVERHANG)
    NEVER appear in RELLIS-3D's remapped targets by taxonomy design
    (perception/taxonomy.py), so their weight is moot for the CE
    gradient either way, but a naive 1/sqrt(count) would divide by
    zero here if not guarded.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    mean_nonzero = weights[nonzero].mean() if np.any(nonzero) else 1.0
    weights = weights / mean_nonzero  # mean of the NONZERO weights is 1, not the sum
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_pixel_counts(items: list, sm: SensorConfig, n_classes: int, sample_every: int = 1):
    """Ticket #30 'Watch out': check rare classes aren't collapsing to
    zero IoU before burning hours -- pixel counts per class over (a
    sample of) the training split, printed before training starts."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for sequence_dir, frame_idx in items[::sample_every]:
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
    sequence_dir,
    sensor_config_path: str,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 3e-4,
    num_workers: int = 2,
    device: Optional[str] = None,
    max_stats_frames: int = 30,
    init_from_checkpoint: Optional[str] = None,
    use_class_weights: bool = True,
    focal_gamma: Optional[float] = None,
    cutmix_clusters_path: Optional[str] = None,
) -> None:
    """`init_from_checkpoint`: load ONLY model weights from a prior
    run's checkpoint (e.g. `checkpoints_multi/checkpoint_epoch19.pt`)
    as a fine-tuning starting point -- fresh optimizer/scheduler/scaler
    state, fresh epoch count. Distinct from the automatic `checkpoint.pt`
    resume in `out_dir` (which restores FULL training state to survive
    an interrupted job, per Ticket #30's own "Colab disconnects" Watch
    out) -- this is for deliberately starting a NEW recipe (augmentation,
    class weighting, a differently-sized LR schedule) from already-
    converged features, not for resuming the SAME run. Only applied
    when `out_dir` has no resumable checkpoint of its own yet (an
    interrupted run of THIS recipe always takes priority)."""
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
    sequence_dirs = [Path(sequence_dir)] if isinstance(sequence_dir, (str, Path)) else [Path(p) for p in sequence_dir]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sm = load_sensor_config(sensor_config_path)

    train_items, val_items, per_seq_counts = build_multi_sequence_splits(sequence_dirs)
    seq_names = ", ".join(f"{d.name} ({n} frames)" for d, n in per_seq_counts)
    print(f"Training on {len(train_items)} frames from RELLIS-3D [{seq_names}] "
          f"(held out {len(val_items)} frames total, last {VAL_FRACTION:.0%} by index, per sequence)")

    stats_path = out_dir / "channel_stats.json"
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"Loaded channel stats from {stats_path} (not recomputed)")
    else:
        raw_stacks = []
        for sequence_dir_i, frame_idx in train_items[:: max(1, len(train_items) // max_stats_frames)][:max_stats_frames]:
            img, ground, _ = _load_frame(sequence_dir_i, frame_idx, sm)
            raw_stacks.append(_raw_channels(img, ground_prior_channel_from_points(img, ground)))
        stats = compute_channel_stats(raw_stacks)
        save_stats(stats, stats_path)
        print(f"Computed channel stats from {len(raw_stacks)} frames, saved to {stats_path}")

    class_counts = compute_class_pixel_counts(
        train_items, sm, N_CLASSES_DEFAULT, sample_every=max(1, len(train_items) // 50)
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

    cutmix_clusters = None
    if cutmix_clusters_path:
        cutmix_clusters = load_clusters(cutmix_clusters_path)
        print(f"Loaded {len(cutmix_clusters.clusters)} CutMix clusters from {cutmix_clusters_path}")

    train_ds = RellisSegDataset(train_items, sm, stats, is_train=True, cutmix_clusters=cutmix_clusters)
    val_ds = RellisSegDataset(val_items, sm, stats, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT).to(device)

    ckpt_path = out_dir / "checkpoint.pt"
    if init_from_checkpoint and not ckpt_path.exists():
        init_ckpt = torch.load(init_from_checkpoint, map_location=device)
        init_state = init_ckpt["model_state"]
        # Older checkpoints (e.g. checkpoint_epoch19.pt) pre-date ASPP/
        # decoder convs being wrapped in CircularConv2d, which renamed
        # their own parameter keys -- see
        # perception.segnet.remap_legacy_conv_keys's own docstring.
        # eval/cache_inference.py already had this fix; this path did
        # not, which is why loading checkpoint_epoch19.pt here failed
        # until now.
        init_state = remap_legacy_conv_keys(init_state, set(model.state_dict().keys()))
        old_in_channels = init_state[STEM_CONV_STATE_DICT_KEY].shape[1]
        if old_in_channels != N_CHANNELS:
            # The checkpoint was trained with a different input-channel
            # count than perception.input_tensor's CURRENT channel list
            # (e.g. loading a 9-channel checkpoint after adding the
            # normal/curvature channels) -- grow the stem conv rather
            # than discarding the old channels' trained weights. See
            # perception.segnet.expand_stem_conv_for_checkpoint's own
            # docstring for exactly what this does and does not do.
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
    # torch.amp.GradScaler (device-agnostic) doesn't exist before torch
    # ~2.3; torch.cuda.amp.GradScaler is deprecated in newer torch but is
    # what the actual training machine (torch 2.0.1) has. Try new, fall
    # back to old -- confirmed this matters: the newer-only form crashes
    # on the real GPU server's installed torch version.
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
    print(f"Majority-class baseline: class {majority_class} "
          f"({class_counts[majority_class] / max(1, class_counts.sum()):.1%} of sampled pixels)")

    epoch_durations = []
    training_log_path = out_dir / "training_log.jsonl"
    best_miou = -1.0
    best_epoch = -1
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    OVERFIT_WINDOW = 3  # consecutive epochs of val-loss-up + train-loss-down before flagging

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
        val_running_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            for x, target, valid in val_loader:
                x, target_d, valid_d = x.to(device), target.to(device), valid.to(device)
                logits = model(x)
                val_loss = loss_fn(logits, target_d, valid_d)
                val_running_loss += val_loss.item()
                val_n_batches += 1
                pred = logits.argmax(dim=1).cpu().numpy()[0]
                confusion_matrix_update(cm, pred, target.numpy()[0], valid.numpy()[0], N_CLASSES_DEFAULT)

        val_loss_avg = val_running_loss / max(1, val_n_batches)
        ious = per_class_iou(cm)
        miou = float(np.nanmean(ious))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: val loss={val_loss_avg:.4f} val mIoU={miou:.4f} lr={current_lr:.2e}")
        for c, iou in enumerate(ious):
            print(f"  class {c} IoU: {iou if not np.isnan(iou) else 'n/a (no pixels)'}")

        # Overfitting signal: train loss falling while val loss RISES,
        # for OVERFIT_WINDOW consecutive epochs -- a real divergence,
        # not the single-epoch noise a naive "did it go up once" check
        # would false-positive on.
        train_loss_history.append(avg_loss)
        val_loss_history.append(val_loss_avg)
        overfitting = False
        if len(val_loss_history) > OVERFIT_WINDOW:
            recent_val = val_loss_history[-(OVERFIT_WINDOW + 1):]
            recent_train = train_loss_history[-(OVERFIT_WINDOW + 1):]
            val_rising = all(recent_val[i] < recent_val[i + 1] for i in range(len(recent_val) - 1))
            train_falling = recent_train[0] > recent_train[-1]
            overfitting = val_rising and train_falling
        if overfitting:
            print(f"  WARNING: val loss has risen for {OVERFIT_WINDOW} consecutive epochs while train loss fell -- likely overfitting")

        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", epoch, model, optimizer, scheduler, scaler)

        with open(training_log_path, "a") as f:
            f.write(json.dumps({
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
                "overfitting_flag": overfitting,
            }) + "\n")

        with open(out_dir / f"val_metrics_epoch{epoch}.json", "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss_avg,
                    "miou": miou,
                    "per_class_iou": [None if np.isnan(v) else float(v) for v in ious],
                    "train_set_size": len(train_items),
                    "val_set_size": len(val_items),
                    "majority_class_baseline_class": majority_class,
                    "lr": current_lr,
                    "epoch_time_s": elapsed,
                    "overfitting_flag": overfitting,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket #30 -- train FusionSegNet on RELLIS-3D")
    parser.add_argument("--sequence-dir", required=True, nargs="+", help="One or more RELLIS-3D sequence directories")
    parser.add_argument("--sensor-config", default="configs/sensor_ouster_os1_64.yaml")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--init-from-checkpoint",
        default=None,
        help="Load ONLY model weights from a prior run's checkpoint as a fine-tuning start "
             "(fresh optimizer/scheduler/epoch count) -- only takes effect if --out-dir has no "
             "resumable checkpoint.pt of its own yet.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-sqrt-frequency class weighting on the secondary CE term (on by default).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Enable focal loss (perception.losses.focal_loss) as an additional loss term with this "
             "gamma (typical: 2.0). Off by default (None) -- byte-identical to the pre-focal-loss "
             "behaviour when omitted.",
    )
    parser.add_argument(
        "--cutmix-clusters",
        default=None,
        help="Path to a .npz produced by eval/extract_rare_clusters.py -- enables CutMix pasting of "
             "real class-4 (STATIC_OBSTACLE) point clusters into training frames. Off by default.",
    )
    args = parser.parse_args()

    train(
        sequence_dir=args.sequence_dir,
        sensor_config_path=args.sensor_config,
        out_dir=args.out_dir,
        init_from_checkpoint=args.init_from_checkpoint,
        use_class_weights=not args.no_class_weights,
        focal_gamma=args.focal_gamma,
        cutmix_clusters_path=args.cutmix_clusters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
    )
