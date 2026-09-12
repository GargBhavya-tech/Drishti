"""
perception/semanticposs_seg_dataset.py

SemanticPOSS training dataset for FusionSegNet. Same
(input_tensor, target, valid_mask) contract and same range-image /
ground-prior / assemble_input_tensor pipeline as
perception.train.RellisSegDataset and
perception.nuscenes_seg_dataset.NuscenesSegDataset -- this is the third
dataset built on that shared contract this session.

Why a THIRD real dataset, not just RELLIS + nuScenes: SemanticPOSS is a
real walking-campus-environment LiDAR dataset (Hesai Pandar40P, 40-beam,
Peking University), a domain distinct from both RELLIS-3D (off-road
trail) and nuScenes (highway/street driving) -- pedestrians, bikes,
building facades, tree trunks, at much closer typical range than a
vehicle dataset. Testing against it is a genuinely different
generalization question than the nuScenes fine-tune (Bible Part G.7)
answered.

Val split: PER SEQUENCE, last ~15% of that sequence's own frames by
frame-id order -- same VAL_FRACTION and same "don't shuffle-split a
continuous walk/drive" reasoning as perception.train's RELLIS split.
Does NOT reuse perception.train.build_multi_sequence_splits directly:
that function assumes frame indices are a contiguous range(n_frames)
starting at 0, which is FALSE for SemanticPOSS (sequence 00 starts at
000000, sequences 01-05 start at 000001 -- confirmed against the real
downloaded files, see perception.semanticposs_loader's own docstring).
This module works from the actual on-disk frame-id strings instead.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from perception.frame_cache import FrameCache, apply_radial_jitter_to_raw
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import (
    ChannelStats,
    _raw_channels,
    assemble_input_tensor,
    ground_prior_channel_from_points,
    normalize_raw_channels,
)
from perception.range_image import project_to_range_image
from perception.semanticposs_loader import (
    collect_semanticposs_frame_ids,
    load_semanticposs_labels,
    load_semanticposs_sweep,
)
from perception.taxonomy import semanticposs_label_ids_to_drishti
from perception.train import (
    AUG_BEAM_DROPOUT_PROB,
    AUG_JITTER_PROB,
    AUG_JITTER_RANGE,
    _apply_beam_dropout,
    _apply_radial_jitter,
    _apply_spatial_augmentation,
)
from sensor.sensor_model import SensorConfig

VAL_FRACTION = 0.15  # last 15% of EACH sequence's own frames, by frame-id order -- same convention as perception.train


def build_semanticposs_splits(sequence_dirs: List[Path]):
    """Per-sequence train/val split, using each sequence's REAL on-disk
    frame-id list (not a synthesised range -- see module docstring).
    Returns (train_items, val_items, per_seq_counts) where each item is
    (sequence_dir, frame_id) -- frame_id a string, not an int, unlike
    perception.train's RELLIS item tuples."""
    train_items: List[Tuple[Path, str]] = []
    val_items: List[Tuple[Path, str]] = []
    per_seq_counts: List[Tuple[Path, int]] = []
    for seq_dir in sequence_dirs:
        frame_ids = collect_semanticposs_frame_ids(seq_dir)
        n = len(frame_ids)
        n_val = max(1, int(round(n * VAL_FRACTION)))
        n_train = n - n_val
        train_items.extend((seq_dir, fid) for fid in frame_ids[:n_train])
        val_items.extend((seq_dir, fid) for fid in frame_ids[n_train:])
        per_seq_counts.append((seq_dir, n))
    return train_items, val_items, per_seq_counts


def _load_semanticposs_frame(sequence_dir: Path, frame_id: str, sm: SensorConfig):
    """Mirrors perception.train._load_frame exactly: sweep -> range
    image -> ground prior -> per-pixel target via the identical
    point_index rasterisation RELLIS and nuScenes both use."""
    sweep = load_semanticposs_sweep(sequence_dir, frame_id)
    raw_labels = load_semanticposs_labels(sequence_dir, frame_id)
    drishti_labels = semanticposs_label_ids_to_drishti(raw_labels)

    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

    target = np.zeros((img.H, img.W), dtype=np.int64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    target[touched] = drishti_labels[src]

    return img, ground, target


class SemanticPossSegDataset(Dataset):
    """Same (input_tensor, target, valid_mask) contract as
    RellisSegDataset / NuscenesSegDataset. `items` is a list of
    (sequence_dir, frame_id) pairs from build_semanticposs_splits."""

    def __init__(
        self,
        items: List[Tuple[Path, str]],
        sm: SensorConfig,
        stats: ChannelStats,
        is_train: bool = False,
        cache: "Optional[FrameCache]" = None,
    ):
        self.items = list(items)
        self.sm = sm
        self.stats = stats
        self.is_train = is_train
        self.cache = cache
        self._rng = random.Random()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        sequence_dir, frame_id = self.items[i]

        if self.cache is not None:
            def _compute():
                img, ground, target = _load_semanticposs_frame(sequence_dir, frame_id, self.sm)
                raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
                return {"raw": raw, "target": target, "valid_mask": img.valid_mask}

            # sequence_dir + frame_id is a stable, filesystem-safe key
            # (frame_id is already e.g. "000001"; sequence_dir.name is
            # e.g. "00") -- unlike RELLIS-3D's own frame_idx, this is
            # the REAL on-disk frame id, so cache keys stay stable even
            # given the sequence-00-starts-at-0/others-start-at-1 quirk
            # perception.semanticposs_loader's own docstring documents.
            key = f"{sequence_dir.name}_{frame_id}"
            cached = self.cache.get_or_compute(key, _compute)
            raw, target, valid_mask_np = cached["raw"], cached["target"], cached["valid_mask"]
            if self.is_train and self._rng.random() < AUG_JITTER_PROB:
                raw = apply_radial_jitter_to_raw(raw, self._rng.uniform(*AUG_JITTER_RANGE))
            tensor_np = normalize_raw_channels(raw, self.stats)
            tensor = torch.from_numpy(tensor_np).float()
            target_t = torch.from_numpy(target).long()
            valid_t = torch.from_numpy(valid_mask_np).bool()
        else:
            img, ground, target = _load_semanticposs_frame(sequence_dir, frame_id, self.sm)
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
