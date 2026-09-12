"""
perception/nuscenes_seg_dataset.py

nuScenes-mini training/fine-tuning dataset for FusionSegNet -- the
missing piece between perception/nuscenes_loader.py (sweep loading,
Ticket #2) and eval/eval_nuscenes.py (zero-shot EVAL only, no training
path existed until now).

Built to answer a real question raised mid-project: RELLIS-3D's
zero-shot transfer to nuScenes-mini was diagnosed as "the Ground
Paradox" (eval/out/nuscenes_zero_shot.json: 80.8% vegetation recall
alongside 0.33% drivable recall) -- the network learned "flat ground =
VEGETATION" because RELLIS-3D's own ground truth is ~85% grass/soil.
Fine-tuning on a small amount of REAL labeled nuScenes asphalt is a
direct test of that diagnosis; re-measuring zero-shot mIoU again is not.

Mirrors perception.train's RELLIS-specific `_load_frame` /
`RellisSegDataset` pattern exactly (same range-image projection, same
ground-prior computation, same `assemble_input_tensor` call, same
per-pixel target rasterisation via `img.point_index`) so this dataset
plugs into perception.train's own augmentation functions and
perception/train_nuscenes.py's training loop without either needing to
know anything dataset-specific beyond "how do I load one frame".

Dataset size reality (Ticket #2, confirmed again here): nuScenes-mini
is 10 scenes / ~404 lidarseg-annotated keyframes -- two orders of
magnitude smaller than RELLIS-3D's 11,522 training frames. This is a
FINE-TUNE dataset, not a from-scratch training set: perception/
train_nuscenes.py defaults to `--init-from-checkpoint` and a handful of
epochs, not 20 epochs from random init.

Val split: by whole SCENE, not by shuffled frame index -- the same
reasoning as perception.train's own RELLIS split (Ticket #30's "do not
shuffle-split a continuous drive"): consecutive keyframes within one
nuScenes scene are ~0.5s apart and nearly identical, so a random
frame-level split would leak near-duplicates into val exactly like it
would for RELLIS. Held-out scenes are whole scenes, never seen in
training.

Multiprocessing note: `NuscenesSegDataset` holds a reference to the
live `nusc` (NuScenes-devkit) object, which is large and not cheaply
picklable. This is safe with DataLoader(num_workers>0) ONLY because the
real training machine is Linux, where the default 'fork' start method
gives each worker a copy-on-write view of the already-loaded `nusc`
object instead of pickling it. On Windows (this repo's local dev
environment) DataLoader defaults to 'spawn', which WOULD try to pickle
`nusc` and fail or hang -- always pass --num-workers 0 for any local
Windows smoke test of this dataset; the real fine-tune run on
`drishti-gpu` (Linux) is unaffected.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

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
from perception.nuscenes_loader import load_nuscenes_sweep
from perception.range_image import project_to_range_image
from perception.train import (
    AUG_BEAM_DROPOUT_PROB,
    AUG_JITTER_PROB,
    AUG_JITTER_RANGE,
    _apply_beam_dropout,
    _apply_radial_jitter,
    _apply_spatial_augmentation,
)
from sensor.sensor_model import SensorConfig

VAL_SCENE_FRACTION = 0.2  # last ~20% of scenes (nuScenes' own scene order), WHOLE scenes held out


def collect_lidarseg_sample_tokens(nusc) -> List[Tuple[str, str]]:
    """Every (scene_token, sample_token) pair that has a lidarseg
    record, in each scene's own temporal order. v1.0-mini is fully
    lidarseg-annotated, but a missing record is skipped rather than
    assumed impossible and crashed on."""
    pairs: List[Tuple[str, str]] = []
    for scene in nusc.scene:
        cur = scene["first_sample_token"]
        while cur:
            sd_token = nusc.get("sample", cur)["data"]["LIDAR_TOP"]
            try:
                nusc.get("lidarseg", sd_token)
                pairs.append((scene["token"], cur))
            except KeyError:
                pass
            cur = nusc.get("sample", cur)["next"]
    return pairs


def build_nuscenes_scene_splits(nusc, val_scene_fraction: float = VAL_SCENE_FRACTION):
    """Whole-SCENE train/val split -- see module docstring. Returns
    (train_tokens, val_tokens, per_scene_counts) where per_scene_counts
    maps scene_token -> number of lidarseg keyframes in that scene."""
    pairs = collect_lidarseg_sample_tokens(nusc)
    # First-seen order, de-duplicated -- nuScenes doesn't guarantee
    # nusc.scene is in any particular order, so this fixes ONE
    # deterministic order (encounter order) rather than relying on
    # whatever order the devkit happens to iterate scenes in.
    scene_order = list(dict.fromkeys(scene_token for scene_token, _ in pairs))
    n_val_scenes = max(1, int(round(len(scene_order) * val_scene_fraction)))
    val_scene_tokens = set(scene_order[-n_val_scenes:])

    train_tokens: List[str] = []
    val_tokens: List[str] = []
    per_scene_counts: Dict[str, int] = {}
    for scene_token, sample_token in pairs:
        per_scene_counts[scene_token] = per_scene_counts.get(scene_token, 0) + 1
        (val_tokens if scene_token in val_scene_tokens else train_tokens).append(sample_token)

    return train_tokens, val_tokens, per_scene_counts


def _load_nuscenes_frame(nusc, sample_token: str, sm: SensorConfig, lut: np.ndarray):
    """Mirrors perception.train._load_frame exactly, for one nuScenes
    keyframe: sweep -> range image -> ground prior -> per-pixel target,
    via the identical point_index rasterisation RELLIS uses. `lut` is a
    perception.taxonomy.build_nuscenes_lidarseg_lut(nusc) result, passed
    in rather than rebuilt per-frame (it's the same 256-entry table for
    every frame in a given nuScenes version)."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]

    sweep = load_nuscenes_sweep(nusc, sample_token)

    lidarseg_record = nusc.get("lidarseg", sd_token)
    lidarseg_path = os.path.join(nusc.dataroot, lidarseg_record["filename"])
    raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
    drishti_labels = lut[raw_labels]

    img = project_to_range_image(sweep, sm)
    ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

    target = np.zeros((img.H, img.W), dtype=np.int64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    target[touched] = drishti_labels[src]

    return img, ground, target


class NuscenesSegDataset(Dataset):
    """Same (input_tensor, target, valid_mask) contract as
    perception.train.RellisSegDataset -- drops into an identically-
    shaped training loop unchanged (see perception/train_nuscenes.py).
    `items` is a list of nuScenes sample tokens, all pre-confirmed by
    the caller (build_nuscenes_scene_splits) to carry lidarseg records.

    See module docstring's multiprocessing note before setting
    num_workers > 0 on a non-Linux machine."""

    def __init__(
        self,
        nusc,
        items: List[str],
        sm: SensorConfig,
        stats: ChannelStats,
        lut: np.ndarray,
        is_train: bool = False,
        cache: "Optional[FrameCache]" = None,
    ):
        self.nusc = nusc
        self.items = list(items)
        self.sm = sm
        self.stats = stats
        self.lut = lut
        self.is_train = is_train
        self.cache = cache
        self._rng = random.Random()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        sample_token = self.items[i]

        if self.cache is not None:
            # Cache holds the RAW (pre-jitter, pre-normalisation) stack
            # -- see perception.frame_cache's own docstring on why
            # caching must happen before jitter, not after.
            def _compute():
                img, ground, target = _load_nuscenes_frame(self.nusc, sample_token, self.sm, self.lut)
                raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
                return {"raw": raw, "target": target, "valid_mask": img.valid_mask}

            cached = self.cache.get_or_compute(sample_token, _compute)
            raw, target, valid_mask_np = cached["raw"], cached["target"], cached["valid_mask"]
            if self.is_train and self._rng.random() < AUG_JITTER_PROB:
                raw = apply_radial_jitter_to_raw(raw, self._rng.uniform(*AUG_JITTER_RANGE))
            tensor_np = normalize_raw_channels(raw, self.stats)
            tensor = torch.from_numpy(tensor_np).float()
            target_t = torch.from_numpy(target).long()
            valid_t = torch.from_numpy(valid_mask_np).bool()
        else:
            img, ground, target = _load_nuscenes_frame(self.nusc, sample_token, self.sm, self.lut)
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
