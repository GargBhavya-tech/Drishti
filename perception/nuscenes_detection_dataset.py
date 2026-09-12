"""
perception/nuscenes_detection_dataset.py

Dataset for the detection head: pairs each nuScenes-mini frame's
existing 13-channel input tensor (same assemble_input_tensor pipeline
perception.nuscenes_seg_dataset already uses) with REAL 3D box targets
(perception.nuscenes_boxes.build_box_targets). Deliberately a SEPARATE
dataset class from NuscenesSegDataset rather than bolting box targets
onto it -- this dataset is detection-only (nuScenes is the only one of
the three real datasets this session with real box annotations; RELLIS-
3D and SemanticPOSS ship no boxes at all), so mixing its item contract
into the shared 3-dataset segmentation path would force RELLIS/
SemanticPOSS's dataset classes to fake box targets they have no data
for. Kept separate, same way perception.train_joint.py keeps validation
per-domain rather than forcing one shape onto everything.

No augmentation applied (unlike the segmentation datasets) for this
first version -- radial jitter/spatial augmentation would need the
SAME transform applied consistently to the regression targets (an
offset target rotated/flipped along with its pixel), which is real,
buildable work not done in this pass. Flagged here rather than silently
assumed harmless; training on unaugmented real boxes for now.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from perception.frame_cache import FrameCache
from perception.ground_prior import compute_ground_prior
from perception.input_tensor import ChannelStats, assemble_input_tensor, normalize_raw_channels, _raw_channels, ground_prior_channel_from_points
from perception.nuscenes_boxes import build_box_targets
from perception.nuscenes_loader import load_nuscenes_sweep
from perception.range_image import project_to_range_image
from sensor.sensor_model import SensorConfig


class NuscenesDetectionDataset(Dataset):
    """items: list of nuScenes sample tokens (reuse
    perception.nuscenes_seg_dataset.build_nuscenes_scene_splits's own
    train/val token lists -- same scene-level split, no leakage, no
    second split function needed).

    No augmentation applied (see module docstring), so caching here is
    simpler than the segmentation datasets' own cache wiring -- the
    ENTIRE __getitem__ payload (raw stack + objectness + regression) is
    deterministic given the frame, with no post-cache jitter step
    needed."""

    def __init__(self, nusc, items: List[str], sm: SensorConfig, stats: ChannelStats, cache: "Optional[FrameCache]" = None):
        self.nusc = nusc
        self.items = list(items)
        self.sm = sm
        self.stats = stats
        self.cache = cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        sample_token = self.items[i]

        if self.cache is not None:
            def _compute():
                sweep = load_nuscenes_sweep(self.nusc, sample_token)
                img = project_to_range_image(sweep, self.sm)
                ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)
                raw = _raw_channels(img, ground_prior_channel_from_points(img, ground))
                box_targets = build_box_targets(self.nusc, sample_token, sweep.xyz, img)
                return {"raw": raw, "objectness": box_targets.objectness, "regression": box_targets.regression}

            cached = self.cache.get_or_compute(sample_token, _compute)
            tensor_np = normalize_raw_channels(cached["raw"], self.stats)
            tensor = torch.from_numpy(tensor_np).float()
            objectness = torch.from_numpy(cached["objectness"]).float()
            regression = torch.from_numpy(cached["regression"]).float()
            return tensor, objectness, regression

        sweep = load_nuscenes_sweep(self.nusc, sample_token)
        img = project_to_range_image(sweep, self.sm)
        ground = compute_ground_prior(sweep, n_azimuth_bins=img.W)

        tensor_np = assemble_input_tensor(img, ground, self.stats)
        tensor = torch.from_numpy(tensor_np).float()

        box_targets = build_box_targets(self.nusc, sample_token, sweep.xyz, img)
        objectness = torch.from_numpy(box_targets.objectness).float()
        regression = torch.from_numpy(box_targets.regression).float()

        return tensor, objectness, regression


def compute_real_pos_weight(nusc, items: List[str], sm: SensorConfig, stats: ChannelStats, sample_every: int = 1) -> float:
    """REAL measured negative/positive pixel ratio across (a sample of)
    `items`, for perception.detection_loss.DetectionLoss's pos_weight --
    computed from actual data, never guessed. Returns 1.0 (no reweighting)
    if zero positive pixels were found in the sample, with a printed
    warning -- silently dividing by zero would be worse than a flat,
    clearly-wrong-looking default."""
    total_pixels = 0
    total_positive = 0
    for sample_token in items[::sample_every]:
        sweep = load_nuscenes_sweep(nusc, sample_token)
        img = project_to_range_image(sweep, sm)
        box_targets = build_box_targets(nusc, sample_token, sweep.xyz, img)
        total_pixels += box_targets.objectness.size
        total_positive += int(box_targets.objectness.sum())
    if total_positive == 0:
        print("WARNING: zero positive (in-box) pixels found while computing pos_weight -- "
              "check DETECTABLE_DRISHTI_CLASSES / category mapping before training. Using pos_weight=1.0.")
        return 1.0
    return (total_pixels - total_positive) / total_positive
