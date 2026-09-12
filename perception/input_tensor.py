"""
perception/input_tensor.py

Ticket #27's original 9-channel network input tensor (x, y, z, range,
intensity, valid_mask, ground_prior, occlusion_count, occlusion_spread)
extended to 13 channels, per real-data validation
(`eval/validate_feature_hypotheses.py`, run BEFORE this change, not
after):

  - channel 4 ("intensity") now holds RANGE-CORRECTED intensity
    (`perception.reflectivity.range_corrected_intensity`), not raw --
    measured to improve DRIVABLE-vs-VEGETATION separability (the
    "Ground Paradox" axis from BASELINE_COMPARISON.md's zero-shot
    analysis) by ~38% (Fisher separability 0.206 -> 0.284) on real
    RELLIS-3D data. Same channel slot, so this is a value change, not a
    channel-count change.
  - channels 9-12 (new): normal_x, normal_y, normal_z, curvature
    (`perception.surface_geometry.compute_surface_geometry`) -- the
    single most strongly validated of the four checks: real data shows
    a clean, monotonic curvature ordering (DRIVABLE 0.2 < VEGETATION
    1.22 < NON_TRAVERSABLE 1.51 < STATIC_OBSTACLE 2.83) and a matching
    inverse ordering in |normal_z| (flatness), exactly matching the
    "rough/erratic surfaces are obstacles" hypothesis.

Adding channels 9-12 changes `N_CHANNELS` from 9 to 13, which requires
`perception.segnet.FusionSegNet`'s first conv layer to be rebuilt for
13 input channels -- see that module's `expand_stem_conv_for_checkpoint`
for how an existing 9-channel checkpoint's weights are preserved (not
discarded) across that change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from perception.ground_prior import GroundPriorResult
from perception.range_image import RangeImage
from perception.reflectivity import range_corrected_intensity
from perception.surface_geometry import compute_surface_geometry

CHANNEL_NAMES = (
    "x", "y", "z", "range", "intensity", "valid_mask",
    "ground_prior", "occlusion_count", "occlusion_spread",
    "normal_x", "normal_y", "normal_z", "curvature",
)
N_CHANNELS = len(CHANNEL_NAMES)
VALID_MASK_CHANNEL = CHANNEL_NAMES.index("valid_mask")


def ground_prior_channel_from_points(img: RangeImage, ground: GroundPriorResult) -> np.ndarray:
    """Project the per-point is_ground boolean into the range image's
    pixel grid via the SAME point_index mapping the projection already
    computed (Ticket #23) -- a hint channel, not a filter (Bible Part
    5.2: ground_prior feeds the network as input channel 7; it never
    strips points, it labels them)."""
    channel = np.zeros((img.H, img.W), dtype=np.float64)
    touched = img.point_index >= 0
    src = img.point_index[touched]
    channel[touched] = ground.is_ground[src].astype(np.float64)
    return channel


def _raw_channels(img: RangeImage, ground_prior_channel: np.ndarray) -> np.ndarray:
    """Stack the 13 raw (unnormalised) channels -> (13, H, W). Channel 4
    is RANGE-CORRECTED intensity (see module docstring), not raw."""
    calibrated_intensity = range_corrected_intensity(img.intensity, img.range)
    geom = compute_surface_geometry(img)
    return np.stack(
        [
            img.x, img.y, img.z, img.range, calibrated_intensity,
            img.valid_mask.astype(np.float64),
            ground_prior_channel,
            img.occlusion_count.astype(np.float64),
            img.occlusion_spread,
            geom.normal_x, geom.normal_y, geom.normal_z, geom.curvature,
        ],
        axis=0,
    )


@dataclass(frozen=True)
class ChannelStats:
    mean: List[float]  # length N_CHANNELS
    std: List[float]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ChannelStats":
        return ChannelStats(mean=list(d["mean"]), std=list(d["std"]))


def compute_channel_stats(raw_stacks: Sequence[np.ndarray], eps: float = 1e-6) -> ChannelStats:
    """Statistics over VALID pixels only, pooled across every frame
    given (Ticket #27: "computed over the training split ... saved to
    disk, not recomputed per batch"). Invalid pixels are placeholder
    zeros, not real signal -- including them would bias mean/std toward
    zero, the input-level form of Part 4.3's "empty pixels read as
    objects at the origin". `valid_mask` itself is excluded -- it is
    never normalised (see assemble_input_tensor).
    """
    means: List[float] = []
    stds: List[float] = []
    for c in range(N_CHANNELS):
        if c == VALID_MASK_CHANNEL:
            means.append(0.0)
            stds.append(1.0)
            continue
        values = []
        for stack in raw_stacks:
            valid = stack[VALID_MASK_CHANNEL] > 0.5
            values.append(stack[c][valid])
        pooled = np.concatenate(values) if values and any(v.size for v in values) else np.array([0.0])
        mean = float(pooled.mean()) if pooled.size else 0.0
        std = float(pooled.std())
        std = std if std > eps else 1.0
        means.append(mean)
        stds.append(std)
    return ChannelStats(mean=means, std=stds)


def save_stats(stats: ChannelStats, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2))


def load_stats(path: str | Path) -> ChannelStats:
    """Loads a saved ChannelStats -- PADDED to the CURRENT N_CHANNELS if
    the saved file is shorter (e.g. an old 9-channel checkpoint's own
    `channel_stats.json`, from before the reflectivity/surface-geometry
    channels existed). Padding with mean=0.0/std=1.0 is not a guess: an
    OLD checkpoint's stem conv has ZERO weight on the new channels (see
    `perception.segnet.expand_stem_conv_for_checkpoint`), so whatever
    normalisation those channels get literally cannot change that
    checkpoint's output -- padding just needs to not crash or produce
    NaN/inf, not be numerically meaningful for a checkpoint that never
    looks at these channels at all."""
    stats = ChannelStats.from_dict(json.loads(Path(path).read_text()))
    n_missing = N_CHANNELS - len(stats.mean)
    if n_missing > 0:
        stats = ChannelStats(mean=stats.mean + [0.0] * n_missing, std=stats.std + [1.0] * n_missing)
    return stats


def assemble_input_tensor(img: RangeImage, ground: GroundPriorResult, stats: ChannelStats) -> np.ndarray:
    """(13, H, W) tensor, normalised to ~zero-mean/unit-variance per
    channel using SAVED stats (never recomputed here -- Ticket #27
    "Watch out": per-batch normalisation makes train/inference statistics
    differ, which degrades silently). `valid_mask` is never normalised;
    it stays exactly 0/1 (Ticket #27's own explicit requirement).
    """
    ground_channel = ground_prior_channel_from_points(img, ground)
    raw = _raw_channels(img, ground_channel)

    out = np.empty_like(raw)
    for c in range(N_CHANNELS):
        if c == VALID_MASK_CHANNEL:
            out[c] = raw[c]
            continue
        out[c] = (raw[c] - stats.mean[c]) / stats.std[c]
    return out
