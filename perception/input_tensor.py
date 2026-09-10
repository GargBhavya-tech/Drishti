"""
perception/input_tensor.py

Ticket #27 -- assemble the 9-channel network input tensor: x, y, z,
range, intensity, valid_mask, ground_prior, occlusion_count,
occlusion_spread. Motion residual channels are cut from this build
(Build Map Ticket #27's own note) -- 9 channels, not 14; adding them
later is additive, not a rewrite.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from perception.ground_prior import GroundPriorResult
from perception.range_image import RangeImage

CHANNEL_NAMES = (
    "x", "y", "z", "range", "intensity", "valid_mask",
    "ground_prior", "occlusion_count", "occlusion_spread",
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
    """Stack the 9 raw (unnormalised) channels -> (9, H, W)."""
    return np.stack(
        [
            img.x, img.y, img.z, img.range, img.intensity,
            img.valid_mask.astype(np.float64),
            ground_prior_channel,
            img.occlusion_count.astype(np.float64),
            img.occlusion_spread,
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
    return ChannelStats.from_dict(json.loads(Path(path).read_text()))


def assemble_input_tensor(img: RangeImage, ground: GroundPriorResult, stats: ChannelStats) -> np.ndarray:
    """(9, H, W) tensor, normalised to ~zero-mean/unit-variance per
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
