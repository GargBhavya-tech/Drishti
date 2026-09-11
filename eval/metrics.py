"""
eval/metrics.py

Ticket #57 -- confusion matrix and per-class IoU, BINNED BY THE
CLIPMAP'S OWN LEVEL BOUNDARIES rather than arbitrary distance bins --
so accuracy is reported per RESOLUTION LEVEL, directly answering the
PS's "accuracy across varying distances" requirement in the map's own
native units instead of an unrelated round-number grid.

Each level's own half-extent (N/2 * c_l, the radius from ego out to
that level's window edge) gives the band boundary: for this project's
default schedule (N=512, c0=0.05m, 4 levels) that is exactly
[12.8, 25.6, 51.2, 102.4] m -- verified directly against a real
Clipmap, not hand-copied from the Build Map's own prose.

Watch out (Build Map's own words): "hand-verify the metric on a small
confusion matrix you compute on paper. A buggy metrics harness produces
confidently wrong numbers and those go straight onto slides." This
module's own test suite includes exactly that: a 3x3 confusion matrix
worked by hand before writing the assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from grid.clipmap import Clipmap


def band_boundaries_m(cm: Clipmap) -> List[float]:
    """Each level's own half-extent (N/2 * c_l) -- the radius from ego
    out to that level's window edge, in ascending order (levels are
    already stored finest-to-coarsest, and cell size increases
    monotonically with level, so no separate sort is needed)."""
    return [cm.N / 2 * lvl.cell_size_m for lvl in cm.levels]


def band_index_for_range(r_m: float, boundaries: Sequence[float]) -> int:
    """Which band a range falls into: band 0 is [0, boundaries[0]),
    band i is [boundaries[i-1], boundaries[i]) for i < len(boundaries),
    and the LAST band is [boundaries[-2], infinity) -- it has no outer
    edge, since it extends to wherever the sensor's own usable range
    stops, not to a further arbitrary cutoff."""
    for i, b in enumerate(boundaries):
        if r_m < b:
            return i
    return len(boundaries) - 1


@dataclass(frozen=True)
class ConfusionMatrix:
    matrix: np.ndarray  # (n_classes, n_classes), rows=truth, cols=predicted
    n_classes: int

    @staticmethod
    def from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> "ConfusionMatrix":
        matrix = accumulate_confusion(np.asarray(y_true), np.asarray(y_pred), n_classes)
        return ConfusionMatrix(matrix=matrix, n_classes=n_classes)

    def iou_per_class(self) -> np.ndarray:
        """IoU_c = TP_c / (TP_c + FP_c + FN_c). NaN (not zero) for a
        class with no true AND no predicted pixels: a class absent from
        BOTH truth and prediction carries no information about model
        accuracy either way, and averaging it in as 0 would understate
        accuracy for a reason that has nothing to do with the model."""
        m = self.matrix
        tp = np.diag(m).astype(np.float64)
        fp = m.sum(axis=0).astype(np.float64) - tp
        fn = m.sum(axis=1).astype(np.float64) - tp
        denom = tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def miou(self) -> float:
        iou = self.iou_per_class()
        valid = iou[~np.isnan(iou)]
        if valid.size == 0:
            return float("nan")
        return float(np.mean(valid))


def accumulate_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Vectorised confusion-matrix accumulation (`np.bincount` on a
    combined truth*n_classes+pred index) -- no Python loop over
    points, matching this project's own established convention
    (`grid.scatter`) for anything that scales with point count."""
    if matrix is None:
        matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    combined = y_true[valid].astype(np.int64) * n_classes + y_pred[valid].astype(np.int64)
    counts = np.bincount(combined, minlength=n_classes * n_classes)
    matrix = matrix + counts[: n_classes * n_classes].reshape(n_classes, n_classes)
    return matrix


@dataclass(frozen=True)
class DistanceBandReport:
    boundaries_m: List[float]
    confusion_by_band: Dict[int, ConfusionMatrix]

    def miou_by_band(self) -> Dict[int, float]:
        return {band: cm.miou() for band, cm in self.confusion_by_band.items()}


def compute_miou_by_distance_band(
    ranges_m: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    boundaries: Sequence[float],
) -> DistanceBandReport:
    """Bins every point by its own range into a distance band
    (vectorised via `np.searchsorted`, matching `band_index_for_range`'s
    scalar semantics exactly), then computes one confusion matrix and
    mIoU PER BAND -- so accuracy "across varying distances" (the PS's
    own requirement) is reported natively, never averaged away."""
    boundaries = list(boundaries)
    ranges_m = np.asarray(ranges_m)
    band_idx = np.searchsorted(boundaries, ranges_m, side="right")
    band_idx = np.clip(band_idx, 0, len(boundaries) - 1)

    confusion_by_band: Dict[int, ConfusionMatrix] = {}
    for band in range(len(boundaries)):
        mask = band_idx == band
        matrix = accumulate_confusion(
            np.asarray(y_true)[mask], np.asarray(y_pred)[mask], n_classes
        )
        confusion_by_band[band] = ConfusionMatrix(matrix=matrix, n_classes=n_classes)

    return DistanceBandReport(boundaries_m=boundaries, confusion_by_band=confusion_by_band)
