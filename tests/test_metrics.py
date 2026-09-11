"""
tests/test_metrics.py

Ticket #57 tests, per the Build Map's own instruction: "hand-verify the
metric on a small confusion matrix you compute on paper." The hand
computation, worked here in the docstring before being asserted:

    y_true = [0,0,1,1,2,2]
    y_pred = [0,0,1,2,2,2]

    Confusion (rows=truth, cols=pred):
        [[2, 0, 0],
         [0, 1, 1],
         [0, 0, 2]]

    TP = [2, 1, 2]
    col sums (predicted totals) = [2, 1, 3]  ->  FP = colsum-TP = [0, 0, 1]
    row sums (truth totals)     = [2, 2, 2]  ->  FN = rowsum-TP = [0, 1, 0]

    IoU_0 = 2/(2+0+0) = 1.0
    IoU_1 = 1/(1+0+1) = 0.5
    IoU_2 = 2/(2+1+0) = 0.666...
    mIoU  = (1.0 + 0.5 + 0.6667) / 3 = 0.72222...

Plus the distance-band boundaries derived from a real Clipmap, and the
"spot-check five frames' verdicts against your own read" instruction
implemented as five hand-picked (range, true, pred) triples checked
against `band_index_for_range` by inspection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.metrics import (
    ConfusionMatrix,
    band_boundaries_m,
    band_index_for_range,
    compute_miou_by_distance_band,
)
from grid.clipmap import Clipmap
from sensor.sensor_model import load_sensor_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def hdl64e():
    return load_sensor_config(CONFIGS / "sensor_hdl64e.yaml")


# ---------------------------------------------------------------------------
# The hand-computed 3x3 case -- Build Map's own explicit instruction.
# ---------------------------------------------------------------------------


def test_hand_computed_confusion_matrix_and_miou():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])

    cm = ConfusionMatrix.from_predictions(y_true, y_pred, n_classes=3)

    expected_matrix = np.array([[2, 0, 0], [0, 1, 1], [0, 0, 2]])
    assert np.array_equal(cm.matrix, expected_matrix)

    iou = cm.iou_per_class()
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(0.5)
    assert iou[2] == pytest.approx(2.0 / 3.0)

    assert cm.miou() == pytest.approx((1.0 + 0.5 + 2.0 / 3.0) / 3.0)


def test_a_class_absent_from_both_truth_and_prediction_is_nan_not_zero():
    # Classes 0 and 1 only -- class 2 never appears anywhere.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = ConfusionMatrix.from_predictions(y_true, y_pred, n_classes=3)
    iou = cm.iou_per_class()
    assert np.isnan(iou[2])
    # mIoU must average only the DEFINED classes (0, 1), not treat the
    # absent class as a zero that drags the average down.
    assert cm.miou() == pytest.approx(np.nanmean(iou[:2]))


def test_accumulate_confusion_matches_from_predictions_for_a_larger_random_case():
    rng = np.random.default_rng(0)
    n_classes = 5
    y_true = rng.integers(0, n_classes, size=10_000)
    # A predictor correlated with truth but not perfect, so a real
    # confusion matrix (not a diagonal-only one) is exercised.
    noise = rng.integers(0, n_classes, size=10_000)
    y_pred = np.where(rng.random(10_000) < 0.8, y_true, noise)

    cm = ConfusionMatrix.from_predictions(y_true, y_pred, n_classes)
    assert cm.matrix.sum() == 10_000
    assert 0.0 <= cm.miou() <= 1.0


# ---------------------------------------------------------------------------
# Distance-band boundaries -- derived from a real Clipmap's own schedule.
# ---------------------------------------------------------------------------


def test_band_boundaries_match_the_clipmaps_own_level_half_extents(hdl64e):
    cm = Clipmap(hdl64e, n_levels=4, N=512, c0=0.05)
    boundaries = band_boundaries_m(cm)
    assert boundaries == pytest.approx([12.8, 25.6, 51.2, 102.4])


def test_five_hand_picked_ranges_land_in_the_expected_band():
    boundaries = [12.8, 25.6, 51.2, 102.4]
    # Spot-check, by inspection, five (range -> expected band) cases --
    # Build Map's own "spot-check five frames' verdicts against your
    # own read", adapted to this module's own unit of work (a range,
    # not a frame). Bands: 0=[0,12.8), 1=[12.8,25.6), 2=[25.6,51.2),
    # 3=[51.2, inf).
    cases = [
        (5.0, 0),     # well inside L0's ring
        (12.8, 1),    # exactly on the L0/L1 boundary -- belongs to the FAR side
        (20.0, 1),    # inside L1's ring
        (60.0, 3),    # inside L3's ring (between 51.2 and 102.4)
        (150.0, 3),   # beyond every boundary -- the last, open-ended band
    ]
    for r, expected_band in cases:
        assert band_index_for_range(r, boundaries) == expected_band


# ---------------------------------------------------------------------------
# Per-band mIoU on a small synthetic multi-band dataset.
# ---------------------------------------------------------------------------


def test_compute_miou_by_distance_band_isolates_each_bands_own_accuracy():
    boundaries = [12.8, 25.6, 51.2, 102.4]
    # Band 0: perfect predictions. Band 3: always wrong (class 0 truth,
    # class 1 predicted). The two bands' mIoU must differ accordingly
    # -- a bug that pools everything into one matrix would average
    # these together and hide the band-3 failure.
    ranges_m = np.array([5.0, 5.0, 5.0, 200.0, 200.0, 200.0])
    y_true = np.array([0, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 1])

    report = compute_miou_by_distance_band(ranges_m, y_true, y_pred, n_classes=2, boundaries=boundaries)
    miou = report.miou_by_band()

    assert miou[0] == pytest.approx(1.0)  # band 0: perfect
    assert miou[3] == pytest.approx(0.0)  # band 3: always wrong
    # Bands 1 and 2 have no data at all -- their mIoU must be NaN, not
    # a fabricated 0 or 1 that implies a verdict on no evidence.
    assert np.isnan(report.confusion_by_band[1].miou())
    assert np.isnan(report.confusion_by_band[2].miou())
