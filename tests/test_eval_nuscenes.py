"""
tests/test_eval_nuscenes.py

Tests for the nuScenes zero-shot evaluation pipeline:
1. Verifies build_lidarseg_lut mappings.
2. Verifies compute_metrics_from_matrix on synthetic confusion matrices.
3. Verifies forward pass and size-matching on HDL-32E (32, 1080) shapes.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("pyquaternion")
pytest.importorskip("nuscenes")

from eval.eval_nuscenes import compute_metrics_from_matrix
from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT
from perception.taxonomy import NUSCENES_LIDARSEG_TO_DRISHTI, DrishtiClass


def test_metrics_from_matrix():
    # 3x3 matrix
    # Class 0: TP=10, FP=2, FN=3 -> IoU = 10 / (10 + 2 + 3) = 10/15 = 2/3
    # Class 1: TP=20, FP=5, FN=5 -> IoU = 20 / (20 + 5 + 5) = 20/30 = 2/3
    # Class 2: TP=0, FP=5, FN=4 -> IoU = 0 / 9 = 0.0
    # Mean IoU = (2/3 + 2/3 + 0) / 3 = 4/9 = 0.4444
    mat = np.array([
        [10, 2, 1],
        [1, 20, 4],
        [1, 3, 0],
    ], dtype=np.int64)

    res = compute_metrics_from_matrix(mat)
    assert pytest.approx(res["miou"], rel=1e-3) == (10/15 + 20/30 + 0.0) / 3
    assert res["total_points"] == mat.sum()
    assert res["per_class"][0]["point_count"] == 13  # row sum


from perception.segnet import FusionSegNet, N_CLASSES_DEFAULT, N_INPUT_CHANNELS


def test_model_forward_hdl32e_shape():
    """Verify FusionSegNet accepts a 32x1080 range image without crashing."""
    model = FusionSegNet(n_classes=N_CLASSES_DEFAULT)
    model.eval()
    dummy_input = torch.randn(1, N_INPUT_CHANNELS, 32, 1080)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, N_CLASSES_DEFAULT, 32, 1080)


def test_taxonomy_mapping_completeness():
    """Ensure all 32 nuScenes-lidarseg categories are accounted for in taxonomy."""
    assert len(NUSCENES_LIDARSEG_TO_DRISHTI) == 32
    assert NUSCENES_LIDARSEG_TO_DRISHTI["flat.driveable_surface"] == DrishtiClass.DRIVABLE
    assert NUSCENES_LIDARSEG_TO_DRISHTI["static.vegetation"] == DrishtiClass.VEGETATION
    assert NUSCENES_LIDARSEG_TO_DRISHTI["vehicle.car"] == DrishtiClass.VEHICLE
    assert NUSCENES_LIDARSEG_TO_DRISHTI["human.pedestrian.adult"] == DrishtiClass.PEDESTRIAN
