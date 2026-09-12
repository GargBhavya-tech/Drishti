"""
perception/detection_decode.py

Turns perception.detection_head.DetectionHead's dense per-pixel output
into a real, discrete list of object instances -- this is what actually
answers "how many pedestrians in that frame", the gap this whole
detection-head build exists to close (Bible Part I / the deep-research
report on lightweight range-view detection).

Deliberately NOT classical clustering on the raw semantic mask -- the
research report's own analysis (Section 4.3) is specific about why that
fails: connected-components/DBSCAN on a semantic mask has no notion of
"where is the center of this object", so two objects that are adjacent
or touching in the projection merge into one blob (the "touching
object" dilemma). This decode step instead does PEAK EXTRACTION on the
LEARNED objectness map -- the same family of approach CenterPoint uses
for its own final decode step: a trained network naturally learns to
concentrate objectness near true object centers, so distinct local
maxima correspond to distinct objects even when their pixel footprints
touch or overlap, in a way a footprint-only clustering method cannot
see.

Method (deliberately simple, matching this whole build's "Wilson et al.
2024: simplicity works" stance): threshold objectness, find local
maxima via max-pooling (a standard, cheap non-max-suppression trick --
a pixel survives only if it equals the max of its own neighborhood),
suppress duplicate peaks within a minimum pixel radius, then read off
each surviving peak's own regression channels directly as that
instance's box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from perception.nuscenes_boxes import DETECTABLE_DRISHTI_CLASSES


@dataclass
class DetectedObject:
    row: int
    col: int
    objectness: float
    center_xyz: np.ndarray  # (3,) -- point_xyz + regressed offset, i.e. the box's own regressed center
    wlh: np.ndarray  # (3,)
    yaw: float
    drishti_class: int  # read from the segmentation head's own prediction at (row, col)


def decode_detections(
    detection_pred: torch.Tensor,  # (9, H, W) -- ONE frame's raw DetectionHead output (no batch dim)
    seg_pred_classes: torch.Tensor,  # (H, W) int -- argmax of the segmentation head's own output, same frame
    point_xyz_at_pixel: np.ndarray,  # (H, W, 3) -- the real 3D point each pixel's regression target was built against (nan where no valid point)
    objectness_threshold: float = 0.5,
    nms_pool_size: int = 5,
) -> List[DetectedObject]:
    """Real-valued objectness/regression -> a list of discrete objects.
    `point_xyz_at_pixel` must come from the SAME RangeImage the
    detection head's input was projected from (each pixel's underlying
    3D point, or NaN for an empty pixel) -- the regressed offset is
    relative to that point, not to the pixel's row/col alone."""
    objectness = torch.sigmoid(detection_pred[0])
    regression = detection_pred[1:]

    pooled = F.max_pool2d(
        objectness.unsqueeze(0).unsqueeze(0), kernel_size=nms_pool_size, stride=1, padding=nms_pool_size // 2
    ).squeeze(0).squeeze(0)
    is_peak = (objectness == pooled) & (objectness > objectness_threshold)

    rows, cols = torch.nonzero(is_peak, as_tuple=True)
    detections: List[DetectedObject] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        seg_class = int(seg_pred_classes[r, c])
        if seg_class not in DETECTABLE_DRISHTI_CLASSES:
            # The objectness head can still fire on background (e.g.
            # VEGETATION) before it has fully converged -- a real,
            # observed failure mode (eval/checkpoint_detection_decode.py
            # first run reported VEGETATION "objects" before this check
            # was added). The segmentation head's own class prediction
            # is the authority on whether a pixel is even instance-like
            # at all; an objectness peak on a non-detectable class is a
            # false positive, not a detection, and must never be
            # reported as one.
            continue
        pt = point_xyz_at_pixel[r, c]
        if np.any(np.isnan(pt)):
            continue  # a "peak" at a pixel with no real return underneath it is not a real detection
        dx, dy, dz, w, l, h, sin_yaw, cos_yaw = regression[:, r, c].tolist()
        center = pt + np.array([dx, dy, dz])
        yaw = float(np.arctan2(sin_yaw, cos_yaw))
        detections.append(
            DetectedObject(
                row=r,
                col=c,
                objectness=float(objectness[r, c]),
                center_xyz=center,
                wlh=np.array([w, l, h]),
                yaw=yaw,
                drishti_class=int(seg_pred_classes[r, c]),
            )
        )
    return detections
