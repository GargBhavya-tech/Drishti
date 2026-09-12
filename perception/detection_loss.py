"""
perception/detection_loss.py

Loss for perception/detection_head.py's output, deliberately simple --
Wilson et al. 2024's finding (cited in detection_head.py's own
docstring) that a plain classification loss for objectness plus
smooth-L1 for regression matches complex IoU-based losses for
range-view detection. No custom loss architecture was invented here.

Two terms:
1. Objectness: nn.BCEWithLogitsLoss with `pos_weight` set from the
   REAL measured positive/negative pixel ratio in a training batch
   (computed once, passed in) -- boxes cover a small fraction of any
   frame's pixels, so an unweighted BCE would trivially predict
   "nothing" everywhere and still get a low loss. This is the same
   "class imbalance needs weighting, but keep the mechanism simple"
   principle perception.losses.DrishtiSegLoss already applies to
   semantic segmentation, not a new idea introduced here.
2. Regression: smooth-L1, masked to objectness==1 pixels only (a
   background pixel's regression target is meaningless -- see
   perception.nuscenes_boxes.BoxTargets' own docstring -- and must
   never contribute gradient).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None, regression_weight: float = 1.0):
        super().__init__()
        pos_weight_t = torch.tensor(pos_weight) if pos_weight is not None else None
        self.objectness_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        self.regression_loss = nn.SmoothL1Loss(reduction="none")
        self.regression_weight = regression_weight

    def forward(
        self,
        pred: torch.Tensor,  # (B, 9, H, W) -- perception.detection_head.DetectionHead's raw output
        objectness_target: torch.Tensor,  # (B, H, W) float {0,1}
        regression_target: torch.Tensor,  # (B, 8, H, W)
    ) -> torch.Tensor:
        objectness_logit = pred[:, 0, :, :]
        regression_pred = pred[:, 1:, :, :]

        obj_loss = self.objectness_loss(objectness_logit, objectness_target)

        positive_mask = (objectness_target > 0.5).unsqueeze(1)  # (B, 1, H, W), broadcasts over the 8 regression channels
        n_positive = positive_mask.sum().clamp(min=1.0)
        reg_loss_per_element = self.regression_loss(regression_pred, regression_target)
        reg_loss = (reg_loss_per_element * positive_mask).sum() / (n_positive * regression_pred.shape[1])

        return obj_loss + self.regression_weight * reg_loss
