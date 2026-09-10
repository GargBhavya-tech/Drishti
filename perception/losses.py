"""
perception/losses.py

Ticket #29 -- loss stack: Lovász-Softmax (primary) + confidence-weighted
cross-entropy (secondary) + deep-supervision auxiliary term at 1/8 scale
(Lovász again -- Bible Part 5.3: "with Lovász applied there too it
enforces coarse structural correctness before the full-resolution
decoder refines detail").

Lovász-Softmax is NOT in `FusionSegNet_v5 (1).ipynb` -- that notebook's
own loss cell ("## 8. Loss Functions") uses Dice + Focal + label-smoothed
CE, which Bible Part 5.4 explicitly rejects ("Dice loss is insufficient
at this degree of imbalance ... Lovász-Softmax optimises the metric
directly"). Implemented fresh here, following the standard formulation
from Berman, Triki & Blaschko, "The Lovász-Softmax loss" (CVPR 2018).

What IS reused from the notebook: the confidence-weighting MECHANISM in
its `LabelSmoothingCE` ("CLEVER IDEA 1: multiply by confidence") -- that
per-pixel multiply is exactly Bible Part 5.4's "confidence-weighted
cross-entropy ... reused from FusionSegNet's pseudo-label weighting,
repurposed to weight by point-density confidence." Label smoothing
itself is dropped -- the Bible's own description is just "cross-
entropy", with no mention of smoothing.

Confidence input: Part 5.4 ties this to kappa (Part 11's sparsity-
derived confidence), which isn't built yet (Ticket #38/39, Phase 5).
`confidence` is accordingly an OPTIONAL argument here (defaults to
uniform 1.0, i.e. plain CE) -- whatever computes real kappa later plugs
in without changing this module.

Masking discipline (Ticket #29 "Watch out"): every term is masked by
SELECTING only valid pixels before any softmax/loss computation, not by
zeroing logits first -- zeroing still lets invalid pixels influence the
softmax denominator's gradient. Selecting excludes them from the
computation graph entirely.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_valid(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor, extra: Optional[torch.Tensor] = None):
    """(B, C, H, W) logits + (B, H, W) targets/valid_mask -> flattened,
    valid-only (N, C) logits and (N,) targets (and (N,) extra, if given).
    This IS the masking -- invalid pixels never enter any computation
    below, not even via a zeroed-but-still-differentiated softmax term.
    """
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    targets_flat = targets.reshape(-1)
    valid_flat = valid_mask.reshape(-1).bool()

    logits_valid = logits_flat[valid_flat]
    targets_valid = targets_flat[valid_flat]
    if extra is not None:
        extra_valid = extra.reshape(-1)[valid_flat]
        return logits_valid, targets_valid, extra_valid
    return logits_valid, targets_valid, None


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """The Lovász extension's gradient w.r.t. sorted errors (Berman et al.
    2018, eq. 3-4 -- the standard reference-implementation form)."""
    p = gt_sorted.shape[0]
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """probs: (N, C) softmax probabilities, already valid-pixel-only.
    labels: (N,) int class ids. Averages the per-class Lovász hinge loss
    over classes actually present in `labels` -- classes absent from this
    batch contribute nothing (there is no ground truth to rank against)."""
    n_classes = probs.shape[1]
    losses = []
    for c in range(n_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probs.sum() * 0.0  # no classes present -- zero loss, still differentiable
    return torch.stack(losses).mean()


def lovasz_softmax_loss(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """logits: (B, C, H, W). targets/valid_mask: (B, H, W)."""
    logits_v, targets_v, _ = _flatten_valid(logits, targets, valid_mask)
    if logits_v.shape[0] == 0:
        return logits.sum() * 0.0
    probs = F.softmax(logits_v, dim=1)
    return lovasz_softmax_flat(probs, targets_v)


def confidence_weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """logits: (B, C, H, W). targets/valid_mask/confidence: (B, H, W).
    confidence defaults to uniform 1.0 (plain CE) until Ticket #38/39's
    kappa exists to supply a real one."""
    logits_v, targets_v, conf_v = _flatten_valid(logits, targets, valid_mask, confidence)
    if logits_v.shape[0] == 0:
        return logits.sum() * 0.0
    per_pixel = F.cross_entropy(logits_v, targets_v, reduction="none")
    if conf_v is not None:
        per_pixel = per_pixel * conf_v
    return per_pixel.mean()


class DrishtiSegLoss(nn.Module):
    """Bible Part 5.4's full stack: Lovász (primary) + confidence-
    weighted CE (secondary) + Lovász on the aux head at its own
    resolution (small weight -- deep supervision, not a second primary
    signal). Weights are not given exact numbers by the Bible; chosen
    reasonably and left configurable rather than hardcoded as unlabelled
    magic numbers -- tune during actual training (Ticket #30)."""

    def __init__(self, ce_weight: float = 0.5, aux_weight: float = 0.4):
        super().__init__()
        self.ce_weight = ce_weight
        self.aux_weight = aux_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        primary = lovasz_softmax_loss(logits, targets, valid_mask)
        secondary = confidence_weighted_ce_loss(logits, targets, valid_mask, confidence)
        total = primary + self.ce_weight * secondary

        if aux_logits is not None:
            H_aux, W_aux = aux_logits.shape[2:]
            targets_aux = F.interpolate(
                targets.float().unsqueeze(1), size=(H_aux, W_aux), mode="nearest"
            ).squeeze(1).long()
            valid_aux = F.interpolate(
                valid_mask.float().unsqueeze(1), size=(H_aux, W_aux), mode="nearest"
            ).squeeze(1).bool()
            aux_loss = lovasz_softmax_loss(aux_logits, targets_aux, valid_aux)
            total = total + self.aux_weight * aux_loss

        return total
