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
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """logits: (B, C, H, W). targets/valid_mask/confidence: (B, H, W).
    confidence defaults to uniform 1.0 (plain CE) until Ticket #38/39's
    kappa exists to supply a real one.

    `class_weight`: an OPTIONAL (C,) per-class weight, kept as a
    SEPARATE argument from `confidence` rather than folded into it --
    confidence is reserved for kappa's own eventual per-PIXEL signal
    (Part 5.4), while class_weight is a per-CLASS constant addressing
    training-set imbalance (Bible Part 5.4's own framing: Lovász
    already handles imbalance for the primary term by averaging
    per-class IoU unweighted across classes present -- see
    `lovasz_softmax_flat` -- so class_weight is scoped to THIS
    secondary term only, not applied to Lovász, which would double-
    count the same correction). Passed straight to `F.cross_entropy`'s
    own `weight` argument, which already excludes absent classes
    correctly via the normal CE gradient (no separate handling needed
    for classes 8/9, which never appear in `targets` by taxonomy
    design -- see perception/taxonomy.py)."""
    logits_v, targets_v, conf_v = _flatten_valid(logits, targets, valid_mask, confidence)
    if logits_v.shape[0] == 0:
        return logits.sum() * 0.0
    per_pixel = F.cross_entropy(logits_v, targets_v, weight=class_weight, reduction="none")
    if conf_v is not None:
        per_pixel = per_pixel * conf_v
    return per_pixel.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 2.0,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard focal loss (Lin et al. 2017): FL = (1 - p_t)^gamma * CE,
    where p_t = exp(-CE) recovers the model's own predicted probability
    of the true class from the per-pixel CE value directly (no separate
    softmax needed). Down-weights the (already-easy, already-numerous)
    well-classified pixels' contribution and concentrates gradient on
    hard/rare ones -- added specifically for class 4 (STATIC_OBSTACLE),
    which real-data validation (`eval/validate_feature_hypotheses.py`,
    check C) found is 0.051% of all training pixels, and which stayed at
    EXACTLY 0.0 IoU across all 20 epochs of a fine-tune that used class
    weighting alone (`checkpoints_multi_v2/training_log.jsonl`) -- i.e.
    reweighting the loss was not enough by itself; this is the next,
    complementary lever (used alongside class_weight, not instead of
    it -- the two address different things: class_weight scales a
    class's overall gradient magnitude, gamma re-shapes the PER-PIXEL
    weighting toward hard examples within any class).

    Same masking discipline as every other loss here: `_flatten_valid`
    excludes invalid pixels from the computation graph entirely."""
    logits_v, targets_v, _ = _flatten_valid(logits, targets, valid_mask)
    if logits_v.shape[0] == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits_v, targets_v, weight=class_weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


class DrishtiSegLoss(nn.Module):
    """Bible Part 5.4's full stack: Lovász (primary) + confidence-
    weighted CE (secondary) + Lovász on the aux head at its own
    resolution (small weight -- deep supervision, not a second primary
    signal). Weights are not given exact numbers by the Bible; chosen
    reasonably and left configurable rather than hardcoded as unlabelled
    magic numbers -- tune during actual training (Ticket #30)."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        aux_weight: float = 0.4,
        class_weight: Optional[torch.Tensor] = None,
        focal_gamma: Optional[float] = None,
        focal_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.aux_weight = aux_weight
        # None (default) keeps this term OFF entirely -- existing
        # callers/tests that construct DrishtiSegLoss without this
        # argument get byte-identical behaviour to before it existed.
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        # Registered as a buffer (not a plain attribute) so it moves
        # with the module under .to(device) -- a class_weight left on
        # the wrong device would raise at the first forward() call
        # under AMP/CUDA rather than silently doing nothing.
        self.register_buffer("class_weight", class_weight, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        primary = lovasz_softmax_loss(logits, targets, valid_mask)
        secondary = confidence_weighted_ce_loss(logits, targets, valid_mask, confidence, class_weight=self.class_weight)
        total = primary + self.ce_weight * secondary

        if self.focal_gamma is not None:
            focal = focal_loss(logits, targets, valid_mask, gamma=self.focal_gamma, class_weight=self.class_weight)
            total = total + self.focal_weight * focal

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
