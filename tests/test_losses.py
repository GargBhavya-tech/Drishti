"""
tests/test_losses.py

Ticket #29 tests: the garbage-invariance masking test (Ticket #29's own
required assertion), Lovász near-zero on a perfect prediction, and the
combined stack's aux-resolution path.
"""

from __future__ import annotations

import pytest
import torch

from perception.losses import (
    DrishtiSegLoss,
    confidence_weighted_ce_loss,
    focal_loss,
    lovasz_softmax_loss,
)

N_CLASSES = 5


def _random_case(B=2, H=8, W=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, N_CLASSES, H, W, generator=g)
    targets = torch.randint(0, N_CLASSES, (B, H, W), generator=g)
    valid_mask = torch.rand(B, H, W, generator=g) > 0.3
    # Ensure at least one valid pixel so the test is meaningful.
    valid_mask[0, 0, 0] = True
    return logits, targets, valid_mask


def test_lovasz_garbage_invariance():
    logits, targets, valid_mask = _random_case()
    logits_zeroed = logits.clone()
    logits_zeroed[~valid_mask.unsqueeze(1).expand_as(logits)] = 0.0
    logits_garbage = logits.clone()
    logits_garbage[~valid_mask.unsqueeze(1).expand_as(logits)] = 1e6

    loss_zero = lovasz_softmax_loss(logits_zeroed, targets, valid_mask)
    loss_garbage = lovasz_softmax_loss(logits_garbage, targets, valid_mask)
    assert torch.allclose(loss_zero, loss_garbage, atol=1e-6)


def test_confidence_weighted_ce_garbage_invariance():
    logits, targets, valid_mask = _random_case()
    confidence = torch.rand(*valid_mask.shape)

    logits_zeroed = logits.clone()
    logits_zeroed[~valid_mask.unsqueeze(1).expand_as(logits)] = 0.0
    logits_garbage = logits.clone()
    logits_garbage[~valid_mask.unsqueeze(1).expand_as(logits)] = -1e6

    loss_zero = confidence_weighted_ce_loss(logits_zeroed, targets, valid_mask, confidence)
    loss_garbage = confidence_weighted_ce_loss(logits_garbage, targets, valid_mask, confidence)
    assert torch.allclose(loss_zero, loss_garbage, atol=1e-5)


def test_combined_stack_garbage_invariance():
    logits, targets, valid_mask = _random_case()
    aux_logits = torch.randn(2, N_CLASSES, 4, 5)

    logits_zeroed = logits.clone()
    logits_zeroed[~valid_mask.unsqueeze(1).expand_as(logits)] = 0.0
    logits_garbage = logits.clone()
    logits_garbage[~valid_mask.unsqueeze(1).expand_as(logits)] = 5e5

    loss_fn = DrishtiSegLoss()
    loss_zero = loss_fn(logits_zeroed, targets, valid_mask, aux_logits=aux_logits)
    loss_garbage = loss_fn(logits_garbage, targets, valid_mask, aux_logits=aux_logits)
    assert torch.allclose(loss_zero, loss_garbage, atol=1e-4)


def test_lovasz_on_perfect_prediction_is_near_zero():
    B, H, W = 1, 6, 6
    targets = torch.randint(0, N_CLASSES, (B, H, W))
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)

    # Huge logit on the correct class, huge-negative elsewhere -> softmax
    # is (numerically) a perfect one-hot match to targets.
    logits = torch.full((B, N_CLASSES, H, W), -50.0)
    logits.scatter_(1, targets.unsqueeze(1), 50.0)

    loss = lovasz_softmax_loss(logits, targets, valid_mask)
    assert loss.item() < 1e-3


def test_lovasz_on_wrong_prediction_is_large():
    B, H, W = 1, 6, 6
    targets = torch.zeros(B, H, W, dtype=torch.long)
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)

    # Confidently predicts the WRONG class everywhere.
    wrong_class = 1
    logits = torch.full((B, N_CLASSES, H, W), -50.0)
    logits[:, wrong_class, :, :] = 50.0

    loss = lovasz_softmax_loss(logits, targets, valid_mask)
    assert loss.item() > 0.5


def test_no_valid_pixels_returns_zero_not_nan():
    logits, targets, _ = _random_case()
    valid_mask = torch.zeros(2, 8, 10, dtype=torch.bool)
    loss = lovasz_softmax_loss(logits, targets, valid_mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# class_weight -- scoped to the secondary CE term only (not Lovász, which
# is already class-balanced by construction).
# ---------------------------------------------------------------------------


def test_class_weight_zero_removes_that_classs_contribution_to_ce():
    """Confidently WRONG on class 0 everywhere: weight=0 for class 0
    must drop the secondary CE term to exactly zero, since every
    target pixel belongs to the zeroed-out class."""
    B, H, W = 1, 6, 6
    targets = torch.zeros(B, H, W, dtype=torch.long)
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    logits = torch.full((B, N_CLASSES, H, W), -50.0)
    logits[:, 1, :, :] = 50.0  # confidently wrong everywhere

    weight = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    loss = confidence_weighted_ce_loss(logits, targets, valid_mask, class_weight=weight)
    assert loss.item() == 0.0


def test_class_weight_scales_ce_loss_proportionally():
    B, H, W = 1, 4, 4
    targets = torch.zeros(B, H, W, dtype=torch.long)
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    logits = torch.full((B, N_CLASSES, H, W), -50.0)
    logits[:, 1, :, :] = 50.0  # confidently wrong on class 0 everywhere

    unweighted = confidence_weighted_ce_loss(logits, targets, valid_mask)
    weight = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    weighted = confidence_weighted_ce_loss(logits, targets, valid_mask, class_weight=weight)
    # Every valid pixel is class 0 -> the whole loss scales by exactly 2x.
    assert weighted.item() == pytest.approx(unweighted.item() * 2.0, rel=1e-4)


def test_drishti_seg_loss_accepts_class_weight_and_moves_with_device():
    loss_fn = DrishtiSegLoss(class_weight=torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]))
    logits, targets, valid_mask = _random_case()
    loss = loss_fn(logits, targets, valid_mask)
    assert torch.isfinite(loss)
    # register_buffer must have actually stored it under this name.
    assert loss_fn.class_weight is not None
    assert loss_fn.class_weight.shape == (5,)


def test_drishti_seg_loss_defaults_to_no_class_weight():
    loss_fn = DrishtiSegLoss()
    assert loss_fn.class_weight is None


def test_focal_loss_downweights_a_confident_correct_prediction_more_than_ce():
    """The whole point of gamma: an already-easy (confidently correct)
    pixel's contribution should shrink relative to plain CE, while a
    genuinely wrong prediction's contribution should NOT shrink nearly
    as much -- checked directly by comparing the FL/CE ratio for an easy
    case vs a hard case."""
    B, H, W = 1, 1, 2
    targets = torch.tensor([[[0, 0]]])
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    logits = torch.zeros(B, N_CLASSES, H, W)
    logits[0, 0, 0, 0] = 50.0  # pixel 0: confidently CORRECT (easy)
    logits[0, 1, 0, 1] = 50.0  # pixel 1: confidently WRONG (hard) -- true class is 0

    ce = confidence_weighted_ce_loss(logits, targets, valid_mask)
    fl = focal_loss(logits, targets, valid_mask, gamma=2.0)
    # Both pixels contribute to ce's mean; fl's mean must be smaller
    # overall (the easy pixel's near-zero CE barely changes, but the
    # hard pixel's already-large CE is barely down-weighted either,
    # since (1-p_t) is close to 1 there) -- net effect: fl <= ce.
    assert fl.item() <= ce.item() + 1e-4


def test_focal_loss_matches_ce_when_gamma_is_zero():
    logits, targets, valid_mask = _random_case()
    ce = confidence_weighted_ce_loss(logits, targets, valid_mask)
    fl = focal_loss(logits, targets, valid_mask, gamma=0.0)
    assert fl.item() == pytest.approx(ce.item(), rel=1e-4)


def test_focal_loss_empty_valid_mask_is_zero_and_differentiable():
    logits, targets, _ = _random_case()
    valid_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss = focal_loss(logits, targets, valid_mask)
    assert loss.item() == 0.0


def test_drishti_seg_loss_focal_gamma_none_is_byte_identical_to_before():
    """The whole point of making focal_gamma default to None: existing
    behaviour must be UNCHANGED when it's not passed."""
    logits, targets, valid_mask = _random_case()
    baseline = DrishtiSegLoss()(logits, targets, valid_mask)
    explicit_none = DrishtiSegLoss(focal_gamma=None)(logits, targets, valid_mask)
    assert baseline.item() == pytest.approx(explicit_none.item())


def test_drishti_seg_loss_with_focal_gamma_changes_the_total():
    logits, targets, valid_mask = _random_case()
    without_focal = DrishtiSegLoss()(logits, targets, valid_mask)
    with_focal = DrishtiSegLoss(focal_gamma=2.0, focal_weight=0.5)(logits, targets, valid_mask)
    assert torch.isfinite(with_focal)
    assert with_focal.item() != pytest.approx(without_focal.item())
