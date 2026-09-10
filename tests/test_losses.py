"""
tests/test_losses.py

Ticket #29 tests: the garbage-invariance masking test (Ticket #29's own
required assertion), Lovász near-zero on a perfect prediction, and the
combined stack's aux-resolution path.
"""

from __future__ import annotations

import torch

from perception.losses import (
    DrishtiSegLoss,
    confidence_weighted_ce_loss,
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
