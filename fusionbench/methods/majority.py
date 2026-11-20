"""Differentiable aggregations for a binary majority of three latent concepts."""

import torch
from torch import Tensor


def _validate_logits(logits: Tensor) -> None:
    if logits.ndim == 0 or logits.shape[-1] != 3 or not logits.is_floating_point():
        raise ValueError("expected floating-point logits with shape (..., 3)")


def median_logit(logits: Tensor) -> Tensor:
    """Return the median logit. At distinct logits only the middle input receives gradient."""
    _validate_logits(logits)
    return logits.median(dim=-1).values


def independent_majority_logit(logits: Tensor) -> Tensor:
    """Log odds that at least two of three conditionally independent binary concepts hold.

    The final dimension contains the three concept logits. Each of the eight
    binary assignments has an unnormalized log weight equal to the sum of its
    active logits. Subtracting the same per-concept offsets from every assignment
    keeps the weights nonpositive and preserves small effects from a third vote
    when two extreme, opposite votes cancel.
    """
    _validate_logits(logits)
    offset = logits.clamp_min(0)
    a, b, c = (logits - offset).unbind(-1)
    off_a, off_b, off_c = (-offset).unbind(-1)

    majority = torch.stack(
        (a + b + off_c, a + off_b + c, off_a + b + c, a + b + c), dim=-1
    )
    minority = torch.stack(
        (
            off_a + off_b + off_c,
            a + off_b + off_c,
            off_a + b + off_c,
            off_a + off_b + c,
        ),
        dim=-1,
    )
    return majority.logsumexp(-1) - minority.logsumexp(-1)
