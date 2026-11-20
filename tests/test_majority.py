from itertools import product

import pytest

torch = pytest.importorskip("torch")

from fusionbench.methods.majority import independent_majority_logit, median_logit


def test_independent_majority_matches_conditional_probability():
    logits = torch.tensor(
        [[[0.1, -0.9, 2.1]], [[-3.4, 0.4, -4.0]]], dtype=torch.float64
    )
    p, q, r = logits.sigmoid().unbind(-1)
    expected = p * q + q * r + p * r - 2 * p * q * r
    torch.testing.assert_close(independent_majority_logit(logits).sigmoid(), expected)


def test_sign_corners_and_opposing_extreme_votes():
    corners = torch.tensor(list(product([-10000.0, 10000.0], repeat=3)))
    for fn in (independent_majority_logit, median_logit):
        logits = fn(corners)
        assert logits.isfinite().all()
        torch.testing.assert_close(logits > 0, (corners > 0).sum(-1) >= 2)
    query = torch.tensor([[10000.0, -10000.0, 0.000123]], dtype=torch.float32)
    torch.testing.assert_close(
        independent_majority_logit(query), query[:, 2], atol=1e-10, rtol=1e-6
    )


def test_gradient_reaches_three_latent_logits_or_only_the_median():
    x = torch.tensor([-0.43, 0.0, 0.71], dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(independent_majority_logit(x), x)
    p, q, r = x.sigmoid()
    group = p * q + p * r + q * r - 2 * p * q * r
    slope = torch.stack((q + r - 2 * q * r, p + r - 2 * p * r, p + q - 2 * p * q))
    expected = slope * x.sigmoid() * (1 - x.sigmoid()) / (group * (1 - group))
    torch.testing.assert_close(gradient, expected)
    assert (gradient > 0).all()
    (median_gradient,) = torch.autograd.grad(median_logit(x), x)
    torch.testing.assert_close(
        median_gradient, torch.tensor([0.0, 1.0, 0.0], dtype=x.dtype)
    )


@pytest.mark.parametrize("fn", [independent_majority_logit, median_logit])
def test_logit_shape_and_dtype(fn):
    for logits in (
        torch.tensor(0.0),
        torch.zeros(1, 4),
        torch.zeros(2, 3, dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match="floating-point logits"):
            fn(logits)
