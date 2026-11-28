import numpy as np
import pytest

from fusionbench.methods.provider_reliance import (
    limit_reliance,
    marginal_prediction,
    source_donors,
)


def test_multiclass_donors_keep_whole_reports_and_tied_mass():
    reports = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.75, 0.125, 0.125],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ]
    )
    x = np.stack((np.roll(reports, 1, axis=1), reports), axis=1)
    expected = np.array(sorted(reports.tolist()))
    np.testing.assert_array_equal(source_donors(x, 1, count=4), expected)
    np.testing.assert_array_equal(source_donors(x, 1, count=2), expected[[1, 3]])
    np.testing.assert_array_equal(source_donors(x[::-1], 1, count=2), expected[[1, 3]])
    np.testing.assert_array_equal(
        source_donors(x, 1, count=8), np.repeat(expected, 2, axis=0)
    )


def test_marginalization_of_a_gate_keeps_interactions_and_query_rows_intact():
    # Gate switches between experts. Both experts still vary row by row.
    x = np.array([[0.9, 0.8, 0.2], [0.5, 0.7, 0.5], [0.1, 0.2, 0.9]])
    d = np.array([[0.9, 0.1], [0.5, 0.5], [0.8, 0.2]])
    orig = x.copy()
    seen = []

    def gate(q):
        seen.append(q.copy())
        prob = (1 - q[:, 0, 1]) * q[:, 1, 1] + q[:, 0, 1] * q[:, 2, 1]
        return np.column_stack((1 - prob, prob))

    out = marginal_prediction(gate, x, 0, d)
    gate_mean = d[:, 1].mean()
    np.testing.assert_allclose(
        out[:, 1], (1 - gate_mean) * x[:, 1] + gate_mean * x[:, 2]
    )
    assert len(seen) == len(d)
    for query, donor in zip(seen, d, strict=True):
        np.testing.assert_array_equal(query[:, 0, :], np.broadcast_to(donor, (3, 2)))
        np.testing.assert_array_equal(query[:, 1:, 1], orig[:, 1:])
    np.testing.assert_array_equal(orig, x)
    np.testing.assert_allclose(out.sum(axis=1), 1)


def test_nonlinear_model_is_evaluated_separately_for_every_source_report():
    x = np.array([[[0.2, 0.8]], [[0.7, 0.3]]])
    donors = np.array([[0.2, 0.8], [0.8, 0.2]])

    def nonlinear(z):
        p = z[:, 0, 1] ** 2
        # Even a predictor that mutates its input cannot alter the client's bank.
        z[:] = 0
        return np.column_stack([1 - p, p])

    before = x.copy()
    result = marginal_prediction(nonlinear, x, 0, donors)
    np.testing.assert_allclose(result[:, 1], (0.8**2 + 0.2**2) / 2)
    np.testing.assert_array_equal(x, before)


def test_blend_strength_endpoints_and_interior():
    x = np.array([[0.125, 0.5, 0.375], [0.5, 0.5, 0]])
    m = np.array([[0, 0, 1.0], [0.25, 0.25, 0.5]])
    np.testing.assert_array_equal(limit_reliance(x, m, 0), x)
    np.testing.assert_array_equal(limit_reliance(x, m, 1), m)
    blend = limit_reliance(x, m, 0.25)
    np.testing.assert_allclose(blend, 0.75 * x + 0.25 * m)
    np.testing.assert_allclose(blend.sum(axis=1), 1)
    with pytest.raises(ValueError, match="shapes"):
        limit_reliance(x, m[:1], 0.5)


@pytest.mark.parametrize("value", [True, -0.1, float("nan"), 1.1, 0.5 + 1j])
def test_invalid_strength(value):
    with pytest.raises(ValueError, match="strength"):
        limit_reliance([[0.5, 0.5]], [[0.2, 0.8]], value)


@pytest.mark.parametrize(
    "provider,count", [(-1, 1), (2, 1), (False, 1), (0, 0), (0, 1.5)]
)
def test_invalid_source_donor_arguments(provider, count):
    with pytest.raises(ValueError):
        source_donors(np.full((4, 2), 0.5), provider=provider, count=count)


def test_reject_malformed_donor_and_predictor_outputs():
    x = np.full((3, 2), 0.5)
    with pytest.raises(ValueError, match="class axis"):
        marginal_prediction(lambda q: q[:, 0, :], x, 0, [[0.2, 0.4, 0.4]])
    with pytest.raises(ValueError, match="finite"):
        marginal_prediction(lambda q: q[:, 0, :], x, 0, [[np.nan, 0.4]])
    with pytest.raises(ValueError, match="one fused"):
        marginal_prediction(lambda q: q[:1, 0, :], x, 0, [[0.5, 0.5]])
    with pytest.raises(ValueError, match="finite"):
        marginal_prediction(lambda q: 0.1 * q[:, 0, :], x, 0, [[0.5, 0.5]])
