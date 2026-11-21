from fractions import Fraction
from itertools import combinations
from math import prod

import numpy as np
import pytest

from fusionbench.eval.matching_posterior import (
    MatchingFamily,
    four_active_counterexample,
)


def queries(n, sizes):
    z = []
    for s in sizes:
        for active in combinations(range(n), s):
            row = [0] * n
            for i in active:
                row[i] = 1
            z.append(row)
    return np.asarray(z, dtype=np.int8).reshape(-1, n)


def literal_truth(z, edges, models):
    """Construct each included model separately, applying OR of its ANDs directly."""
    out = np.empty((len(z), len(models)), dtype=bool)
    for j, model in enumerate(models):
        terms = [(z[:, a] != 0) & (z[:, b] != 0) for a, b in (edges[e] for e in model)]
        out[:, j] = np.logical_or.reduce(terms)
    return out


@pytest.mark.parametrize("k,size", [(2, 3), (4, 105), (6, 10395)])
def test_complete_lexicographic_family(k, size):
    fam = MatchingFamily(k)
    rows = [tuple(map(int, row)) for row in fam.matchings]
    assert len(rows) == size
    assert rows == sorted(set(rows))
    assert np.all(fam.incidence.sum(1) == k)
    for model in rows:
        assert sorted(int(i) for e in model for i in fam.edges[e]) == list(range(2 * k))
    if k == 4:
        brute = [
            chosen
            for chosen in combinations(range(len(fam.edges)), k)
            if len({int(i) for e in chosen for i in fam.edges[e]}) == 2 * k
        ]
        assert rows == brute


@pytest.mark.parametrize("mode", ["conditional", "x_only", "joint"])
def test_posterior_against_fractional_row_by_row_likelihood(mode):
    fam = MatchingFamily(4)
    pairs = queries(8, [2])
    z = pairs[[2, 4, 4, 10, 8, 11]]
    y = np.array([1, 0, 0, 1, 0, 1])
    u = pairs[[0, 0, 1, 1, 1, 7]]
    p0, p1, pi = Fraction(1, 8), Fraction(7, 8), Fraction(1, 2)
    post = fam.fit(
        z,
        None if mode == "x_only" else y,
        u,
        pi=float(pi),
        low=float(p0),
        high=float(p1),
        mode=mode,
    )
    active = literal_truth(np.vstack((z, u)), fam.edges, fam.matchings)
    expected = []
    errors = []
    for hits in active.T:
        likelihood = Fraction(1)
        if mode != "conditional":
            likelihood *= prod(
                pi / fam.k if hit else (1 - pi) / (len(fam.edges) - fam.k)
                for hit in hits
            )
        if mode != "x_only":
            for hit, label in zip(hits[: len(y)], y, strict=True):
                prob = p1 if hit else p0
                likelihood *= prob if label else 1 - prob
        expected.append(likelihood)
        errors.append(
            sum(
                (int(label) - (p1 if hit else p0)) ** 2
                for label, hit in zip(y, hits[: len(y)], strict=True)
            )
        )
    total = sum(expected)
    np.testing.assert_allclose(
        post.weights, [float(p / total) for p in expected], rtol=1e-13, atol=2e-16
    )
    assert post.map_index == expected.index(max(expected))
    assert fam.fit_erm(z, y, low=float(p0), high=float(p1)) == errors.index(min(errors))
    if mode != "x_only":
        assert post.fit_tie_count == errors.count(min(errors))
    if mode == "conditional":
        assert post.erm_index == post.map_index


def test_uniform_pair_design_is_exactly_uninformative_and_unlabeled_labels_are_unused():
    fam = MatchingFamily(4)
    z = queries(8, [2])
    y = np.arange(len(z)) % 4 == 0
    u = np.repeat(z[[0]], 90, axis=0)
    cond = fam.fit(z, y, u, pi=1 / 7, mode="conditional")
    joint = fam.fit(z, y, u, pi=1 / 7, mode="joint")
    np.testing.assert_array_equal(joint.weights, cond.weights)
    uniform = fam.fit(z, None, u, pi=1 / 7, mode="x_only")
    np.testing.assert_array_equal(
        uniform.weights, np.full_like(uniform.weights, uniform.weights[0])
    )
    assert uniform.fit_tie_count is None
    informative = fam.fit(z, None, u, pi=0.5, mode="x_only")
    assert informative.edge_weights[0] > 0.99
    np.testing.assert_array_equal(
        informative.weights, fam.fit(z, ~y, u, pi=0.5, mode="x_only").weights
    )


@pytest.mark.parametrize("k", [4, 6])
def test_compilation_rowwise_on_all_subsets_through_five_actives(k):
    fam = MatchingFamily(k)
    rng = np.random.default_rng(879 + k)
    fit = queries(fam.n, [2])
    selected = rng.integers(len(fit), size=22)
    y = rng.integers(2, size=len(selected))
    post = fam.fit(fit[selected], y, fit[::2], pi=0.5)
    z = queries(fam.n, [0, 1, 2, 3, 4, 5])
    truth = literal_truth(z, fam.edges, fam.matchings)
    np.testing.assert_array_equal(fam.contains(z, [1, 2]), truth[:, [1, 2]])
    np.testing.assert_array_equal(fam.contains(z, 2), truth[:, 2])
    explicit = 1 / 64 + (62 / 64) * (truth @ post.weights)
    np.testing.assert_allclose(post.explicit(z), explicit, atol=1e-13, rtol=0)
    first, second = post.compile(1), post.compile(2)
    assert first.n_units == len(fam.edges)
    assert second.n_units == len(fam.edges) + len(fam.quadruples)
    assert not hasattr(second, "weights") and not hasattr(second, "family")
    eligible = z.sum(1) <= 3
    np.testing.assert_allclose(
        first.predict(z[eligible]), explicit[eligible], atol=5e-13, rtol=0
    )
    np.testing.assert_allclose(second.predict(z), explicit, atol=5e-13, rtol=0)
    np.testing.assert_allclose(
        second.predict(1 / 64 + z * 62 / 64), explicit, atol=5e-13, rtol=0
    )
    for j in [0, len(fam.quadruples) - 1]:
        quad = set(fam.quadruples[j])
        weight = sum(
            w
            for w, model in zip(post.weights, fam.matchings, strict=True)
            if sum(set(fam.edges[e]).issubset(quad) for e in model) == 2
        )
        assert abs(weight - post.quad_weights[j]) < 5e-13


def test_eight_expert_uniform_ambiguity_and_even_the_clipped_first_order_failure():
    fam = MatchingFamily(4)
    z = queries(8, [4])
    prior = fam.fit(z[:0], [], None, pi=0.5, low=0, high=1, mode="conditional")
    counts = literal_truth(z, fam.edges, fam.matchings)
    assert np.all(counts.sum(1) == 81)
    ref = four_active_counterexample()
    assert ref["exact_event"] == Fraction(81, 105) == Fraction(27, 35)
    assert ref["first_event"] == Fraction(6, 7)
    assert ref["event_gap"] == Fraction(3, 35)
    np.testing.assert_allclose(
        prior.explicit(z), float(ref["exact_event"]), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        prior.predict(z, 2), prior.explicit(z), atol=1e-14, rtol=0
    )
    for approx in [prior.predict(z), np.clip(prior.predict(z), 0, 1)]:
        np.testing.assert_allclose(
            approx - prior.explicit(z), float(ref["event_gap"]), rtol=0, atol=1e-14
        )
    assert four_active_counterexample(Fraction(1, 64), Fraction(63, 64))[
        "probability_gap"
    ] == Fraction(62, 64) * Fraction(3, 35)


def test_schema_and_impossible_data():
    fam = MatchingFamily(2)
    z = queries(4, [2])
    with pytest.raises(ValueError, match="zero probability"):
        fam.fit(z[[0, 0]], [0, 1], pi=0.5, low=0, high=1, mode="conditional")
    with pytest.raises(ValueError, match="two active"):
        fam.fit(queries(4, [3]), np.zeros(4), pi=0.5)
    with pytest.raises(ValueError, match="zero probability"):
        fam.fit(np.vstack([z[0], z[1]]), None, pi=1.0, mode="x_only")
    for k in (1, 7, False):
        with pytest.raises(ValueError):
            MatchingFamily(k)
    deterministic = fam.fit(z[:1], [1], pi=0.5, low=0, high=1, mode="conditional")
    assert deterministic.weights.sum() == 1.0
    assert deterministic.weights[deterministic.map_index] == 1.0


def test_second_order_limit_when_three_matching_edges_can_be_active():
    fam = MatchingFamily(3)
    empty = np.empty((0, 6), dtype=np.uint8)
    post = fam.fit(empty, [], pi=0.5, low=0, high=1, mode="conditional")
    z = np.ones((1, 6), dtype=np.uint8)
    # Every possible perfect matching has exactly three pairs within this set.
    np.testing.assert_allclose(post.explicit(z), [1.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(post.predict(z, 1), [3.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(post.predict(z, 2), [0.0], rtol=0, atol=1e-14)


def test_asymmetric_emission_rates_can_separate_brier_erm_and_label_map():
    fam = MatchingFamily(2)
    z = np.tile([[1, 1, 0, 0]], (10, 1))
    y = np.array([1] * 7 + [0] * 3)
    post = fam.fit(z, y, pi=1 / 3, low=0.5, high=0.875, mode="conditional")
    assert fam.fit_erm(z, y, low=0.5, high=0.875) == post.erm_index == 0
    assert post.map_index == 1 and post.fit_tie_count == 1


@pytest.mark.parametrize("order", [1, 2])
def test_exported_torch_model_is_frozen_and_consumes_raw_endpoints(order):
    torch = pytest.importorskip("torch")
    fam = MatchingFamily(4)
    z = queries(8, (2, 3, 4))
    fit = z[:28]
    post = fam.fit(fit, np.arange(len(fit)) % 2, fit[:8], pi=0.5)
    deployed = post.compile(order)
    state = torch.random.get_rng_state().clone()
    model = deployed.to_torch()
    assert torch.equal(state, torch.random.get_rng_state())
    assert [type(layer) for layer in model] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]
    assert all(
        not p.requires_grad and p.dtype == torch.float64 for p in model.parameters()
    )
    scores = np.asarray(1 / 64 + z * 62 / 64, dtype=np.float64)
    raw = torch.from_numpy(scores)
    with torch.no_grad():
        traced = torch.jit.trace(model, raw[:1])
        np.testing.assert_allclose(
            traced(raw).numpy()[:, 0], deployed.predict(z), rtol=0, atol=1e-12
        )
    expected = deployed.n_units * (fam.n + 2) + 1
    assert sum(p.numel() for p in model.parameters()) == expected
