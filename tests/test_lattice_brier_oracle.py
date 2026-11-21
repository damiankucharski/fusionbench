from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("networkx")

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier


def exact_loss(p, y):
    return sum(
        (Fraction(float(a)) - int(b)) ** 2 for a, b in zip(p, y, strict=True)
    ) / len(y)


@pytest.fixture
def reference():
    return pytest.importorskip("okapi.lattice_search")


def test_against_exhaustive_exact_rational_loss(reference):
    for seed in range(36):
        rng = np.random.default_rng(seed)
        d = seed % 4 + 1
        x = rng.random((13, d))
        if seed % 2:
            x = np.round(x, 1)
        x[-1] = x[0]
        y = rng.integers(0, 2, len(x))
        y[-1] = 1 - y[0]
        result = fit_lattice_brier(x, y)
        losses = [
            exact_loss(reference.SmallLattice(terms, d, 0).predict(x), y)
            for terms in reference.small_lattice_programs(d)
        ]
        assert result.minimum_loss == min(losses)
        for pred in result.bounds(x):
            assert exact_loss(pred, y) == result.minimum_loss
        np.testing.assert_array_equal(
            result.predict(x), reference.SmallLattice(result.terms(), d, 0).predict(x)
        )


def test_all_tied_and_nearly_tied_probabilities(reference):
    tiny = np.nextafter(0.0, 1.0)
    x = np.array([[0.5, np.nextafter(0.5, 1.0)], [tiny, 0.0], [1.0, 1.0]])
    y = np.array([1, 0, 1])
    result = fit_lattice_brier(x, y)
    assert result.minimum_loss == min(
        exact_loss(reference.SmallLattice(t, 2, 0).predict(x), y)
        for t in reference.small_lattice_programs(2)
    )
    tied = fit_lattice_brier(np.full((3, 2), 0.5), [1, 0, 0])
    assert tied.minimum_loss == Fraction(1, 4)
    lo, hi = tied.bounds(np.array([[0, 1], [1, 0]]))
    np.testing.assert_array_equal(lo, [0, 0])
    np.testing.assert_array_equal(hi, [1, 1])


def test_registered_comparator_predicts_binary_probabilities():
    from fusionbench.methods.registry import default_methods

    rng = np.random.default_rng(42)
    positive = rng.uniform(size=(32, 8))
    providers = np.stack([1 - positive, positive], axis=-1)
    y = rng.integers(0, 2, len(providers))
    [model] = default_methods(["lattice_brier_exact"])
    model.fit(providers, y, task="binary", n_classes=2)
    scores = model.predict(providers)
    np.testing.assert_allclose(scores.sum(1), 1)
    assert exact_loss(scores[:, 1], y) == model.rule.minimum_loss
    assert "lattice_brier_exact" not in [m.name for m in default_methods()]
