from itertools import product

import numpy as np
import pytest

from fusionbench.eval.paired_brier import (
    deterministic_gap_requests,
    draw_pair_audit,
    sharp_pair_bounds,
)


def test_finite_ambiguous_completions_and_deterministic_radius():
    f = np.array([0.05, 0.4, 0.6, 0.25])
    g = np.array([0.9, 0.4, 0.7, 0.1])
    labels = [0, 0, 1, 1]
    for budget in range(5):
        seen = deterministic_gap_requests(f, g, budget)
        revealed = np.asarray(labels)[seen]
        lo, hi = sharp_pair_bounds(f, g, seen, revealed)
        observed = []
        for y in map(np.array, product((0, 1), repeat=4)):
            if np.array_equal(y[seen], revealed):
                observed.append(np.mean((f - y) ** 2 - (g - y) ** 2))
        np.testing.assert_allclose([lo, hi], [min(observed), max(observed)])
        for mask in product((0, 1), repeat=4):
            if sum(mask) == len(seen):
                other = np.flatnonzero(mask)
                l, h = sharp_pair_bounds(f, g, other, np.asarray(labels)[other])
                assert h - l >= hi - lo - 1e-14


def test_sampler_uses_only_requested_labels_and_repetitions_remain():
    f, g = np.array([0.1, 0.5, 0.75]), np.array([0.9, 0.5, 0.55])
    y = np.array([0, 1, 1])
    audit = draw_pair_audit(f, g, 25000, rng=np.random.default_rng(570), proposal="gap")
    assert set(audit.indices) == {0, 2}
    d = np.mean((f - y) ** 2 - (g - y) ** 2)
    assert abs(audit.estimate(y[audit.indices]) - d) < 0.008
    lo, hi = audit.exact_interval(y[audit.indices], confidence=0.999999)
    assert lo - 1e-15 <= d <= hi + 1e-15
    with pytest.raises(ValueError):
        audit.estimate(y)
    uniform = draw_pair_audit(
        f, g, 25000, rng=np.random.default_rng(570), proposal="uniform"
    )
    assert abs(uniform.estimate(y[uniform.indices]) - d) < 0.008
    with pytest.raises(ValueError):
        uniform.exact_interval(y[uniform.indices])


def test_identical_predictions_require_no_paid_labels():
    a = draw_pair_audit([0.2, 0.9], [0.2, 0.9], 3, rng=np.random.default_rng(2))
    assert len(a.indices) == 0
    assert a.estimate([]) == 0 and a.exact_interval([]) == (0, 0)
