import numpy as np
import pytest

pytest.importorskip("okapi")

from fusionbench.eval.search_regret import (
    CUBE,
    IID,
    POWERS,
    SHIFT,
    Counts,
    FitnessArchive,
    Problem,
    conditional,
    family,
    population_risk,
)


def test_family_is_complete_and_population_mismatch_is_exact():
    tables, codes = family()
    assert len(set(codes)) == 166
    # Independent exhaustive characterization, including the two endpoint constraints.
    all_bits = (np.arange(1 << 16)[:, None] >> np.arange(16)) & 1
    keep = (all_bits[:, 0] == 0) & (all_bits[:, -1] == 1)
    for lower in range(16):
        for axis in range(4):
            upper = lower | (1 << axis)
            keep &= all_bits[:, lower] <= all_bits[:, upper]
    assert set(all_bits[keep] @ POWERS) == set(codes)
    np.testing.assert_allclose(
        population_risk(tables, conditional("two_pathways"), IID).min(), 0.18
    )
    np.testing.assert_allclose(
        population_risk(tables, conditional("parity"), IID).min(), 0.34
    )
    assert SHIFT.sum() == pytest.approx(1.0)


def test_fitness_uses_only_fit_rows_even_with_unobserved_cells():
    ix = np.array([1, 1, 1, 3, 8])
    y = np.array([1, 0, 0, 1, 1], dtype=bool)
    fit = Counts.from_rows(ix, y)
    objective = FitnessArchive(fit)
    tables, _ = family()
    for table in tables:
        got = objective(table[:, None], None)
        assert got == pytest.approx(np.mean(table[ix] == y))
    assert objective.calls == len(objective.errors) == 166


def test_erm_ties_and_signed_risk_decomposition():
    p = Problem("parity", seed=12, n_fit=32, max_fit=128, n_val=70, n_test=100)
    tables, codes = family()
    metric = p.metrics(int(CUBE[:, 0] @ POWERS))
    loss = metric["iid_risk"]
    ref = p.metrics(p.erm)
    assert loss == pytest.approx(
        p.family_oracles["iid"]
        + ref["delta_family_iid"]
        + metric["iid_minus_exact_erm"]
    )
    assert ref["delta_family_fit_errors"] == 0
    assert p.tied == np.sum(p.fit.errors(tables) == p.best_errors)
