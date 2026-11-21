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


def test_matched_sampling_agrees_with_enumerating_all_subsets():
    from itertools import combinations, product
    from fusionbench.eval.search_coverage import arities, reference

    e = np.array([3, 1, 0, 5, 4, 1])
    classes = np.array([1, 1, 1, 2, 2, 2])
    minima = [
        e[list(a + b)].min()
        for a, b in product(combinations(range(3), 2), combinations(range(3, 6), 1))
    ]
    hit, gap = reference(e, classes, [0, 2, 1])
    assert hit == pytest.approx(np.mean(np.asarray(minima) == 0))
    assert gap == pytest.approx(np.mean(minima))
    np.testing.assert_array_equal(
        arities(
            np.stack(
                [CUBE[:, 0], CUBE[:, 0] & CUBE[:, 1], conditional("two_pathways") > 0.5]
            )
        ),
        [1, 2, 4],
    )


def test_starter_intervention_reaches_prespecified_arities(tmp_path):
    from fusionbench.eval.search_coverage import arities
    from fusionbench.eval.search_initialization import starters
    from okapi.globals import set_backend

    set_backend("numpy")
    paths = [tmp_path / f"provider{i}.npy" for i in range(4)]
    for mode, expected in [("one_per_arity", [1, 2, 3, 4]), ("balanced", [4] * 4)]:
        trees = starters(mode, paths, seed=19)
        decisions = np.stack([tree.predict().reshape(-1) > 0.5 for tree in trees])
        assert sorted(arities(decisions).tolist()) == expected
        if mode == "balanced":
            assert [tree.nodes_count for tree in trees] == [7] * 4


def test_graded_exact_risk_and_fit_counts_match_raw_observations():
    from fusionbench.eval.search_graded import (
        IID,
        SCORES,
        brier,
        conditional,
        counts,
        lattice,
        risk,
    )

    rules, _ = lattice()
    assert rules.shape == (166, 256)
    task = conditional("matched", [0, 1, 3, 2])
    np.testing.assert_allclose(risk(rules, task, IID).min(), risk(task, task, IID))
    cells = np.array([2, 2, 18, 82, 150, 233, 251])
    y = np.array([1, 0, 1, 1, 0, 1, 0], dtype=bool)
    brute = ((rules[:, cells] - y) ** 2).mean(1)
    np.testing.assert_allclose(brier(rules, counts(cells, y)), brute)
    assert np.allclose(
        brier(rules, counts(cells, y)) * len(y) * 400, np.round(brute * len(y) * 400)
    )
    assert risk(SCORES.mean(1), conditional("mean", range(4)), IID) == pytest.approx(
        risk(conditional("mean", range(4)), conditional("mean", range(4)), IID)
    )
