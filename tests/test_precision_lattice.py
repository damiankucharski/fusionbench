import numpy as np

from fusionbench.eval import precision_lattice as experiment
from fusionbench.eval.search_graded import IID, SCORES, SHIFT, counts, risk


def test_lattice_complete_and_matched_and_additive_laws_separate():
    rules, lookup = experiment.exact_family()
    assert rules.shape == (166, 256)
    for j in range(4):
        assert lookup[SCORES[:, j].tobytes()] == j
    corners = np.all((SCORES == 0.1) | (SCORES == 0.9), axis=1)
    assert len(set(map(tuple, rules[:, corners]))) == 166
    for law in ("matched", "additive"):
        bank = experiment.draw_bank(law, 9)
        if law == "matched":
            assert (rules == bank.eta).all(axis=1).sum() == 1
            assert np.isclose(
                risk(rules, bank.eta, IID).min(),
                risk(bank.eta, bank.eta, IID),
                atol=1e-14,
            )
        else:
            assert not np.any((rules == bank.eta).all(axis=1))
            assert risk(rules, bank.eta, SHIFT).min() > risk(bank.eta, bank.eta, SHIFT)


def test_integer_empirical_optima_agree_with_literal_row_axis_and_size_scaling():
    rng = np.random.default_rng(8)
    cells, labels = rng.integers(0, 256, 256), rng.random(256) > 0.55
    rules, _ = experiment.exact_family()
    for n in (32, 256):
        z = counts(cells[:n], labels[:n])
        actual = experiment.integer_losses(rules, z)
        ticks = np.rint(rules[:, cells[:n]] * 20).astype(int)
        np.testing.assert_array_equal(
            actual, np.sum((ticks - labels[:n] * 20) ** 2, axis=1)
        )
        f = experiment.selection_score(actual, n, False)
        assert len(set(np.round(f, 6))) == len(set(actual))
        assert np.unique(
            np.round(experiment.selection_score(actual, n, True), 3)
        ).tolist() == [1.0]


def test_tie_counts_distinguish_equal_fit_vectors_from_numerically_merged_losses():
    diag = experiment.archive_metrics(
        np.arange(5),
        np.array([3, 3, 3, 6, 8]),
        [b"x", b"x", b"y", b"z", b"x2"],
        np.array([0.91, 0.91, 0.91, 0.91, 0.87]),
    )
    assert diag["pairs_same_fit_vector"] == 1
    assert diag["pairs_equal_loss_distinct_fit_vectors"] == 2
    assert diag["pairs_merged_only_by_grid"] == 3


def test_native_path_is_replayable_and_exact_archive_is_equal_across_equivalent_units(
    tmp_path,
):
    import inspect

    import pytest

    core = pytest.importorskip("okapi")
    if "selection_fitness_decimals" not in inspect.signature(core.Okapi).parameters:
        pytest.skip("This native regression needs configurable selection precision")
    bank = experiment.draw_bank("matched", 1)
    paths = [tmp_path / f"p{k}.npy" for k in range(4)]
    for j, path in enumerate(paths):
        np.save(path, SCORES[:, j : j + 1])
    label_file = tmp_path / "labels.npy"
    np.save(label_file, bank.eta[:, None])
    fit = counts(bank.fit_cells, bank.fit_labels)
    units = experiment.integer_losses(experiment.exact_family()[0], fit)

    def visit(decimals, compressed):
        snaps, grid = experiment.native_search(
            paths, label_file, units, 256, 43, decimals, compressed, (4, 19, 48)
        )
        assert len(snaps[4][0]) == 4 and np.allclose(
            grid, experiment.selection_score(units, 256, compressed), atol=0.0005
        )
        return [s[1] for s in snaps.values()]

    full = visit(None, False)
    assert visit(6, False) == full
    assert visit(None, True) == full
