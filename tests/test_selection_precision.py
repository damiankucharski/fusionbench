"""Regression tests for score quantization, native archive paths and passive survivor audits."""

import inspect
import json
from itertools import combinations

import numpy as np
import pytest

from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree


@pytest.fixture
def precision_sdk():
    sdk = pytest.importorskip("okapi")
    if "selection_fitness_decimals" not in inspect.signature(sdk.Okapi).parameters:
        pytest.skip("requires OKAPI's native selection-precision option")
    return sdk


def inputs(classes):
    rng = np.random.default_rng([981, classes])
    fit = rng.dirichlet(np.ones(classes), size=(33, 7))
    validation = rng.dirichlet(np.ones(classes), size=(15, 7))
    return (
        fit,
        rng.permutation(np.arange(len(fit)) % classes),
        validation,
        rng.permutation(np.arange(len(validation)) % (classes - 1)),
    )


def archive(checkpoint):
    return (
        checkpoint.calls,
        checkpoint.generation,
        checkpoint.n_trees,
        checkpoint.path_digest,
        checkpoint.fit_brier,
        checkpoint.validation_brier,
        checkpoint.fit_tree.structural_key(),
        checkpoint.validation_tree.structural_key(),
    )


@pytest.mark.parametrize(
    ("values", "raw", "three", "six"),
    [
        ([0.9120001, 0.9120001, 0.9120004, 0.9121399, 0.91261, 0.91290], 5, 2, 4),
        # At these boundaries, rounding (1 - fitness) yields a different tie count.
        ([0.5135, np.nextafter(0.5135, 1.0)], 2, 1, 1),
        ([0.5242885, np.nextafter(0.5242885, 1.0)], 2, 1, 2),
    ],
)
def test_resolution_counts_raw_and_rounded_native_fitness(values, raw, three, six):
    from fusionbench.eval.selection_precision import resolution

    scores = np.array(values)
    scores.setflags(write=False)
    assert resolution(scores) == {
        "distinct_raw": raw,
        "distinct_3": three,
        "distinct_6": six,
        "distinct_compressed_3": 1,
    }


@pytest.mark.parametrize("classes", [2, 4])
def test_three_decimals_matches_stock_at_initial_and_later_checkpoints(
    classes, monkeypatch, precision_sdk
):
    Okapi = precision_sdk.Okapi

    from fusionbench.eval.selection_precision import fit_selection_precision

    fit, fit_y, validation, validation_y = inputs(classes)
    budgets = (1, 19, 37, 38, 115, 279)
    seed = 759
    complete = []
    native = Okapi._calculate_fitnesses

    def record_cohort(engine, trees=None):
        result = native(engine, trees)
        end = len(result) + (complete[-1]["call_end"] if complete else 0)
        raw = result[:, 0]
        complete.append(
            {
                "call_end": end,
                "n": len(raw),
                "kind": "survival" if complete else "initial",
                "distinct_raw": len(set(raw)),
                "distinct_3": len(set(raw.round(3))),
                "distinct_6": len(set(raw.round(6))),
                "distinct_compressed_3": len(
                    set(np.round(1 - 2.0**-12 * (1 - raw), 3))
                ),
            }
        )
        for key, values in (("raw", raw), ("3", raw.round(3)), ("6", raw.round(6))):
            complete[-1]["tied_pairs_" + key] = sum(
                a == b for a, b in combinations(values, 2)
            )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Okapi, "_calculate_fitnesses", record_cohort)
        # Extending the reference beyond the last tested checkpoint gives
        # the actual batch end for a checkpoint inside its final generation.
        reference = fit_suite_point(
            fit,
            fit_y,
            validation,
            validation_y,
            seed=seed,
            point_mutation=False,
            budgets=(*budgets, 400),
        )
    actual = fit_selection_precision(
        fit, fit_y, validation, validation_y, seed=seed, decimals=3, budgets=budgets
    )
    assert set(actual) == set(budgets)
    assert reference[1].generation == reference[37].generation == 1
    assert reference[279].generation > 1
    # At the 37th fitness call the initial 7 providers and 30 injected shallow
    # programs are complete. The next native survival batch has just begun at 38.
    assert [len(actual[b].cohorts) for b in (19, 37, 38)] == [0, 1, 1]
    assert actual[37].cohorts[0]["kind"] == "initial"
    assert actual[37].cohorts[0]["call_end"] == 37
    assert any(c["kind"] == "survival" for c in actual[279].cohorts)

    for budget in budgets:
        cp = actual[budget]
        assert archive(cp) == archive(reference[budget])
        assert cp.counts["fit_calls"] == budget
        assert cp.counts["validation_calls"] == cp.n_trees
        assert cp.counts["complete_batches"] == len(cp.cohorts)
        assert cp.counts["complete_scored_rows"] == sum(c["n"] for c in cp.cohorts)
        assert cp.counts["complete_scored_rows"] <= budget
        expected = [c for c in complete if c["call_end"] <= budget]
        assert [c["call_end"] for c in cp.cohorts] == [c["call_end"] for c in expected]
        for row, native_row in zip(cp.cohorts, expected, strict=True):
            assert {column: row[column] for column in native_row} == native_row
        for field in ("fit_tree", "validation_tree"):
            np.testing.assert_array_equal(
                predict_tree(getattr(cp, field), validation),
                predict_tree(getattr(reference[budget], field), validation),
            )


@pytest.mark.parametrize(
    ("decimals", "policy"), [(3, "round3"), (6, "round6"), (None, "raw")]
)
@pytest.mark.usefixtures("precision_sdk")
def test_survival_audit_preserves_both_future_trajectory_and_archive(decimals, policy):
    from fusionbench.eval.selection_precision import fit_selection_precision

    observations = []
    for active in (False, True):
        observations.append(
            fit_selection_precision(
                *inputs(2),
                seed=759,
                decimals=decimals,
                budgets=(38, 139, 279),
                audit_survival=active,
            )
        )
    plain, audited = observations
    assert plain[279].generation >= 3
    for budget in plain:
        assert archive(audited[budget]) == archive(plain[budget])
        assert audited[budget].counts == plain[budget].counts
    for row in audited[279].cohorts:
        if row["kind"] == "initial":
            assert "native_indices" not in row
            continue
        selected = json.loads(row["native_indices"])
        assert len(selected) == len(set(selected)) == row["survivor_capacity"]
        assert selected == json.loads(row[policy + "_indices"])
        for name in ("round3", "round6", "raw"):
            indices = json.loads(row[name + "_indices"])
            assert len(indices) == len(set(indices)) == len(selected)
            assert all(0 <= i < row["n"] for i in indices)
            assert (
                len(set(selected) - set(indices))
                == row[name + "_membership_replacements"]
            )
            assert (
                sum(i != j for i, j in zip(indices, selected, strict=True))
                == row[name + "_positions_changed"]
            )
