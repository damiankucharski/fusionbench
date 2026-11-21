"""Independently replay the marked-gate ablations from the frozen serialized programs."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def edit_program(node, paths, edit, path=()):
    """Use original ordered indices even after preceding siblings disappear."""
    result = dict(node)
    result["children"] = []
    for i, child in enumerate(node["children"]):
        child_path = (*path, i)
        selected = child_path in paths
        if selected and edit == "bypassed":
            continue
        changed = edit_program(child, paths, edit, child_path)
        if selected and edit == "cutoff_half":
            changed["threshold"] = 0.5
        result["children"].append(changed)
    return result


def check_paths(program, paths, subtree):
    assert len(paths) == len(set(paths))
    for path in paths:
        assert path and subtree(program, path)["type"] in (
            "CloseThresholdNode",
            "FarThresholdNode",
        )


def check_sibling_and_nested(evaluate):
    def leaf(i):
        return {"type": "ValueNode", "id": f"p{i}.npy", "children": []}

    def gate(close, threshold, child):
        return {
            "type": "CloseThresholdNode" if close else "FarThresholdNode",
            "close": close,
            "threshold": threshold,
            "children": [child],
        }

    a, b, c, d, e = [leaf(i) for i in range(5)]
    b["children"] = [gate(True, 0.25, c)]
    a["children"] = [
        gate(False, 0.7, b),
        gate(True, 0.75, d),
        {"type": "MeanNode", "children": [e]},
    ]
    x = np.random.default_rng(7310).integers(1, 64, (40, 5)) / 64

    def near(p, q, cut):
        return np.where(abs(p - cut) <= abs(q - cut), p, q)

    def far(p, q, cut):
        return np.where(abs(p - cut) >= abs(q - cut), p, q)

    av, bv, cv, dv, ev = x.T
    three = {(0,), (0, 0, 0), (1,)}
    np.testing.assert_array_equal(
        evaluate(edit_program(a, three, "bypassed"), x), (av + ev) / 2
    )
    np.testing.assert_array_equal(
        evaluate(edit_program(a, three, "cutoff_half"), x),
        (near(far(av, near(bv, cv, 0.5), 0.5), dv, 0.5) + ev) / 2,
    )
    np.testing.assert_array_equal(
        evaluate(edit_program(a, {(0, 0, 0)}, "bypassed"), x),
        (near(far(av, bv, 0.7), dv, 0.75) + ev) / 2,
    )


def verify_intervals(measured, summaries):
    groups = ["fit_regime", "test_regime", "decision", "provenance"]
    for row in summaries.itertuples():
        cell = measured
        for col in groups:
            cell = cell[cell[col] == getattr(row, col)]
        control = cell[cell.method == "intact"].set_index(["data_seed", "gp_seed"])
        treatment = cell[cell.method == row.edit].set_index(["data_seed", "gp_seed"])
        errors = treatment[row.metric] - control[row.metric]
        if row.subset == "marked":
            errors = errors[control.occurrences > 0]
        across_banks = errors.groupby("data_seed").mean()
        assert len(across_banks) == row.banks and len(errors) == row.starts
        n = len(across_banks)
        center = across_banks.mean()
        half = (
            t.ppf(0.975, n - 1) * across_banks.std() / np.sqrt(n) if n > 1 else np.nan
        )
        np.testing.assert_allclose(
            [row.difference, row.low_95, row.high_95],
            [center, center - half, center + half],
            rtol=0,
            atol=1e-13,
            equal_nan=True,
        )


def verify(source, path):
    # The source verifier already supplies an independent draw, stacked baseline,
    # and serialized-tree interpreter. None invokes a native OKAPI intervention.
    refs = runpy.run_path(str(source.resolve().parent / "verify.py"))
    evaluate, subtree, count_nodes = (
        refs[k] for k in ("evaluate", "subtree", "count_nodes")
    )
    check_sibling_and_nested(evaluate)
    source_rows = pd.read_csv(source, float_precision="round_trip")
    winners = source_rows[
        (source_rows.method == "fit_pair_fit_cut") & (source_rows.split == "fit")
    ]
    assert len(winners) == 192
    runs = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    assert set(runs.data_seed) == set(range(4100, 4116))
    assert not runs.duplicated(
        [
            "data_seed",
            "fit_regime",
            "test_regime",
            "decision",
            "provenance",
            "gp_seed",
            "method",
        ]
    ).any()
    assert len(runs) == 16 * 3 * (3 * 3 + 2 * 3 * 3 * 3)
    programs = changes = 0
    largest = 0.0
    for seed, bank in runs.groupby("data_seed"):
        (fit, test), _ = refs["make_draws"](seed, ((0, 512), (9701, 32768)))
        x = test[0]
        source_bank = winners[winners.data_seed == seed].set_index(
            ["fit_regime", "gp_seed", "decision"]
        )
        for law, group in bank.groupby("fit_regime"):
            baselines, (n, ranking, _) = refs["make_baselines"](
                fit[0], fit[2][law], [x]
            )
            cache = {}
            for row in group.itertuples():
                if row.method in baselines:
                    p = baselines[row.method][0]
                    assert pd.isna(row.rows_changed) and pd.isna(row.max_change)
                    if row.method == "top_n":
                        assert int(row.n_selected) == n
                        assert np.array_equal(json.loads(row.ranking), ranking)
                else:
                    winner = source_bank.loc[law, row.gp_seed, row.decision]
                    records = json.loads(winner.winner_occurrences)
                    if row.provenance == "archive_first":
                        paths = [tuple(r["path"]) for r in records]
                    else:
                        assert (
                            row.decision == "fit"
                            and row.provenance == "first_marked_copy"
                        )
                        paths = [
                            tuple(v)
                            for v in json.loads(winner.lineage_winner_ever_paths)
                        ]
                    assert json.loads(row.occurrence_paths) == [list(v) for v in paths]
                    assert (
                        row.occurrences == len(paths)
                        and row.admitted == winner.admitted
                    )
                    key = winner.program, tuple(paths), row.method
                    if key not in cache:
                        original = json.loads(winner.program)
                        check_paths(original, paths, subtree)
                        variant = edit_program(original, set(paths), row.method)
                        assert (
                            count_nodes(original) == row.intact_nodes
                            and count_nodes(variant) == row.edited_nodes
                        )
                        first = evaluate(original, x)
                        edited = evaluate(variant, x)
                        delta = np.abs(first - edited)
                        cache[key] = (
                            edited,
                            int(np.count_nonzero(delta)),
                            float(delta.max()),
                        )
                    p, changed, maximum = cache[key]
                    assert row.rows_changed == changed and row.max_change == maximum
                    programs += 1
                    changes += int(changed > 0)
                assert (
                    hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()
                    == row.prediction_sha256
                )
                target, truth = test[2][row.test_regime], test[1][row.test_regime]
                error = max(
                    abs(np.mean((p - target) ** 2) - row.observed_brier),
                    abs(np.mean((p - truth) ** 2) - row.conditional_excess),
                )
                assert error < 1e-13, (seed, law, row.method, row.test_regime, error)
                largest = max(largest, error)
    summaries = pd.read_csv(path / "paired.csv", float_precision="round_trip")
    verify_intervals(runs[runs.gp_seed.notna()], summaries)
    summary = {
        "okapi_commit": "c3c4060e1474f502fdcb0784b2274e71d9feb5ea",
        "protocol_commit": "6f1954a5ea91e609cb9a6b4f22060e42d9187a7b",
        "matched_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ablations_sha256": hashlib.sha256(
            (path / "runs.csv.gz").read_bytes()
        ).hexdigest(),
        "prediction_vectors": len(runs),
        "losses": 2 * len(runs),
        "frozen_fit_and_validation_winners": len(winners),
        "program_intervention_target_rows": programs,
        "rows_with_changed_predictions": changes,
        "paired_contrasts": len(summaries),
        "maximum_absolute_loss_difference": largest,
    }
    (path / "verified.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Verified {len(runs)} prediction vectors and {2 * len(runs)} losses, {programs} "
        f"program-intervention-target rows ({changes} had changed predictions), and "
        f"{len(summaries)} paired contrasts. Maximum loss difference {largest:.3g}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Matched-graft runs.csv.gz beside its verify.py",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source, args.out)
