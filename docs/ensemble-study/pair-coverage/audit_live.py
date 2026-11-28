"""Replay the fixed-budget experiment and cross-check its probes against terminal menus."""

import ast
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from live import banks, fit, schedule

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import brier

OUT = Path(__file__).resolve().parent / "results"


def main():
    started = perf_counter()
    runs = pd.read_csv(OUT / "live-scores.csv").set_index(["family", "seed", "arm"])
    trace = pd.read_csv(OUT / "live-generations.csv.gz").set_index(
        ["family", "seed", "arm"]
    )
    controlled = pd.read_csv(OUT / "r1-probes.csv").query("policy == 'unique'")
    panel = pd.read_csv(OUT / "r3-proposals.csv").query("policy == 'unique'")
    checked_calls = checked_probes = 0
    for family, seed, x, y, test, test_y, q in banks():
        is_panel = q is None
        if is_panel:
            selected = panel.loc[
                (panel.scenario_id == family) & (panel.data_seed == seed)
            ].sort_values("slot")
            menu = list(
                zip(
                    selected.operator,
                    selected.provider_left,
                    selected.provider_right,
                    strict=True,
                )
            )
        else:
            selected = controlled.loc[
                (controlled.law == family) & (controlled.seed == seed)
            ].sort_values("slot")
            menu = [
                ("min", int(row.left), int(row.right)) for row in selected.itertuples()
            ]
        assert tuple(menu) == schedule(x.shape[1], seed, is_panel)
        for arm in ("ordinary", "probes"):
            cell = family, seed, arm
            rec = runs.loc[cell]
            log = trace.loc[[cell]].sort_values("generation")
            expected_founders = min(8, x.shape[1]) + 1
            np.testing.assert_array_equal(
                log.objective_calls, expected_founders + 8 * np.arange(1, 17)
            )
            inserted = [ast.literal_eval(s) for s in log.inserted.dropna()]
            assert inserted == (menu if arm == "probes" else [])
            if inserted:
                assert log.generation[log.inserted.notna()].tolist() == list(
                    range(1, 17, 2)
                )
            search_seed = 0 if is_panel else seed + 1_000_000
            tree, info, replay = fit(x, y, search_seed, inserted)
            assert (
                info["winner_digest"] == rec.winner_digest
                and info["founder_digest"] == rec.founder_digest
            )
            assert [
                step["survivor_digest"] for step in replay
            ] == log.survivor_digest.tolist()
            p = predict_tree(tree, test)
            np.testing.assert_allclose(
                brier(p, test_y), rec.test_brier, rtol=0, atol=1e-12
            )
            if q is not None:
                np.testing.assert_allclose(
                    2 * np.mean((p[:, 1] - q) ** 2), rec.test_excess, rtol=0, atol=1e-12
                )
            checked_probes += len(inserted)
            checked_calls += int(rec.calls)

    suite = runs[runs.cohort == "panel"].reset_index()
    risk = suite.pivot(index=["family", "seed"], columns="arm", values="test_brier")
    family_delta = (risk.probes - risk.ordinary).groupby("family").mean()
    # Descriptive concentration: identify the two most improved families after observation.
    top_two = family_delta.nsmallest(2)
    other = family_delta.drop(top_two.index)
    gaps = (suite.test_brier - suite.fit_brier).groupby(suite.arm).mean()
    main_suite = pd.read_csv(OUT / "r3-scores.csv")
    panel_cells = main_suite.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    )
    vs_lr = (
        (panel_cells.regularized - panel_cells.prob_lr)
        .groupby(level="scenario_id")
        .mean()
    )
    result = {
        "replayed_fits": len(runs),
        "replayed_calls_including_founders": checked_calls,
        "verified_survivor_generations": len(trace),
        "insertions_matching_terminal_menu": checked_probes,
        "test_minus_fit_by_arm": gaps.to_dict(),
        "live_family_mean_delta": family_delta.to_dict(),
        "two_most_improved_families": top_two.to_dict(),
        "fraction_of_net_panel_gain_from_those_two": top_two.sum() / family_delta.sum(),
        "remaining_23_macro_delta": other.mean(),
        "fixed_families_lower_brier_for_regularized_than_lr": int(
            (vs_lr < -1e-12).sum()
        ),
        "regularized_family_selection": main_suite.loc[
            main_suite.method == "regularized", "family"
        ]
        .value_counts()
        .to_dict(),
        "wall_seconds": perf_counter() - started,
    }
    (OUT / "live-audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
