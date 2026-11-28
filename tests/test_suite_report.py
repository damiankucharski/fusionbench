import json

import numpy as np
import pandas as pd
import pytest

from fusionbench import run_suite, summarize_suite
from fusionbench.cli import main
from fusionbench.methods.averaging import SimpleAverage


def test_named_panel_from_cli_preserves_weights_and_ids(tmp_path):
    main(
        [
            "run",
            "--suite",
            "standard-v6",
            "--group",
            "context_routing",
            "--n",
            "40",
            "--methods",
            "simple_average",
            "--out",
            str(tmp_path),
        ]
    )
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert set(manifest["scenario_weights"]) == {
        "gated_experts",
        "rare_specialist",
        "selective_fallback",
    }
    assert sum(manifest["scenario_weights"].values()) == pytest.approx(1)
    raw = pd.read_csv(tmp_path / "results.csv")
    scores = raw[raw.method == "simple_average"]
    panel = pd.read_csv(tmp_path / "mechanisms.csv").set_index("group")
    assert set(panel.index) == {"selected", "context_routing"}
    assert set(panel.scored) == {3}
    np.testing.assert_allclose(panel.mean_brier, scores.test_brier.mean())
    assert set(scores.scenario_id) == set(manifest["scenarios"])


def test_scoring_never_hides_failed_or_missing_selected_banks():
    raw = pd.DataFrame(
        [
            {
                "scenario_id": s,
                "method": m,
                "data_seed": seed,
                "test_brier": {"committee": 0.4, "parity": 0.1}[s],
                "error": "",
            }
            for seed in (0, 1)
            for s in ("committee", "parity")
            for m in ("a", "b")
        ]
    )
    weights = {"committee": 1, "parity": 3}

    def selected(data):
        summary = summarize_suite(data, weights=weights)
        return summary[summary.group == "selected"].set_index("method")

    np.testing.assert_allclose(selected(raw).mean_brier, 0.175)
    for failed in (raw.drop(index=0), raw.assign(error=["exception"] + [""] * 7)):
        out = selected(failed)
        assert out.loc["a", "scored"] == 1 and np.isnan(out.loc["a", "mean_brier"])
        assert out.loc["b", "mean_brier"] == pytest.approx(0.175)

    with pytest.raises(ValueError, match="one result per"):
        selected(pd.concat([raw, raw.iloc[:1]], ignore_index=True))


def test_one_id_works_from_normal_entry_point():
    got = run_suite("committee", [SimpleAverage()], n_per_split=24, seed=3)
    assert set(got.scenario_id) == {"committee"}
    assert set(got.data_seed) == {3}


def test_default_suite_methods_are_the_three_lightweight_references():
    result = run_suite("committee", n_per_split=30, seed=6)
    fitted = result.loc[result.method != "oracle"]
    assert fitted.method.tolist() == ["simple_average", "top_n", "stacking_lr"]
    assert fitted.error.eq("").all()


def test_method_ids_and_iterators_work_for_each_scenario():
    methods = iter(["simple_average", "top_n"])
    scores = run_suite(["committee", "miscalibration"], methods, n_per_split=24, seed=4)
    fitted = scores[scores.method != "oracle"]
    assert fitted.groupby("scenario_id").method.nunique().to_dict() == {
        "committee": 2,
        "miscalibration": 2,
    }
    assert fitted.error.eq("").all()
