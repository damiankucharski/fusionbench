import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fusionbench.cli import main
from fusionbench.eval.transfer_grid import brier, iter_test_conditions
from fusionbench.methods.averaging import SimpleAverage
from fusionbench.methods.registry import FACTORIES
from fusionbench.scenarios.native_switch import NativeSwitch
from fusionbench.scenarios.registry import SCENARIOS
from fusionbench.transfer import load_preset, resolve_config, summarize


def small():
    cfg = load_preset()
    cfg.update(
        fit_rows=24, test_rows=30, methods=["simple_average"], dev_noise=[0.0, 0.15]
    )
    return cfg


def test_cli_paired_fingerprints_match_panel_and_subset(tmp_path):
    expected_seed = 1_714_000
    options = [
        "--preset",
        "smoke",
        "--methods",
        "simple_average",
        "--dev-noise",
        "0",
        ".15",
    ]
    main(["transfer", *options, "--out", str(tmp_path / "grid")])
    manifest = json.loads((tmp_path / "grid/manifest.json").read_text())
    bank = next(b for b in manifest["banks"] if b["scenario"] == "native_switch")
    assert bank["data_seed"] == expected_seed
    score = pd.read_csv(tmp_path / "grid/scores.csv")
    clean = score[(score.scenario == "native_switch") & (score["mode"] == "clean")]
    _, _, x, y = next(iter_test_conditions("native_switch", expected_seed, 2048))
    assert np.allclose(
        clean.brier, brier(SimpleAverage().predict(x), y), rtol=0, atol=1e-14
    )
    assert np.isclose(clean.brier.iloc[0], 0.44858456982506645, rtol=0, atol=1e-13)
    main(
        [
            "transfer-grid",
            *options,
            "--scenarios",
            "native_switch",
            "--out",
            str(tmp_path / "one"),
        ]
    )
    second = json.loads((tmp_path / "one/manifest.json").read_text())
    assert second["banks"] == [bank]
    assert set(score["mode"]) == {
        "clean",
        "canonical_shift",
        "log_jitter",
        "provider_swap",
    }
    assert len(pd.read_csv(tmp_path / "grid/fits.csv")) == 4


def test_custom_registered_scenario_method_and_replay(tmp_path, monkeypatch):
    class CustomSwitch(NativeSwitch):
        pass

    fit_calls = []

    class TrackedAverage(SimpleAverage):
        def fit(self, fit_preds, fit_y, **options):
            fit_calls.append(tuple(fit_y))
            return super().fit(fit_preds, fit_y, **options)

    monkeypatch.setitem(SCENARIOS, "external_switch", CustomSwitch)
    monkeypatch.setitem(FACTORIES, "my_average", TrackedAverage)
    cfg = small()
    cfg.update(
        scenarios={
            "external_switch": {
                "seed_slot": 71,
                "knobs": {"mixing": 0.2},
                "canonical_shift": {"mixing": 0.7},
            }
        },
        methods=["my_average"],
        repeats=2,
    )
    source = tmp_path / "panel.yaml"
    source.write_text(yaml.safe_dump(cfg))
    main(["transfer", "--config", str(source), "--out", str(tmp_path / "a")])
    assert len(fit_calls) == 4
    main(
        [
            "transfer",
            "--config",
            str(tmp_path / "a/resolved.yaml"),
            "--out",
            str(tmp_path / "b"),
        ]
    )
    assert len(fit_calls) == 8
    assert (tmp_path / "a/scores.csv").read_bytes() == (
        tmp_path / "b/scores.csv"
    ).read_bytes()
    data = pd.read_csv(tmp_path / "a/scores.csv")
    assert (
        len(data) == 16 and (data[data["mode"] == "clean"].delta_from_clean == 0).all()
    )
    assert sorted(data.data_seed.unique()) == [1771000, 1771001]


def test_optional_error_and_controls(monkeypatch, tmp_path):
    class Unavailable(SimpleAverage):
        @classmethod
        def available(cls):
            return False

    monkeypatch.setitem(FACTORIES, "unavailable", Unavailable)
    with pytest.raises(
        SystemExit, match="Requested method 'unavailable' is unavailable"
    ):
        main(
            [
                "transfer-grid",
                "--methods",
                "unavailable",
                "--out",
                str(tmp_path / "missing"),
            ]
        )
    assert not (tmp_path / "missing").exists()
    cfg = small()
    monkeypatch.setitem(FACTORIES, "okapi", SimpleAverage)
    cfg["methods"] = ["okapi", "stacking_logit"]
    assert resolve_config(cfg)["methods"] == [
        "okapi",
        "stacking_logit",
        "simple_average",
        "top_n",
        "stacking_lr",
    ]
    with pytest.raises(ValueError, match="constructor knob"):
        resolve_config(
            {
                **cfg,
                "scenarios": {
                    "native_switch": {"seed_slot": 0, "knobs": {"mixng": 0.4}}
                },
            }
        )
    with pytest.raises(ValueError, match="overlap"):
        resolve_config(
            {
                **cfg,
                "scenarios": {
                    "native_switch": {"seed_slot": 0},
                    "committee": {"seed_slot": 0},
                },
            }
        )


def test_equal_scenario_weight_when_endpoints_differ():
    rows = []
    for scenario, modes, risk in (
        ("a", ["clean", "log_jitter", "canonical_shift"], 0.1),
        ("b", ["clean", "log_jitter"], 0.5),
    ):
        for rep in range(2):
            for mode in modes:
                rows.append(
                    {
                        "scenario": scenario,
                        "method": "x",
                        "dev_noise": 0.0,
                        "mode": mode,
                        "severity": 0.0 if mode == "clean" else 1.0,
                        "replicate": rep,
                        "brier": risk,
                        "delta_from_clean": 0.0,
                    }
                )
    summary, overall = summarize(pd.DataFrame(rows))
    assert np.isclose(overall.brier.item(), 0.3)
    assert summary[summary["mode"] == "canonical_shift"].scenarios.item() == 1


def test_validation_default_override_and_independent_noisy_streams(
    tmp_path, monkeypatch
):
    import hashlib

    from fusionbench.eval.transfer_grid import STREAMS, noisy_labels
    from fusionbench.scenarios.blocks import draw_block

    fitted = []
    source = small()
    source.pop("validation_rows")  # Previously published schema v1.
    source.update(
        scenarios={"native_switch": {"seed_slot": 14}},
        methods=["tracked"],
        dev_noise=[0.0, 0.4],
        fit_rows=83,
        test_rows=43,
    )

    class Tracked(SimpleAverage):
        def fit(self, x, y, **kw):
            assert not x.flags.writeable and not y.flags.writeable
            fitted.append((x.copy(), y.copy()))
            return super().fit(x, y, **kw)

        def predict(self, x):
            assert not x.flags.writeable
            return super().predict(x)

    monkeypatch.setitem(FACTORIES, "tracked", Tracked)
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(yaml.safe_dump(source))
    main(
        [
            "transfer-grid",
            "--config",
            str(config_file),
            "--out",
            str(tmp_path / "default"),
        ]
    )
    main(
        [
            "transfer",
            "--config",
            str(config_file),
            "--validation-rows",
            "83",
            "--out",
            str(tmp_path / "override"),
        ]
    )
    assert len(fitted) == 4
    seed = 1_714_000
    train_x, train_y, _ = draw_block(NativeSwitch(), seed, 0, 83)
    val_x, val_y, _ = draw_block(NativeSwitch(), seed, 3, 83)
    np.testing.assert_array_equal(fitted[1][0], train_x)
    np.testing.assert_array_equal(fitted[1][1], noisy_labels(train_y, 2, seed, 0.4))
    np.testing.assert_array_equal(fitted[1][1], fitted[3][1])
    assert not np.array_equal(train_x, val_x)
    rng = np.random.default_rng([seed, 14])
    noisy_val = np.where(rng.random(83) < 0.4, 1 - val_y, val_y)
    assert not np.array_equal(noisy_val != val_y, fitted[1][1] != train_y)
    assert (
        STREAMS["source_validation"] == 3
        and STREAMS["validation_annotation_noise"] == 14
    )
    manifests = [
        json.loads((tmp_path / folder / "manifest.json").read_text())
        for folder in ("default", "override")
    ]
    assert [m["config"]["validation_rows"] for m in manifests] == [192, 83]
    own = manifests[1]["banks"][0]
    for what, arr in (
        (own["validation_inputs_sha256"], val_x),
        (own["validation_annotations"]["0.4"], noisy_val),
    ):
        assert what == hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    assert (tmp_path / "default/scores.csv").read_bytes() == (
        tmp_path / "override/scores.csv"
    ).read_bytes()
    fits = pd.read_csv(tmp_path / "override/fits.csv").set_index("dev_noise")
    assert np.isclose(
        fits.loc[0.4, "validation_brier"],
        brier(SimpleAverage().predict(val_x), noisy_val),
    )
    with pytest.raises(ValueError, match="validation_rows"):
        resolve_config({**source, "validation_rows": 0})


def test_generic_selection_exact_validation_ties_tolerant_hits_and_candidate_sets(
    tmp_path, monkeypatch
):
    from fusionbench.transfer import run_transfer

    class Constant(SimpleAverage):
        def __init__(self, p):
            self.p = p

        def predict(self, x):
            return np.tile([1 - self.p, self.p], (len(x), 1))

    # For this seeded bank the one validation row has label 1; test prevalence is .4.
    methods = ["half", "second_exact", "first_alphabetically"]
    monkeypatch.setitem(FACTORIES, methods[0], lambda: Constant(0.5))
    for key in methods[1:]:
        monkeypatch.setitem(FACTORIES, key, lambda: Constant(0.5 + 1e-13))
    cfg = small()
    cfg.update(
        base_seed=0,
        validation_rows=1,
        scenarios={"native_switch": {"seed_slot": 0}},
        methods=methods,
        modes=[],
        dev_noise=[0.0],
    )
    run_transfer(cfg, tmp_path)
    row = pd.read_csv(tmp_path / "selection.csv").iloc[0]
    assert row.n_methods == 3 and row.validation_best_method == methods[1]
    assert (
        row.test_best_method == "half" and 0 < row.test_excess < 1e-12 and row.test_hit
    )
    fits, scores = [pd.read_csv(tmp_path / file) for file in ("fits.csv", "scores.csv")]
    assert set(fits.method) == set(scores.method) == set(methods)
    assert (
        fits.loc[fits.method == methods[1], "validation_brier"].item()
        < fits.loc[fits.method == "half", "validation_brier"].item()
    )
    np.testing.assert_allclose(
        row.selected_test_brier, scores.loc[scores.method == methods[1], "brier"].item()
    )
