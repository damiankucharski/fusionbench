"""Recreate every decision and baseline probability vector with a separate tree evaluator."""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from fusionbench.eval.opportunity_wine import build_wine_fold
from fusionbench.scenarios.registry import STANDARD_V1_SCENARIOS as SCENARIOS

# Reuse the earlier immutable replay's NumPy interpreter and independent baseline fitter.
previous = (
    Path(__file__).resolve().parents[1] / "full-suite-point" / "verify_predictions.py"
)
spec = spec_from_file_location("suite_prediction_reference", previous)
reference = module_from_spec(spec)
spec.loader.exec_module(reference)


def inputs(name, seed):
    if name == "wine":
        fold = build_wine_fold(seed)
        return {
            k: (v.preds, v.y, None)
            for k, v in (
                ("fit", fold.fit),
                ("validation", fold.validation),
                ("iid", fold.test),
            )
        }
    splits = {}
    for index, (split, count) in enumerate(
        (("fit", 384), ("validation", 256), ("iid", 2048), ("shift", 2048))
    ):
        options = {}
        if split == "shift":
            if name not in reference.SHIFTS:
                continue
            options = reference.SHIFTS[name]
        scenario = SCENARIOS[name](**options)
        p, y, latents = scenario._generate(count, np.random.default_rng([seed, index]))
        splits[split] = p, y, scenario.bayes_oracle(p, latents)
    return splits


def verify(folder):
    raw = pd.concat(
        [
            pd.read_csv(folder / f, float_precision="round_trip")
            for f in (
                "runs.csv.gz",
                "wine-runs.csv.gz",
                "refresh-independent-runs.csv.gz",
                "wine-refresh-independent-runs.csv.gz",
            )
        ]
    )
    assert set(raw.scenario) == set(SCENARIOS) | {"wine"}
    assert set(raw[raw.split.eq("shift")].scenario) == set(reference.SHIFTS)
    assert set(raw[raw.scenario.ne("wine")].data_seed) == set(range(5100, 5106))
    assert set(raw[raw.scenario.eq("wine")].data_seed) == set(range(5))
    vectors = metrics = unique_native = 0
    largest = 0.0
    for (name, seed), group in raw.groupby(["scenario", "data_seed"]):
        splits = inputs(name, seed)
        x, fy, _ = splits["fit"]
        baselines, order, count, prefixes = reference.baseline_fit(x, fy)
        for split, (z, y, oracle) in splits.items():
            panel = group[group.split.eq(split)]
            assert len(panel) == 43
            assert not panel.duplicated(
                ["method", "selection", "checkpoint", "gp_seed"]
            ).any()
            models = baselines(z)
            ranked = panel[panel.method.eq("top_n")].iloc[0]
            assert (
                json.loads(ranked.ranking) == list(order) and ranked.n_selected == count
            )
            np.testing.assert_allclose(
                json.loads(ranked.prefix_scores), prefixes, atol=3e-15, rtol=0
            )
            for r in panel.itertuples():
                key = r.method if r.checkpoint == "fixed" else r.program
                if key not in models:
                    models[key] = reference.read_program(json.loads(r.program), z)
                    unique_native += 1
                p = models[key]
                assert p.shape == (r.rows, r.classes) and p.shape == (
                    len(y),
                    z.shape[-1],
                )
                assert (
                    r.providers == z.shape[1]
                    and np.isfinite(p).all()
                    and (p >= 0).all()
                )
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-12, rtol=0)
                assert reference.digest(p) == r.prediction_sha256, (
                    name,
                    seed,
                    split,
                    r.method,
                )
                canon = reference.normalized(p)
                yh = canon.argmax(1)
                scores = {
                    "brier": np.mean(np.sum((np.eye(r.classes)[y] - p) ** 2, 1)),
                    "macro_f1": f1_score(y, yh, average="macro", zero_division=0),
                    "accuracy": accuracy_score(y, yh),
                    "log_loss": log_loss(y, canon, labels=np.arange(r.classes)),
                }
                if oracle is None:
                    assert pd.isna(r.conditional_excess) and pd.isna(r.oracle_sha256)
                else:
                    scores["conditional_excess"] = np.mean(np.sum((p - oracle) ** 2, 1))
                    assert reference.digest(oracle) == r.oracle_sha256
                err = max(abs(v - getattr(r, k)) for k, v in scores.items())
                assert err < 5e-12, (name, seed, split, r.method, err)
                vectors += 1
                metrics += len(scores)
                largest = max(largest, err)
    report = {
        "bitwise_vectors": vectors,
        "checked_metrics": metrics,
        "max_error": largest,
        "decoded_unique_program_split_pairs": unique_native,
        "regenerated_scenario_banks": 114,
        "regenerated_external_folds": 5,
    }
    (folder / "predictions-verified.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
