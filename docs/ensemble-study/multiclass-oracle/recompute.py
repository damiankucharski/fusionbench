"""Recover corrected probability-oracle diagnostics for two archived studies without retraining GP."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss

from fusionbench.eval.full_suite_point import data_for, score_spread, summarize
from fusionbench.scenarios import calibrated

BASE = Path(__file__).resolve().parents[1]
CASES = {"saturated", "multiclass_committee"}


def module(path):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


def risk_oracle(q, y):
    eye = np.eye(q.shape[1])[y]
    return {
        "brier": np.mean(np.sum((q - eye) ** 2, axis=1)),
        "log_loss": log_loss(y, q, labels=np.arange(q.shape[1])),
        "accuracy": accuracy_score(y, q.argmax(1)),
        "macro_f1": f1_score(y, q.argmax(1), average="macro"),
    }


def recalc(folder):
    folder.mkdir(exist_ok=True, parents=True)
    ref = module(BASE / "full-suite-point" / "verify_predictions.py")
    opp = module(BASE / "mutation-opportunity" / "summarize.py")
    files = {
        "full-suite-point": ["runs.csv.gz"],
        "mutation-opportunity": ["runs.csv.gz", "refresh-independent-runs.csv.gz"],
    }
    effects = []
    oracle_scores = {}
    hashes = {}
    output_rows = []
    for study, sources in files.items():
        frames = []
        for fp in sources:
            frame = pd.read_csv(BASE / study / fp, float_precision="round_trip")
            for (name, seed), indices in (
                frame[frame.scenario.isin(CASES)]
                .groupby(["scenario", "data_seed"])
                .groups.items()
            ):
                sc, splits = data_for(name, seed)
                baseline, *_ = ref.baseline_fit(*splits["fit"][:2])
                k = sc.k
                rho = getattr(sc, "corr", 0)
                w_old = 1 / k if name == "saturated" else 1
                w = calibrated.mc_strength(sc.accuracy, sc.n_classes) / (
                    1 + (k - 1) * rho
                )
                for split, (x, y, corrected) in splits.items():
                    ix = [i for i in indices if frame.at[i, "split"] == split]
                    old = calibrated.multiclass_independent_oracle(
                        x, strengths=np.full(k, w_old) if name == "saturated" else None
                    )
                    refq = softmax(w * np.log(x).sum(axis=1), axis=1)
                    np.testing.assert_allclose(corrected, refq, atol=1e-14, rtol=1e-13)
                    assert np.min(x) > 1e-12 and np.array_equal(
                        corrected.argmax(1), old.argmax(1)
                    )
                    h0, h1 = ref.digest(old), ref.digest(corrected)
                    metrics0, metrics1 = risk_oracle(old, y), risk_oracle(corrected, y)
                    oracle_scores[(study, name, seed, split)] = {
                        "study": study,
                        "scenario": name,
                        "data_seed": seed,
                        "split": split,
                        "n": len(y),
                        "old_weight": w_old,
                        "weight": w,
                        "smallest_input_probability": float(x.min()),
                        **{f"old_{m}": metrics0[m] for m in metrics0},
                        **{f"corrected_{m}": metrics1[m] for m in metrics1},
                    }
                    probs = baseline(x)
                    for i in ix:
                        r = frame.loc[i]
                        assert r.oracle_sha256 == h0
                        key = r.method if pd.isna(r.program) else r.program
                        if key not in probs:
                            probs[key] = ref.read_program(json.loads(r.program), x)
                        p = probs[key]
                        assert ref.digest(p) == r.prediction_sha256
                        assert (
                            abs(
                                r.brier
                                - np.mean(np.sum((p - np.eye(sc.n_classes)[y]) ** 2, 1))
                            )
                            < 1e-14
                        )
                        e0 = np.mean(np.sum((p - old) ** 2, axis=1))
                        e1 = np.mean(np.sum((p - corrected) ** 2, axis=1))
                        assert abs(e0 - r.conditional_excess) < 1e-15
                        keys = [
                            "scenario",
                            "data_seed",
                            "gp_seed",
                            "method",
                            "checkpoint",
                            "selection",
                            "split",
                            "prediction_sha256",
                        ]
                        output_rows.append(
                            {
                                "study": study,
                                "source": fp,
                                **{k: r[k] for k in keys},
                                "former_oracle_sha256": h0,
                                "corrected_oracle_sha256": h1,
                                "former_conditional_excess": e0,
                                "corrected_conditional_excess": e1,
                            }
                        )
                        frame.at[i, "conditional_excess"] = e1
                        frame.at[i, "oracle_sha256"] = h1
            frames.append(frame)
        full = pd.concat(frames)
        if study == "full-suite-point":
            means, paired = summarize(full)
            means[means.scenario.isin(CASES)][
                [
                    "scenario",
                    "checkpoint",
                    "selection",
                    "split",
                    "method",
                    "conditional_excess",
                ]
            ].to_csv(folder / "suite-methods.csv", index=False)
            paired[
                paired.scenario.isin(CASES) & paired.metric.eq("conditional_excess")
            ].to_csv(folder / "suite-pairs.csv", index=False)
            spread = score_spread(full)
            spread[
                spread.scenario.isin(CASES) & spread.metric.eq("conditional_excess")
            ].to_csv(folder / "suite-spread.csv", index=False)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                opp.synthetic(full, tmp)
                for src in ("methods.csv.gz", "paired.csv.gz", "suite-contrasts.csv"):
                    tab = pd.read_csv(tmp / src)
                    tab = tab[tab.metric.eq("conditional_excess")]
                    if "scenario" in tab:
                        tab = tab[tab.scenario.isin(CASES)]
                    tab.to_csv(
                        folder / ("opportunity-" + src.replace(".gz", "")), index=False
                    )
    pd.DataFrame(output_rows).to_csv(folder / "corrected-rows.csv.gz", index=False)
    summary = pd.DataFrame(oracle_scores.values())
    summary.to_csv(folder / "oracle-scores.csv", index=False)
    for (study, case), g in summary[summary.split == "iid"].groupby(
        ["study", "scenario"]
    ):
        effects.append(
            dict(
                study=study,
                scenario=case,
                **g[[c for c in g.columns if c.startswith(("old_", "corrected_"))]]
                .mean()
                .to_dict(),
            )
        )
    pd.DataFrame(effects).to_csv(folder / "iid-oracle.csv", index=False)
    for r in output_rows:
        hashes[(r["study"], r["source"])] = hashes.get((r["study"], r["source"]), 0) + 1
    audit = {
        "replayed_affected_rows": len(output_rows),
        "affected_split_groups": len(oracle_scores),
        "checked_predictions_and_label_brier_unchanged": len(output_rows),
        "affected_rows_by_archive": {"/".join(k): v for k, v in hashes.items()},
        "minimum_observed_provider_probability": float(
            summary.smallest_input_probability.min()
        ),
    }
    (folder / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    recalc(parser.parse_args().output)
