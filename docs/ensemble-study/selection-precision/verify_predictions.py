"""Independently replay all ordered decision trees and reconstruct the primary intervals."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, f1_score, log_loss


def reference():
    path = (
        Path(__file__).resolve().parents[1]
        / "mutation-opportunity"
        / "verify_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("opportunity_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(folder):
    ref = reference()
    raw = pd.read_csv(
        folder / "runs.csv.gz", float_precision="round_trip", dtype={"checkpoint": str}
    )
    vectors = scores = 0
    max_error = 0
    for (name, seed), group in raw.groupby(["scenario", "data_seed"]):
        splits = ref.inputs(name, seed)
        x, y, _ = splits["fit"]
        base, order, count, prefix = ref.reference.baseline_fit(x, y)
        for s, (p, y, q) in splits.items():
            candidates = base(p)
            part = group[group.split.eq(s)]
            assert len(part) == 27
            for row in part.itertuples():
                if row.method == "top_n":
                    assert (
                        list(order) == json.loads(row.ranking)
                        and row.n_selected == count
                    )
                    np.testing.assert_allclose(
                        json.loads(row.prefix_scores), prefix, rtol=0, atol=3e-15
                    )
                key = row.method if row.checkpoint == "fixed" else row.program
                if key not in candidates:
                    candidates[key] = ref.reference.read_program(
                        json.loads(row.program), p
                    )
                v = candidates[key]
                norm = ref.reference.normalized(v)
                assert (
                    ref.reference.digest(v) == row.prediction_sha256
                    and ref.reference.digest(q) == row.oracle_sha256
                )
                wanted = {
                    "brier": np.mean(np.sum((v - np.eye(p.shape[2])[y]) ** 2, 1)),
                    "conditional_excess": np.mean(np.sum((v - q) ** 2, 1)),
                    "log_loss": log_loss(y, norm, labels=np.arange(p.shape[2])),
                    "accuracy": accuracy_score(y, norm.argmax(1)),
                    "macro_f1": f1_score(
                        y, norm.argmax(1), average="macro", zero_division=0
                    ),
                }
                difference = max(abs(z - getattr(row, m)) for m, z in wanted.items())
                assert difference < 1e-13, (name, seed, s, row.method, difference)
                max_error = max(max_error, difference)
                vectors += 1
                scores += len(wanted)
    # Independent direct per-start pairing; average fixed cases within each common replicate before the interval.
    sut = pd.read_csv(folder / "suite-contrasts.csv", dtype={"checkpoint": str})
    checked = 0
    for r in sut.query('metric=="brier"').itertuples():
        g = raw[
            raw.checkpoint.eq(r.checkpoint)
            & raw.selection.eq(r.selection)
            & raw.split.eq(r.split)
        ]
        ends = g.set_index(["data_seed", "scenario", "gp_seed", "method"]).brier
        ds = []
        for bank in sorted(g.data_seed.unique()):
            ds.append(
                statistics.mean(
                    ends[bank, f, i, r.a] - ends[bank, f, i, r.b]
                    for f in sorted(g.scenario.unique())
                    for i in (709, 719)
                )
            )
        m = statistics.mean(ds)
        margin = statistics.stdev(ds) * t.ppf(0.975, 5) / np.sqrt(6)
        np.testing.assert_allclose(
            [m, m - margin, m + margin],
            [r.mean, r.low_95, r.high_95],
            atol=2e-14,
            rtol=1e-12,
        )
        checked += 1
    report = {
        "vectors_bitwise": vectors,
        "verified_metrics": scores,
        "max_absolute_error": max_error,
        "aggregate_brier_intervals": checked,
    }
    (folder / "predictions-verified.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
