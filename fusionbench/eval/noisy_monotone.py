"""Compare a cut-based monotone ERM interval with a noise-budget minimax center."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.brier_center import fit_brier_center
from okapi.lattice_search import SmallLattice, small_lattice_programs
from okapi.monotone_erm import fit_monotone_erm

from fusionbench.eval.monotone_envelope import target_cube


def rules(cube):
    n = cube.shape[1]
    programs = small_lattice_programs(n)
    return np.stack(
        [
            SmallLattice(terms, n, len(programs)).predict(cube).astype(int)
            for terms in programs
        ]
    )


def summarize(seed, n, truth, table, valid, case, pred, lower_bound=None):
    lo, hi = valid.min(0), valid.max(0)
    center = (lo + hi) / 2
    loss = np.mean((pred - valid) ** 2, axis=1)
    membership = np.any(np.all(valid == table[truth], axis=1))
    return {
        "seed": seed,
        "labels": n,
        "truth": truth,
        "candidate_set": case,
        "candidates": len(valid),
        "truth_in_set": membership,
        "endpoints_in_set": np.any(np.all(valid == lo, axis=1))
        and np.any(np.all(valid == hi, axis=1)),
        "ambiguity": np.mean(lo != hi),
        "endpoint_worst": np.mean((center - valid) ** 2, axis=1).max(),
        "optimal_worst": loss.max(),
        "certificate_gap": 0 if lower_bound is None else loss.max() - lower_bound,
        "actual_brier": np.mean((pred - table[truth]) ** 2),
        "endpoint_actual": np.mean((center - table[truth]) ** 2),
    }


def control():
    cube = target_cube(3)
    table = rules(cube)
    source = np.where(cube.sum(1) == 1)[0]
    valid = table[table[:, source].sum(1) <= 1][:, source]
    distinct = np.unique(valid, axis=0)
    fit = fit_brier_center(distinct)
    return {
        "candidates": distinct.tolist(),
        "prediction": fit.prediction.tolist(),
        "lower": fit.lower,
        "upper": fit.upper,
        "midpoint_brier": 0.25,
        "reference": 2 / 9,
    }


def run(out, seeds):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    cube = target_cube(4)
    table = rules(cube)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = (rng.random((32, 4)) < 0.2).astype(int)
        indices = x @ (2 ** np.arange(4))
        for truth in rng.choice(len(table), 6, replace=False):
            for n in [12, 32]:
                labels = table[truth, indices[:n]].copy()
                flip = int(rng.integers(n))
                labels[flip] = 1 - labels[flip]
                count = np.sum(table[:, indices[:n]] != labels, axis=1)
                minimum = count.min()
                best = table[count == minimum]
                bounded = table[count <= 1]
                cut = fit_monotone_erm(x[:n], labels)
                assert cut.min_errors == minimum
                lo, hi = cut.bounds(cube)
                np.testing.assert_array_equal(lo, best.min(0))
                np.testing.assert_array_equal(hi, best.max(0))
                a = summarize(
                    seed, n, truth, table, best, "exact_erm", cut.predict_positive(cube)
                )
                assert a["endpoints_in_set"]
                np.testing.assert_allclose(a["optimal_worst"], a["ambiguity"] / 4)
                # If the extremal functions are feasible, their midpoint is already the exact optimum.
                blo, bhi = bounded.min(0), bounded.max(0)
                endpoints = np.any(np.all(bounded == blo, axis=1)) and np.any(
                    np.all(bounded == bhi, axis=1)
                )
                if endpoints:
                    pred = (blo + bhi) / 2
                    dual = np.mean(blo != bhi) / 4
                else:
                    ball = fit_brier_center(bounded)
                    pred, dual = ball.prediction, ball.lower
                b = summarize(seed, n, truth, table, bounded, "one_error", pred, dual)
                assert b["truth_in_set"]
                rows += [
                    dict(min_errors=int(minimum), flipped_row=flip, **row)
                    for row in (a, b)
                ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "cells.csv", index=False)
    fields = [
        "candidates",
        "truth_in_set",
        "endpoints_in_set",
        "ambiguity",
        "endpoint_worst",
        "optimal_worst",
        "certificate_gap",
        "actual_brier",
        "endpoint_actual",
    ]
    df.groupby(["labels", "candidate_set"])[fields].mean().to_csv(out / "means.csv")
    (out / "control.json").write_text(json.dumps(control(), indent=2) + "\n")
    config = {
        "seeds": seeds,
        "rules_per_bank": 6,
        "source_positive_bit_probability": 0.2,
        "source_sizes": [12, 32],
        "flips_per_task": 1,
        "max_certificate_gap": df.certificate_gap.max(),
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1450, 1462)))
    args = parser.parse_args()
    run(args.output, args.seeds)
