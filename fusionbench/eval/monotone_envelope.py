"""Check monotone symbolic envelopes against exhaustive and nonclosed classes."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.lattice_search import SmallLattice, small_lattice_programs
from okapi.monotone_envelope import fit_monotone_envelope


def target_cube(d):
    return ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(int)


def small_panel(seed, cube, table):
    rng = np.random.default_rng(seed)
    observed = (rng.random((64, 4)) < 0.2).astype(int)
    indices = observed @ (2 ** np.arange(4))
    full = []
    restricted = []
    for n in [4, 12, 32, 64]:
        source = table[:, indices[:n]]
        for truth, labels in enumerate(source):
            keep = np.all(source == labels, axis=1)
            valid = table[keep]
            envelope = fit_monotone_envelope(observed[:n], labels)
            lo, hi = envelope.bounds(cube)
            np.testing.assert_array_equal(lo, valid.min(0))
            np.testing.assert_array_equal(hi, valid.max(0))
            has_lower = np.any(np.all(valid == lo, axis=1))
            has_upper = np.any(np.all(valid == hi, axis=1))
            assert has_lower and has_upper
            center = envelope.predict_positive(cube)
            posterior = valid.mean(0)
            D = np.mean(lo != hi)
            candidates = {
                "endpoint": center,
                "uniform": posterior,
                "simplest": valid[0],
            }
            for method, p in candidates.items():
                loss = np.mean((p - valid) ** 2, axis=1)
                if method == "endpoint":
                    np.testing.assert_allclose(loss, D / 4, rtol=0, atol=1e-14)
                full.append(
                    {
                        "seed": seed,
                        "labels": n,
                        "truth": truth,
                        "method": method,
                        "compatible": len(valid),
                        "ambiguity": D,
                        "worst_brier": loss.max(),
                        "average_brier": loss.mean(),
                        "true_brier": np.mean((p - table[truth]) ** 2),
                    }
                )
        for truth in range(4):
            idx = np.all(observed[:n] == observed[:n, truth, None], axis=0)
            restricted.append(restricted_cell(seed, n, truth, cube[:, idx].T))
    return full, restricted


def restricted_cell(seed, n, truth, valid):
    lower, upper = valid.min(0), valid.max(0)
    m = len(valid)
    mean = valid.mean(0)
    mid = (lower + upper) / 2
    uniform_risk = np.mean((valid - mean) ** 2, axis=1).max()
    midpoint_risk = np.mean((valid - mid) ** 2, axis=1).max()
    two = 0 if m == 1 else (0.125 if m == 2 else 0.375)
    np.testing.assert_allclose(uniform_risk, (m - 1) / (4 * m))
    np.testing.assert_allclose(midpoint_risk, (1 - 2.0 ** (1 - m)) / 4)
    if m > 1:
        fitted_two = 0.5 * (valid[0] + valid[1])
        np.testing.assert_allclose(
            np.mean((valid - fitted_two) ** 2, axis=1).max(), two
        )
    return {
        "seed": seed,
        "labels": n,
        "truth": truth,
        "compatible": m,
        "both_endpoints_members": bool(m == 1),
        "ambiguity": np.mean(lower != upper),
        "endpoint_brier": midpoint_risk,
        "minimax_brier": uniform_risk,
        "best_two_member_brier": two,
    }


def large_panel(seed, d):
    rng = np.random.default_rng([seed, d])
    columns = rng.choice(d, 3, replace=False)
    src = (rng.random((512, d)) < 0.2).astype(int)
    query = rng.integers(0, 2, size=(4096, d))
    y = src[:, columns].sum(1) >= 2
    yt = query[:, columns].sum(1) >= 2
    rows = []
    for n in [128, 512]:
        model = fit_monotone_envelope(src[:n], y[:n])
        lower, upper = model.bounds(query)
        assert np.all(lower <= yt) & np.all(yt <= upper)
        p = model.predict_positive(query)
        rows.append(
            {
                "seed": seed,
                "d": d,
                "labels": n,
                "target_ambiguity": np.mean(lower != upper),
                "brier": np.mean((p - yt) ** 2),
                "source_positives": int(y[:n].sum()),
            }
        )
    return rows


def run(out, seeds):
    start = datetime.now(UTC)
    timer = time.perf_counter()
    out.mkdir(exist_ok=True, parents=True)
    cube = target_cube(4)
    programs = small_lattice_programs(4)
    table = np.stack(
        [
            SmallLattice(terms, 4, len(programs)).predict(cube).astype(int)
            for terms in programs
        ]
    )
    assert np.unique(table, axis=0).shape == (166, 16)
    rows, restricted, large = [], [], []
    for seed in seeds:
        f, r = small_panel(seed, cube, table)
        rows += f
        restricted += r
        for d in (8, 16):
            large += large_panel(seed, d)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        out / "closed_cells.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    frame.groupby(["labels", "method"])[
        ["compatible", "ambiguity", "worst_brier", "average_brier", "true_brier"]
    ].mean().to_csv(out / "closed_means.csv")
    df = pd.DataFrame(restricted)
    df.to_csv(out / "restricted_cells.csv", index=False)
    df.groupby("labels")[
        [
            "compatible",
            "both_endpoints_members",
            "endpoint_brier",
            "minimax_brier",
            "best_two_member_brier",
        ]
    ].mean().to_csv(out / "restricted_means.csv")
    big = pd.DataFrame(large)
    big.to_csv(out / "higher_dimension.csv", index=False)
    big.groupby(["d", "labels"])[
        ["target_ambiguity", "brier", "source_positives"]
    ].mean().to_csv(out / "higher_means.csv")
    witness = restricted_cell(-1, 2, 0, cube.T)
    (out / "four_coordinate_control.json").write_text(
        json.dumps(witness, indent=2) + "\n"
    )
    config = {
        "seeds": seeds,
        "all_full_rules": len(programs),
        "source_bit_probability": 0.2,
        "target_bit_probability": 0.5,
        "small_n": [4, 12, 32, 64],
        "larger_d": [8, 16],
        "larger_n": [128, 512],
        "compute_start_utc": start.isoformat(),
        "compute_seconds": time.perf_counter() - timer,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1400, 1412)))
    args = parser.parse_args()
    run(args.output, args.seeds)
