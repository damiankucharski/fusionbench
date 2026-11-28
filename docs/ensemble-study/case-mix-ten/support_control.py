"""Can a full 16-cell source frequency table use the same case-mix weights?"""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from banks import _focus
from okapi.case_mix import case_mix_weights

OUT = Path(__file__).resolve().parent / "results"
PAIRS = (
    ("table_full", "table_native"),
    ("table_cap2", "table_native"),
    ("table_native", "native"),
    ("table_full", "native"),
    ("table_cap2", "native"),
    ("table_full", "full"),
    ("table_cap2", "cap2"),
    ("full", "native"),
    ("cap2", "native"),
)


def vote_pattern(x):
    """Index the four observed positive-class votes in anonymous column order."""
    if x.shape[1:] != (4, 2):
        raise ValueError("The finite table requires four binary provider reports")
    return (x[:, :, 1] > 0.5) @ np.array([8, 4, 2, 1])


def fit(codes, y, weights):
    # Give a symmetric half-count per class to every pattern, including unseen ones.
    w = weights / weights.mean()
    n = np.bincount(codes, weights=w, minlength=16)
    positive = np.bincount(codes, weights=w * y, minlength=16)
    return (positive + 0.5) / (n + 1)


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def measure(bank):
    codes = vote_pattern(bank.x)
    w_full, _ = case_mix_weights(bank.env, bank.ctx, min_cell=4)
    w_cap2, _ = case_mix_weights(bank.env, bank.ctx, min_cell=4, cap=2)
    weights = {
        "table_native": np.ones(len(codes)),
        "table_full": w_full,
        "table_cap2": w_cap2,
    }
    rows = []
    for name, w in weights.items():
        p = fit(codes, bank.y, w)
        common = {
            "family": bank.family,
            "seed": bank.seed,
            "method": name,
            "n_fit": len(bank.y),
            "observed_patterns": np.unique(codes).size,
            "effective_n": w.sum() ** 2 / np.dot(w, w),
            "source_x_hash": digest(bank.x),
            "source_y_hash": digest(bank.y),
            "weight_hash": digest(w),
            "table_hash": digest(p),
        }
        # These are scoring-only quantities. The fitted table above has no access
        # to region, shortcut identity, query rows, or exact target probabilities.
        for world, x, _, mass, q in bank.worlds:
            estimate = p[vote_pattern(x)]
            oracle = 2 * np.sum(mass * q * (1 - q))
            excess = 2 * np.sum(mass * (estimate - q) ** 2)
            rows.append(
                {
                    **common,
                    "world": world,
                    "brier": oracle + excess,
                    "oracle_brier": oracle,
                    "test_x_hash": digest(x),
                    "target_hash": digest(np.column_stack((mass, q))),
                }
            )
    return rows


def summarize(table, native, r):
    key = ["family", "seed", "world", "method"]
    hashes = [
        "source_x_hash",
        "source_y_hash",
        "test_x_hash",
        "target_hash",
        "weight_hash",
    ]
    ref = native.set_index(key)[hashes].sort_index()
    same = table.assign(method=table.method.str.removeprefix("table_")).set_index(key)[
        hashes
    ]
    if not same.sort_index().equals(ref):
        raise ValueError("Source, weights, reports or targets differ from native runs")
    keep = ["family", "seed", "world", "method", "brier", "oracle_brier"]
    rows = pd.concat([table[keep], native[keep]], ignore_index=True)
    panels = rows.pivot(
        index=["family", "seed", "world"], columns="method", values="brier"
    )
    if panels.isna().any().any():
        raise ValueError("Native and table predictions must score the same banks")
    oracle = rows.groupby(["family", "seed", "world"]).oracle_brier
    if (oracle.max() - oracle.min()).abs().max() > 1e-12:
        raise ValueError("Exact target risks differ")
    endpoints = panels[panels.index.get_level_values("world").isin(("s0", "s100"))]
    excess = endpoints.sub(oracle.first().reindex(endpoints.index), axis=0)
    metric = excess.groupby(level=["family", "seed"]).max()
    out = []
    for pop, values in [
        ("max_regret", metric),
        *(
            (world, part.droplevel("world"))
            for world, part in panels.groupby(level="world")
        ),
    ]:
        for family, paired in values.groupby(level="family"):
            n = len(paired)
            resamples = np.random.default_rng(103000 + r).integers(n, size=(10_000, n))
            for method, ref in PAIRS:
                change = (paired[method] - paired[ref]).to_numpy()
                lo, hi = np.quantile(change[resamples].mean(axis=1), [0.025, 0.975])
                out.append(
                    {
                        "round": r,
                        "family": family,
                        "endpoint": pop,
                        "source_banks": n,
                        "method": method,
                        "reference": ref,
                        "risk": paired[method].mean(),
                        "reference_risk": paired[ref].mean(),
                        "delta": change.mean(),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return pd.DataFrame(out)


def run(r):
    scores = pd.read_csv(OUT / f"r{r:02}.csv", float_precision="round_trip")
    native = scores[
        (scores.cohort == "focus") & scores.method.isin(("native", "full", "cap2"))
    ]
    raw = pd.DataFrame(
        row for bank in _focus(r, sorted(native.seed.unique())) for row in measure(bank)
    )
    raw.insert(0, "round", r)
    summary = summarize(raw, native, r)
    raw.to_csv(OUT / f"r{r:02}-support-scores.csv", index=False)
    summary.to_csv(OUT / f"r{r:02}-support-contrasts.csv", index=False)


if __name__ == "__main__":
    opts = argparse.ArgumentParser(description=__doc__)
    opts.add_argument("round", type=int, choices=range(1, 11))
    run(opts.parse_args().round)
