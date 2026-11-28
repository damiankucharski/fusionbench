"""Summarize the locked controlled replacement endpoints at the bank level."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

DEFAULT = Path(__file__).resolve().parent / "results"


def paired(x):
    x = np.asarray(x, dtype=float)
    delta = x.mean()
    width = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {
        "n": len(x),
        "mean": delta,
        "lower": delta - width,
        "upper": delta + width,
        "lower_risk": int(np.sum(x < -1e-12)),
        "tied": int(np.sum(np.abs(x) <= 1e-12)),
        "higher_risk": int(np.sum(x > 1e-12)),
    }


def summarize(out):
    df = pd.read_csv(out / "search.csv", keep_default_na=False)
    if (
        df.duplicated(["bank", "law", "policy"]).any()
        or df.shape[0] != 48 * 2 * 4
        or set(df.law) != {"signal", "null"}
    ):
        raise ValueError("Incomplete search panel")
    group = df.groupby(["law", "policy"])
    means = group[
        [
            "exact_brier",
            "exact_excess",
            "test_brier",
            "exact_accuracy",
            "test_accuracy",
            "initial_excess",
            "deploy_ops",
            "n_providers",
            "best_offered_excess",
            "unique_functions",
            "unique_genotypes",
            "discovery_evaluations",
            "proposal_count",
            "n_promotions",
            "n_true_improving_moves",
            "n_adverse_moves",
        ]
    ].mean()
    means.to_csv(out / "means.csv")
    counts = group[
        [
            "initial_oracle",
            "offer_oracle",
            "ever_oracle",
            "ever_oracle_incumbent",
            "promoted_oracle",
            "final_oracle",
            "offered_better_than_mean",
            "final_better_than_mean",
            "ever_harmful_move",
            "ever_nonoptimal_after",
            "final_has_partial_ancestor",
            "offered_exact_via_partial",
            "final_unconfirmed_ancestor",
        ]
    ].sum()
    counts.to_csv(out / "counts.csv")
    signal = df[df.law == "signal"].copy()
    for k in ("ever_below_one_update_bound", "deploy_below_one_update_bound"):
        signal[k] = signal[k].astype(str).str.lower().eq("true")
    signal.groupby("policy")[
        ["ever_below_one_update_bound", "deploy_below_one_update_bound"]
    ].sum().to_csv(out / "one-update.csv")
    contrasts = []
    for law, g in df.groupby("law"):
        for metric in ("exact_excess", "test_brier", "best_offered_excess"):
            frame = g.pivot(index="bank", columns="policy", values=metric)
            for a, b in [
                ("strict_kl", "open_kl"),
                ("open_kl", "open_raw"),
                ("open_raw", "ordinary"),
            ]:
                contrasts.append(
                    {
                        "law": law,
                        "endpoint": metric,
                        "comparison": f"{a} minus {b}",
                        **paired(frame[a] - frame[b]),
                    }
                )
    pd.DataFrame(contrasts).to_csv(out / "paired.csv", index=False)
    refs = pd.read_csv(out / "baselines.csv.gz", keep_default_na=False)
    refs.groupby(["law", "split", "method"])[
        ["exact_brier", "exact_excess", "test_brier", "exact_accuracy", "test_accuracy"]
    ].mean().to_csv(out / "reference-means.csv")
    power = pd.read_csv(out / "direct-witness.csv", keep_default_na=False)
    power.groupby(["law", "slots", "gate"]).agg(
        n=("bank", "size"),
        accepted=("accept", "sum"),
        mean_bound=("lower_bound", "mean"),
        mean_gain=("empirical_gain", "mean"),
        true_gain=("true_gain", "mean"),
    ).to_csv(out / "power.csv")
    signal_births = defaultdict(list)
    with gzip.open(out / "births.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["law"] == "signal":
                signal_births[r["policy"], r["bank"]].append(r)
    ancestry = []
    for (policy, bank), records in signal_births.items():
        strong = [r for r in records if r["exact_excess"] < 3 / 32 - 1e-12]
        ancestry.append(
            {
                "bank": bank,
                "policy": policy,
                "n_birth_records": len(records),
                "partial_appears": any(r["partial"] for r in records),
                "below_bound_via_partial": any(r["partial_ancestors"] for r in strong),
                "below_bound_via_unconfirmed_parent": any(
                    r["unconfirmed_ancestor"] for r in strong
                ),
                "any_eligible": any(r["confirmed"] for r in records),
            }
        )
    pd.DataFrame(ancestry).to_csv(out / "ancestry.csv", index=False)
    return means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(summarize(args.out))
