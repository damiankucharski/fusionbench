"""Measured deployment sizes, CV fit cost, and observed binary AP on eligible banks."""

import argparse
import json

import pandas as pd

from summarize_auto import KEYS, OUT, slices


def summarize(r):
    scores = pd.read_csv(OUT / f"r{r}-scores.csv", float_precision="round_trip")
    with (OUT / f"r{r}-decisions.jsonl").open() as traces:
        times = pd.DataFrame(
            {
                "family": bank["family"],
                "seed": bank["seed"],
                "tree_lookup_seconds": bank.get("tree_seconds"),
                "cv_fit_seconds": bank.get("cv", {}).get("fit_seconds"),
                "cv_oof_fits": bank.get("cv", {}).get("oof_fits"),
            }
            for bank in map(json.loads, traces)
        )
    records = []
    for group, sample in slices(scores):
        matched = sample.merge(times, on=KEYS, validate="many_to_one")
        for method, rows in matched.groupby("method"):
            binary = rows[rows.n_classes == 2]
            record = {
                "group": group,
                "method": method,
                "banks": len(rows),
                "mean_providers": rows.providers.mean(),
                "banks_returning_tree": rows.nodes.notna().sum(),
                "mean_nodes_if_tree": rows.nodes.mean(),
                "binary_banks": len(binary),
                "binary_macro_ap": binary.ap.mean(),
            }
            if method == "tree_rule":
                record["mean_lookup_seconds_excluding_fingerprint"] = (
                    rows.tree_lookup_seconds.mean()
                )
            if method == "cv_rule":
                record["mean_cv_seconds_excluding_refit"] = rows.cv_fit_seconds.mean()
                record["mean_extra_fits"] = rows.cv_oof_fits.mean()
            records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=(1, 2, 3))
    r = parser.parse_args().round
    summarize(r).to_csv(OUT / f"r{r}-configuration-costs.csv", index=False)
