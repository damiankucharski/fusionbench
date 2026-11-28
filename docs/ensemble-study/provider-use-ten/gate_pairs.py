"""Paired direct/routing utility of the planted selector before and after shuffling its reports."""

import gzip
import json

import numpy as np
import pandas as pd


def gate_pairs(trace_file, r):
    rows = []
    with gzip.open(trace_file, "rt") as stream:
        for trace in map(json.loads, stream):
            if trace["cohort"] != "gate_control":
                continue
            ids = trace["selector"]
            value = trace["evidence"][f"{ids}/real"]
            rows.append(
                {
                    "family": trace["family"],
                    "seed": trace["seed"],
                    "direct_gain": value["direct_gain"],
                    "routing_gain": value["routing_gain"],
                    "flagged_selector": float(trace["flags"]["original"] == ids),
                }
            )
    source = pd.DataFrame(rows)
    draws = len(source.seed.unique())
    idx = np.random.default_rng(93000 + r).integers(draws, size=(10_000, draws))
    result = []
    for metric in ("direct_gain", "routing_gain", "flagged_selector"):
        panel = source.pivot(index="seed", columns="family", values=metric).sort_index()
        if panel.isna().any().any():
            raise ValueError("Unpaired selector intervention")
        a, b = panel.gated_original, panel.gated_shuffled
        delta = (a - b).to_numpy()
        lo, hi = np.quantile(delta[idx].mean(1), [0.025, 0.975])
        result.append(
            {
                "metric": metric,
                "paired_sources": len(delta),
                "original": a.mean(),
                "shuffled": b.mean(),
                "original_minus_shuffled": delta.mean(),
                "lo": lo,
                "hi": hi,
                "original_above_002": int((a > 0.02).sum())
                if metric.endswith("gain")
                else None,
                "shuffled_above_002": int((b > 0.02).sum())
                if metric.endswith("gain")
                else None,
            }
        )
    return source, pd.DataFrame(result)
