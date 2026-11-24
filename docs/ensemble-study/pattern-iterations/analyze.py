"""Compute paired fixed-suite effects; the replicate is the unit of uncertainty."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

KEY = ["replicate", "scenario"]
METHODS = ["native", "challenger", "mean", "top_n", "prob_lr", "logit"]
COMPONENTS = ["flat", "augmented", "gp"]


def interval(sample):
    width = t.ppf(0.975, len(sample) - 1) * sample.sem()
    return sample.mean(), sample.mean() - width, sample.mean() + width


def panels(scores):
    piv = scores.pivot(index=[*KEY, "mode"], columns="method", values="brier")
    by_mode = {m: g.droplevel("mode") for m, g in piv.groupby("mode", sort=False)}
    by_mode["stress"] = (by_mode["provider_swap"] + by_mode["log_jitter"]) / 2
    by_mode["joint"] = (by_mode["clean"] + by_mode["stress"]) / 2
    return by_mode


def summarize(out):
    raw = pd.read_csv(out / "results.csv.gz")
    primary = raw[raw.gp_seed == raw.data_seed]
    per_mode = panels(primary)
    effects, means = [], []
    for mode, frame in per_mode.items():
        repl = frame.groupby("replicate")[METHODS].mean()
        means.append({"panel": mode, "families": len(frame.index.unique("scenario")), **repl.mean()})
        for ref in METHODS:
            if ref == "challenger":
                continue
            delta, lo, hi = interval(repl.challenger - repl[ref])
            fam_delta = (frame.challenger - frame[ref]).groupby("scenario").mean()
            effects.append({"panel": mode, "reference": ref, "delta": delta, "low": lo,
                            "high": hi, "family_wins": int((fam_delta < 0).sum()),
                            "families": len(fam_delta)})
    pd.DataFrame(effects).to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame(means).to_csv(out / "means.csv", index=False)
    per_mode["joint"].groupby("scenario")[METHODS].mean().to_csv(out / "families.csv")

    d = pd.read_csv(out / "diagnostics.csv.gz")
    d = d[d.gp_seed == d.data_seed].copy()
    selected = np.array([COMPONENTS.index(k) for k in d.chosen])
    rows = np.arange(len(d))
    for stage in ("discovery", "refit"):
        d["chosen_" + stage] = d[[stage + "_" + k for k in COMPONENTS]].to_numpy()[rows, selected]
    d["refit_change"] = d["chosen_refit"] - d["chosen_discovery"]
    values = ["chosen_discovery", "chosen_refit", "refit_change", "chosen_discovery_regret",
              "chosen_refit_regret", "chosen_ties_discovery_oracle", "chosen_ties_refit_oracle",
              *["refit_" + k for k in COMPONENTS]]
    d.groupby(["method", "mode"])[values].mean().to_csv(out / "candidate-summary.csv")
    extra = raw[raw.gp_seed != raw.data_seed]
    if not extra.empty:
        a = panels(extra)["joint"]
        b = per_mode["joint"].loc[a.index]
        spread = (a[["native", "challenger"]] - b[["native", "challenger"]]).abs()
        spread.to_csv(out / "search-seed-sensitivity.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
