"""Check identity, provider retention and score changes for the fitted calibration grids."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
BASE = {
    "cal_pull": "pull_mean",
    "raw_pull": "pull_mean",
    "cal_sparse": "sparse_gp",
    "cal_mean": "mean",
}
KEY = ["cohort", "family", "seed"]


def summarize(r):
    tab = pd.read_csv(OUT / f"r{r}-scores.csv", float_precision="round_trip")
    rows = []
    for name, base in BASE.items():
        f = tab[tab.method == name].set_index(KEY, verify_integrity=True).sort_index()
        if f.empty:
            continue
        g = tab[tab.method == base].set_index(KEY, verify_integrity=True).sort_index()
        assert f.index.equals(g.index)
        assert np.array_equal(f.providers, g.providers)
        changed = f.beta != 1
        for column in ("fit_hash", "test_hash"):
            assert (f.loc[~changed, column] == g.loc[~changed, column]).all()
        if base != "mean":
            assert np.array_equal(f.nodes, g.nodes + changed.astype(int))
        data = f[["beta", "ap", "macro_f1"]].copy()
        data["delta_brier"] = f.brier - g.brier
        data["delta_expected_brier"] = f.expected_brier - g.expected_brier
        data["delta_fit_brier"] = f.fit_brier - g.fit_brier
        data["delta_ap"] = f.ap - g.ap
        data["delta_f1"] = f.macro_f1 - g.macro_f1
        data = data.reset_index()
        for cohort, group in data.groupby("cohort"):
            parts = [(cohort, group)] if cohort in {"standard", "factorial"} else []
            parts.extend(group.groupby("family"))
            for label, chunk in parts:
                rows.append(
                    {
                        "method": name,
                        "base": base,
                        "group": label,
                        "fits": len(chunk),
                        "identity": int((chunk.beta == 1).sum()),
                        "sharpened": int((chunk.beta > 1).sum()),
                        "softened": int((chunk.beta < 1).sum()),
                        "mean_beta": chunk.beta.mean(),
                        "delta_brier": chunk.delta_brier.mean(),
                        "delta_expected_brier": (
                            chunk.delta_expected_brier.mean()
                            if chunk.delta_expected_brier.notna().all()
                            else np.nan
                        ),
                        "delta_fit_brier": chunk.delta_fit_brier.mean(),
                        "ap_available": int(chunk.ap.notna().sum()),
                        "ap_changed": int((chunk.delta_ap.abs() > 1e-12).sum()),
                        "max_absolute_ap_change": chunk.delta_ap.abs().max(),
                        "f1_changed": int((chunk.delta_f1.abs() > 1e-12).sum()),
                        "max_absolute_f1_change": chunk.delta_f1.abs().max(),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=(3, 4))
    r = parser.parse_args().round
    summary = summarize(r)
    if len(summary):
        summary.to_csv(OUT / f"r{r}-temperature.csv", index=False)
