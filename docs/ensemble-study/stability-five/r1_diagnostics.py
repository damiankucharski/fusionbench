"""Separate validation loss of the selected tree from its pre-refit test loss."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
KEY = ["cohort", "family", "seed"]
NAMES = ("regularized", "gp_top1", "gp_top8")


def main():
    scores = pd.read_csv(OUT / "round1-scores.csv").set_index([*KEY, "method"])
    records = [
        json.loads(line)
        for line in (OUT / "round1-fits.jsonl").read_text().splitlines()
    ]
    rows = []
    for record in records:
        keys = tuple(record[k] for k in KEY)
        info = record["variants"]
        full = info["choices"]["regularized"]
        for name in NAMES:
            choice = info["choices"][name]
            score = scores.loc[(*keys, name)]
            baseline = scores.loc[(*keys, "regularized")]
            assert choice["eligible_gp"] == min(
                {"regularized": info["archive_size"], "gp_top1": 1, "gp_top8": 8}[name],
                info["archive_size"],
            )
            assert choice["gp_discovery_rank"] <= choice["eligible_gp"]
            assert abs(choice["validation_brier"] - score.validation_brier) <= 1e-14
            assert info["gp_calls"] == 600
            rows.append(
                {
                    "group": keys[0] if keys[0] == "standard" else keys[1],
                    "method": name,
                    "archive": info["archive_size"],
                    "eligibility": choice["eligible_gp"],
                    "discovery_rank": choice["gp_discovery_rank"],
                    "same_gp": choice["gp_hash"] == full["gp_hash"],
                    "same_deployment": score.test_prediction_hash
                    == baseline.test_prediction_hash,
                    "selected_family": choice["family"],
                    "validation": score.validation_brier,
                    "pre_refit_audit": score.pre_refit_test,
                    "gap": score.pre_refit_test - score.validation_brier,
                    "gap_delta": (score.pre_refit_test - score.validation_brier)
                    - (baseline.pre_refit_test - baseline.validation_brier),
                }
            )
    frame = pd.DataFrame(rows)
    out = []
    for (group, name), g in frame.groupby(["group", "method"]):
        row = {
            "group": group,
            "method": name,
            "banks": len(g),
            "mean_eligible": g.eligibility.mean(),
            "mean_archive": g.archive.mean(),
            "median_gp_discovery_rank": g.discovery_rank.median(),
            "same_gp_as_reg": int(g.same_gp.sum()),
            "same_final_as_reg": int(g.same_deployment.sum()),
            "mean_validation_brier": g.validation.mean(),
            "mean_pre_refit_audit": g.pre_refit_audit.mean(),
            "mean_validation_to_audit_gap": g.gap.mean(),
            "mean_gap_minus_regularized": g.gap_delta.mean(),
        }
        if group in ("weak", "strong", "equal"):
            draws = np.random.default_rng(89001).integers(0, len(g), (10000, 1, len(g)))
            row["gap_diff_lo"], row["gap_diff_hi"] = np.quantile(
                g.gap_delta.to_numpy()[draws].mean(axis=(1, 2)), [0.025, 0.975]
            )
        row.update(
            {
                f"chose_{f}": int((g.selected_family == f).sum())
                for f in ("gp", "flat", "augmented")
            }
        )
        out.append(row)
    assert len(frame) == 106 * 3
    pd.DataFrame(out).to_csv(OUT / "round1-validation.csv", index=False)


if __name__ == "__main__":
    main()
