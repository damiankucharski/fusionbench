"""Summarize which full-data model each out-of-fold decision actually deployed."""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
KEY = ["cohort", "family", "seed"]


def main():
    rows = pd.read_csv(RESULTS / "round4-scores.csv").set_index([*KEY, "method"])
    records = []
    for line in (RESULTS / "round4-fits.jsonl").read_text().splitlines():
        fit = json.loads(line)
        key = tuple(fit[k] for k in KEY)
        meta = fit["variants"]
        simplex = rows.loc[(*key, "simplex")]
        regularized = rows.loc[(*key, "regularized")]
        for rule in ("cf_gain", "cf_margin"):
            chosen = meta["selected_family"][rule]
            selected = rows.loc[(*key, rule)]
            assert (
                selected.test_prediction_hash
                == rows.loc[(*key, chosen)].test_prediction_hash
            )
            records.append(
                {
                    **dict(zip(KEY, key)),
                    "policy": rule,
                    "deployed_model": chosen,
                    "oof_gain": meta["oof_gain"],
                    "oof_se": meta["oof_se"],
                    "test_gain_regularized_over_simplex": simplex.test_brier
                    - regularized.test_brier,
                    "selection_excess_over_better_full_fit": selected.test_brier
                    - min(simplex.test_brier, regularized.test_brier),
                }
            )
    table = pd.DataFrame(records)
    # Gain on held-out rows is retrospective and cannot enter either decision.
    table.to_csv(RESULTS / "round4-decisions.csv", index=False)


if __name__ == "__main__":
    main()
