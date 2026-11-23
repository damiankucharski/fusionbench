"""Recompute every ranked prefix and check the published held-out Top-N probabilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.full_suite_point import data_for, fingerprint
from fusionbench.methods.averaging import TopN

OUT = Path(__file__).resolve().parent


def main():
    scores = pd.read_csv(OUT / "scores.csv.gz")
    fixed = scores[scores.method.isin(["top_n", "top_n_640"])]
    records = []
    for (name, bank), group in fixed.groupby(["scenario", "data_seed"]):
        scenario, splits = data_for(name, bank)
        classes = scenario.n_classes
        for method, selected in group.groupby("method"):
            training = ["fit", "validation"] if method.endswith("_640") else ["fit"]
            x = np.concatenate([splits[k][0] for k in training])
            y = np.concatenate([splits[k][1] for k in training])
            fitted = TopN("brier").fit(
                x, y, task="binary" if classes == 2 else "multiclass", n_classes=classes
            )
            assert len(fitted.prefix_scores_) == x.shape[1]
            assert fitted.n_selected_ == 1 + np.argmin(fitted.prefix_scores_)
            for row in selected.itertuples():
                p = fitted.predict(splits[row.split][0])
                assert fingerprint(p) == row.prediction_sha256
            records.append(
                {
                    "scenario": name,
                    "data_seed": int(bank),
                    "method": method,
                    "chosen": int(fitted.n_selected_),
                    "ranking": fitted.ranking_.tolist(),
                    "prefix_brier": fitted.prefix_scores_.tolist(),
                }
            )
    with (OUT / "topn-prefixes.jsonl").open("w") as output:
        for record in records:
            print(json.dumps(record, separators=(",", ":")), file=output)
    print(
        f"{len(records)} rankings verified against {len(fixed)} prediction fingerprints"
    )


if __name__ == "__main__":
    main()
