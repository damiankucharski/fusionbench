"""Resolve the expected Brier difference of a compact pool on the existing round 2 audit."""

import json
from hashlib import sha256
from itertools import takewhile
from pathlib import Path

import numpy as np
import pandas as pd
from banks_study import banks
from sparse_gp import fit_sparse_pool
from stats_four import weights

OUT = Path(__file__).resolve().parent / "results"


def geometry(bank, saved):
    p = bank.audit_x[:, :, 1]
    center = p.mean(1)
    q = bank.q
    m = p.shape[1]
    dispersion = 2 * np.square(p - center[:, None]).mean()
    key = (bank.context["family"], bank.context["seed"])
    rows = []
    for name in ("sparse_pool", "random_pool"):
        model = fit_sparse_pool(bank.x, random_subset=name == "random_pool")
        saved_row = saved.loc[(*key, name)]
        assert list(model.selected_indices_) == json.loads(saved_row.subset_seed)
        pred = np.ascontiguousarray(model.predict(bank.audit_x), dtype=np.float64)
        assert sha256(pred.tobytes()).hexdigest() == saved_row.test_hash
        d = pred[:, 1] - center
        distance = 2 * np.square(d).mean()
        alignment = 4 * (d * (center - q)).mean()
        k = len(model.selected_indices_)
        sum_terms = distance + alignment
        np.testing.assert_allclose(
            sum_terms, saved_row.expected_brier - saved.loc[(*key, "mean")].expected_brier, rtol=0, atol=2e-12
        )
        # Uniform k-without-replacement gives E[d]=0 on each audit row.
        random_cost = dispersion * (m - k) / (k * (m - 1))
        rows.append({
            "family": key[0], "seed": key[1], "method": name, "distance": distance,
            "alignment": alignment, "delta_mean": sum_terms, "uniform_subset_delta_mean": random_cost,
            "delta_uniform_subset": sum_terms - random_cost,
        })
    return rows


def main():
    scores = pd.read_csv(OUT / "r2-scores.csv", float_precision="round_trip")
    scores = scores.query("cohort=='factorial'").set_index(["family", "seed", "method"], verify_integrity=True)
    data = []
    for b in takewhile(lambda item: item.context["cohort"] == "factorial", banks(2)):
        data.extend(geometry(b, scores))
    table = pd.DataFrame(data)
    assert len(table) == 2 * 64
    table.to_csv(OUT / "r2-sparse-geometry.csv", index=False)
    fields = ["distance", "alignment", "delta_mean", "uniform_subset_delta_mean", "delta_uniform_subset"]
    summaries = []
    for method, group in table.groupby("method"):
        for label, sample in [("factorial", group), *group.groupby("family")]:
            paired = sample.groupby("seed")[fields].mean()
            assert len(paired) == 8
            boot = weights(1, len(paired), 2) @ paired.to_numpy()
            lo, hi = np.quantile(boot, [.025, .975], axis=0)
            for metric, value, lower, upper in zip(fields, paired.mean(), lo, hi, strict=True):
                summaries.append({"group": label, "method": method, "metric": metric, "value": value,
                                  "lo": lower, "hi": upper, "seeds": len(paired)})
    pd.DataFrame(summaries).to_csv(OUT / "r2-sparse-summary.csv", index=False)


if __name__ == "__main__":
    main()
