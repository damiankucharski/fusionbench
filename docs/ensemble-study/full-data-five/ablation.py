"""Describe the three matched controls and which deployed trees reproduce the mean."""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from okapi.paired_fitness import arithmetic_mean_tree

OUT = Path(__file__).resolve().parent / "results"
PAIRS = {
    2: (("seeded_mean", "sham"), ("sham", "full")),
    3: (("pull_mean", "pull_zero"), ("pull_zero", "seeded_mean")),
    4: (("paired_se", "paired_zero"), ("paired_zero", "seeded_mean")),
}


@cache
def mean_key(n):
    # The OKAPI file adapter reads m0.npy ... m(n-1).npy in lexical order.
    ids = sorted(f"m{i}.npy" for i in range(n))
    return repr(arithmetic_mean_tree(dict.fromkeys(ids, np.array([[.5]]))).structural_key())


def main():
    controls, decisions = [], []
    for r, pairs in PAIRS.items():
        scores = pd.read_csv(OUT / f"round{r}-scores.csv")
        scores["group"] = scores.family.where(scores.cohort != "standard", "standard")
        methods = {entry for pair in pairs for entry in pair}
        fits = [json.loads(line) for line in (OUT / f"round{r}-fits.jsonl").read_text().splitlines()]
        keys = {
            (record["cohort"], record["family"], record["seed"], method): fitted["program"] == mean_key(record["n_providers"])
            for record in fits
            for method, fitted in record["fits"].items() if method in methods
        }
        scores = scores[scores.method.isin(methods)].copy()
        index = ["cohort", "family", "seed", "method"]
        scores["seed_structure"] = [keys[key] for key in scores[index].itertuples(index=False, name=None)]
        assert len(keys) == len(scores)
        tables = pd.read_csv(OUT / f"round{r}-contrasts.csv")
        for candidate, control in pairs:
            paired = tables[(tables.metric == "brier") & (tables.method == candidate) & (tables.reference == control)]
            paired = paired.copy()
            paired.insert(0, "round", r)
            controls.append(paired)
        for (group, method), g in scores.groupby(["group", "method"]):
            decisions.append({
                "round": r,
                "group": group,
                "method": method,
                "banks": len(g),
                "exact_seed_structure": int(g.seed_structure.sum()),
                "equals_mean_fit": int(g.equals_mean_fit.sum()),
                "equals_mean_audit": int(g.equals_mean_audit.sum()),
                "equals_mean_both": int((g.equals_mean_fit & g.equals_mean_audit).sum()),
                "mean_nodes": g.nodes.mean(),
                "mean_providers_used": g.providers_used.mean(),
            })
    pd.concat(controls, ignore_index=True).to_csv(OUT / "matched-ablations.csv", index=False)
    pd.DataFrame(decisions).to_csv(OUT / "mean-deployment.csv", index=False)


if __name__ == "__main__":
    main()
