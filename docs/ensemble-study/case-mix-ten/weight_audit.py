"""Check every recorded guard decision and reconstruct overlap and effective sample size."""

import gzip
import json

import numpy as np
import pandas as pd

SOURCE = ["cohort", "family", "seed"]


def moments(counts, cap):
    n = counts.sum()
    expected = counts.sum(1)[:, None] * counts.sum(0)[None, :] / n
    ratios = expected / counts
    w = ratios if cap is None else np.minimum(ratios, cap)
    w = w / (np.sum(w * counts) / n)
    return n * n / np.sum(counts * w * w), w.min(), w.max()


def audit(raw, source_file):
    with gzip.open(source_file, "rt") as stream:
        traces = {
            (v["cohort"], v["family"], v["seed"]): v for v in map(json.loads, stream)
        }
    recorded = raw.drop_duplicates([*SOURCE, "method"])
    sources = recorded.groupby(SOURCE, sort=False)
    if set(traces) != set(sources.groups):
        raise ValueError("Archived source records do not cover the scoring panel")
    support = []
    checks = 0
    for key, rows in sources:
        t = traces[key]
        n = t["n_fit"]
        tab = rows.set_index("method")
        report = t["full_weights"]
        n_env = len(report["environment_ids"])
        n_ctx = len(report["context_ids"])
        counts = np.zeros((n_env, n_ctx), int)
        for i, j, v in report["cell_counts"]:
            counts[i, j] = v
        weighted = report["reason"] == "weighted"
        support.append(
            {
                "cohort": key[0],
                "family": key[1],
                "seed": key[2],
                "rows": n,
                "n_environments": n_env,
                "n_contexts": n_ctx,
                "smallest_cell": counts.min() if counts.size else 0,
                "missing_cells": report["missing_cells"],
                "eligible_cells": weighted,
                "reason": report["reason"],
            }
        )
        if counts.size and counts.sum() != n:
            raise ValueError("Joint context counts do not cover all source rows")
        if weighted != (
            n_env > 1 and n_ctx > 1 and bool(counts.size) and counts.min() >= 4
        ):
            raise ValueError(
                "Overlap fallback inconsistent with observed training evidence"
            )
        for method, cap in (("full", None), ("cap2", 2.0), ("cap15", 1.5)):
            if method not in tab.index:
                continue
            expected = moments(counts, cap) if weighted else (n, 1.0, 1.0)
            np.testing.assert_allclose(
                tab.loc[method, ["ess", "weight_min", "weight_max"]].astype(float),
                expected,
                rtol=2e-14,
                atol=1e-12,
            )
        fallback = (
            "full"
            if tab.at["full", "ess"] >= 0.65 * n
            else "cap2"
            if tab.at["cap2", "ess"] >= 0.65 * n
            else "native"
        )
        if tab.at["guarded", "policy"] != fallback:
            raise ValueError("Guarded rule chose an unregistered weight mode")
        fields = [
            "stage",
            "reason",
            "weight_hash",
            "fit_hash",
            "test_hash",
            "path_hash",
            "tree_hash",
            "chosen",
        ]
        if not tab.loc["guarded", fields].equals(tab.loc[fallback, fields]):
            raise ValueError("Guarded deployment differs from the selected source fit")
        for method, meta in t["methods"].items():
            for col in [
                "weight_hash",
                "tree_hash",
                "path_hash",
                "chosen",
                "fit_hash",
                "ess",
                "providers",
                "stage",
            ]:
                if meta[col] != tab.at[method, col]:
                    raise ValueError(
                        f"Source and score records differ on {method}/{col}"
                    )
            checks += 1
        models = rows[~rows.method.isin(("mean", "top_n", "lr"))]
        same = models.groupby(["stage", "weight_hash"], dropna=False)
        if (
            not same[["fit_hash", "test_hash", "path_hash", "tree_hash", "chosen"]]
            .nunique()
            .eq(1)
            .all()
            .all()
        ):
            raise ValueError(
                "Identical native configurations do not share fitted forecasts"
            )
        for method in ("selection_only", "refit_only"):
            if (
                method in tab.index
                and tab.at[method, "path_hash"] != tab.at["native", "path_hash"]
            ):
                raise ValueError("Post-discovery weighting changed discovery")
        if "permuted_weights" in tab.index:
            cols = ["ess", "weight_min", "weight_max"]
            np.testing.assert_allclose(
                tab.loc["permuted_weights", cols].astype(float),
                tab.loc["full", cols].astype(float),
                rtol=2e-14,
                atol=1e-12,
            )
    return pd.DataFrame(support), {
        "guard_decisions_verified": len(sources),
        "source_weight_records_verified": checks,
    }
