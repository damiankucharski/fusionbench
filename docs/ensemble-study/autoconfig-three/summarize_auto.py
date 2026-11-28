"""Paired comparisons and action-matching controls for native configuration rules."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
HELDOUT = {
    "grouped_rare_event",
    "shared_error_bloc",
    "rare_scene_specialist",
    "cloned_shortcut",
    "parity",
    "similar_classes_38",
}
NEW = {"grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"}
PORTFOLIO = ("mean", "top_n", "cal_mean", "flat", "cal_pull")
ROUTERS = ("tree_rule", "cv_rule", "constant", "portfolio_oracle")
METRICS = (
    "brier",
    "expected_brier",
    "fit_brier",
    "macro_f1",
    "ap",
    "providers",
    "nodes",
)
KEYS = ["family", "seed"]


def draw_weights(families, n, r):
    """Keep each scenario equally weighted and every candidate paired."""
    ids = np.random.default_rng(73000 + r).integers(n, size=(10_000, families, n))
    counts = np.stack([(ids == i).sum(axis=2) for i in range(n)], axis=2)
    return counts.reshape(10_000, families * n) / (families * n)


def slices(scores):
    standard = scores[scores.cohort == "standard"]
    names = set(standard.family)
    if len(names) != 30 or not HELDOUT <= names:
        raise ValueError("Expected all 30 standard-v6 scenarios")
    yield "standard", standard
    yield "legacy27", standard[~standard.family.isin(NEW)]
    yield "trained24", standard[~standard.family.isin(HELDOUT)]
    yield "heldout6", standard[standard.family.isin(HELDOUT)]
    fac = scores[scores.cohort == "factorial"]
    if not fac.groupby("seed").family.nunique().eq(8).all():
        raise ValueError("Every factorial seed must have all eight settings")
    yield "factorial", fac
    yield from scores[scores.cohort != "standard"].groupby("family")


def bank_matrix(tab, group):
    """Collapse all eight paired worlds into a single draw only for factorial macro."""
    if group == "factorial":
        return tab.groupby(level="seed").mean().sort_index()
    return tab.sort_index()


def action_stats(s, group, weights):
    """Conditional expected test loss after uniformly permuting the chosen actions.

    Randomize observed choices within the displayed population. Resample the
    empirical action frequencies and action losses jointly for the interval.
    """
    cols = list(PORTFOLIO)
    panel = s.pivot(index=KEYS, columns="method", values="brier")
    costs = panel[cols]
    summary, freqs = [], []
    for method in ROUTERS:
        if method not in panel:
            continue
        selected = s[s.method == method].set_index(KEYS).loc[costs.index, "action"]
        if not selected.isin(cols).all():
            raise ValueError(f"Unknown route for {method}")
        actions = pd.get_dummies(selected, dtype=float).reindex(
            columns=cols, fill_value=0
        )
        loss = np.sum(costs.to_numpy() * actions.to_numpy(), axis=1)
        if not np.allclose(loss, panel[method], atol=2e-12, rtol=0):
            raise ValueError(
                f"Recorded route differs from deployed forecast for {method}"
            )
        x = bank_matrix(costs, group).to_numpy()
        a = bank_matrix(actions, group).to_numpy()
        y = bank_matrix(panel[[method]], group).to_numpy()[:, 0]
        random_loss = float(a.mean(0) @ x.mean(0))
        bootstrap_delta = weights @ y - np.sum((weights @ a) * (weights @ x), axis=1)
        lo, hi = np.quantile(bootstrap_delta, [0.025, 0.975])
        # Also hold the action proportions within each named scenario fixed. A
        # difference accrued purely between scenarios then disappears.
        within = np.mean(
            [
                actions.xs(family).mean().to_numpy()
                @ costs.xs(family).mean().to_numpy()
                for family in costs.index.get_level_values("family").unique()
            ]
        )
        summary.append(
            {
                "group": group,
                "method": method,
                "actual": y.mean(),
                "expected_random": random_loss,
                "delta_actual_minus_random": y.mean() - random_loss,
                "lo": lo,
                "hi": hi,
                "expected_within_family_random": within,
                "delta_actual_minus_within_family_random": y.mean() - within,
            }
        )
        for i, action in enumerate(cols):
            freqs.append(
                {
                    "group": group,
                    "method": method,
                    "action": action,
                    "count": int(actions.iloc[:, i].sum()),
                    "fraction": a[:, i].mean(),
                }
            )
    return summary, freqs


def summarize(scores, r):
    if (
        set(scores["round"]) != {r}
        or scores.duplicated(["cohort", *KEYS, "method"]).any()
    ):
        raise ValueError("Expected one row per method and bank for the requested round")
    rows, diffs, frequencies, mix = [], [], [], []
    for group, s in slices(scores):
        sizes = s.groupby("family").seed.nunique()
        if sizes.nunique() != 1:
            raise ValueError(f"Unbalanced data within {group}")
        n_family, n = len(sizes), int(sizes.iloc[0])
        weights = draw_weights(1 if group == "factorial" else n_family, n, r)
        records = {
            m: {
                "group": group,
                "method": m,
                "families": n_family,
                "data_seeds_per_family": n,
            }
            for m in sorted(s.method.unique())
        }
        for metric in METRICS:
            if metric not in s:
                continue
            tab = s.pivot(index=KEYS, columns="method", values=metric).sort_index()
            tab = tab.dropna(axis=1)
            matrix = bank_matrix(tab, group)
            if not np.isfinite(matrix.to_numpy()).all():
                raise ValueError(f"Invalid scores for {group} {metric}")
            for m in matrix:
                records[m][metric] = matrix[m].mean()
            for ref in (
                "mean",
                "top_n",
                "lr",
                "cal_pull",
                "constant",
                "portfolio_oracle",
                "cv_rule",
                "tree_rule",
            ):
                if ref not in matrix:
                    continue
                delta = matrix.subtract(matrix[ref], axis=0)
                means = delta.mean().to_numpy()
                lo, hi = np.quantile(weights @ delta.to_numpy(), [0.025, 0.975], axis=0)
                for i, m in enumerate(delta):
                    if m == ref:
                        continue
                    diffs.append(
                        {
                            "group": group,
                            "metric": metric,
                            "method": m,
                            "reference": ref,
                            "delta": means[i],
                            "lo": lo[i],
                            "hi": hi[i],
                            "resampling_units": len(delta),
                        }
                    )
        rows.extend(records.values())
        if any(m in records for m in ROUTERS):
            exchange, counts = action_stats(s, group, weights)
            mix.extend(exchange)
            frequencies.extend(counts)
    standard = scores[scores.cohort == "standard"]
    stats = standard.groupby(["family", "method"])[
        [m for m in METRICS if m in standard]
    ].mean()
    for ref in ("cal_pull", "constant"):
        if ref in stats.index.get_level_values("method"):
            stats["delta_" + ref] = stats.brier.sub(
                stats.xs(ref, level="method").brier, level="family"
            )
    return {
        "levels": pd.DataFrame(rows),
        "contrasts": pd.DataFrame(diffs),
        "families": stats.reset_index(),
        "action-mix": pd.DataFrame(mix),
        "actions": pd.DataFrame(frequencies),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=(1, 2, 3))
    r = parser.parse_args().round
    scores = pd.read_csv(OUT / f"r{r}-scores.csv", float_precision="round_trip")
    with threadpool_limits(limits=1):
        for name, data in summarize(scores, r).items():
            if len(data):
                data.to_csv(OUT / f"r{r}-{name}.csv", index=False)


if __name__ == "__main__":
    main()
