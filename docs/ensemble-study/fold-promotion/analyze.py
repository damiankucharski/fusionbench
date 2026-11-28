"""Paired risk contrasts and a decomposition of decisions relative to the fit-only anchor."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
SEEDS = tuple(range(287200, 287232))
DESIGNS = {(0, 24, 8), (1, 24, 8), (1, 192, 8)}
METHODS = (
    "pooled_final",
    "pooled_pooled",
    "pooled_fold",
    "fold_pooled",
    "fold_fold",
    "anchor",
    "mean",
    "top_n",
    "prob_lr",
    "regularized",
)
DESIGN = ["law", "scene_error_share", "n_scenes", "rows_per_scene"]
COMPARISONS = [
    ("fold_fold", "pooled_pooled"),
    ("fold_fold", "pooled_final"),
    ("fold_fold", "pooled_fold"),
    ("pooled_fold", "pooled_pooled"),
    ("fold_pooled", "pooled_pooled"),
    ("fold_fold", "fold_pooled"),
    ("pooled_pooled", "pooled_final"),
] + [
    (policy, ref)
    for policy in ("pooled_final", "pooled_pooled", "fold_fold")
    for ref in ("anchor", "mean", "top_n", "prob_lr", "regularized")
]


def interval(values, draws):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite metric in bank-level study")
    lo, hi = np.quantile(x[draws].mean(axis=1), [0.025, 0.975])
    return {"estimate": x.mean(), "ci_low": lo, "ci_high": hi}


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    pairs, levels, changes = [], [], []
    draws = np.random.default_rng(97825).integers(len(SEEDS), size=(10000, len(SEEDS)))
    actual = scores[DESIGN].drop_duplicates()
    expected = {(law, *design) for law in ("equal", "signal") for design in DESIGNS}
    if set(actual.itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected all six paired fit conditions")
    for keys, group in scores.groupby(DESIGN):
        cell = dict(zip(DESIGN, keys, strict=True))
        ordered = group.set_index(["seed", "method"]).sort_index()
        expected_index = pd.MultiIndex.from_product(
            [SEEDS, sorted(METHODS)], names=["seed", "method"]
        )
        if not ordered.index.equals(expected_index):
            raise ValueError("Incomplete or duplicated bank-level results")
        for metric in ("fit_brier", "audit_brier", "audit_excess", "n_providers"):
            wide = ordered[metric].unstack()
            for method in METHODS:
                levels.append(
                    {
                        **cell,
                        "metric": metric,
                        "method": method,
                        **interval(wide[method], draws),
                    }
                )
            for method, ref in COMPARISONS:
                diff = (wide[method] - wide[ref]).to_numpy()
                pairs.append(
                    {
                        **cell,
                        "metric": metric,
                        "method": method,
                        "reference": ref,
                        **interval(diff, draws),
                        "n_improved": (diff < -1e-12).sum(),
                        "n_tied": (np.abs(diff) <= 1e-12).sum(),
                        "n_worse": (diff > 1e-12).sum(),
                    }
                )
        risk = ordered.audit_excess.unstack()
        signal = ordered.selected_is_signal.unstack()
        if cell["law"] == "signal" and signal.isna().any().any():
            raise ValueError("Missing specialist-equivalence decisions")
        for method in METHODS[:5]:
            risk_diff = risk[method] - risk.anchor
            same_program = ordered.xs(method, level="method").selected_digest.eq(
                ordered.xs("anchor", level="method").selected_digest
            )
            policy = ordered.xs(method, level="method")
            discordant = (policy.robust_gain > 1e-12) & (policy.pooled_gain < -1e-12)
            event = {
                "positive_fold_negative_pooled": int(discordant.sum()),
                "positive_fold_negative_pooled_worse_audit": int(
                    (discordant & (risk_diff > 1e-12)).sum()
                ),
                **cell,
                "method": method,
                "n_banks": len(risk),
                "same_anchor_program": same_program.sum(),
                "better_than_anchor_on_audit": (risk_diff < -1e-12).sum(),
                "worse_than_anchor_on_audit": (risk_diff > 1e-12).sum(),
                "average_excess_change": risk_diff.mean(),
            }
            if cell["law"] == "signal":
                correct = signal.anchor.eq(True)
                chosen = signal[method].eq(True)
                event.update(
                    anchor_exact=correct.sum(),
                    chosen_exact=chosen.sum(),
                    displaced_exact=(correct & ~chosen).sum(),
                    recovered_exact=(~correct & chosen).sum(),
                    mean_cost_from_exact_starts=risk_diff.where(correct, 0).mean(),
                    mean_change_from_inexact_starts=risk_diff.where(~correct, 0).mean(),
                )
            changes.append(event)
    for name, data in (
        ("contrasts", pairs),
        ("levels", levels),
        ("anchor-decisions", changes),
    ):
        pd.DataFrame(data).to_csv(OUT / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
