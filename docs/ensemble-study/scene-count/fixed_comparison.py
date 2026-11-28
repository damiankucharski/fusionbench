"""Sampling variance of the empirical Brier difference for two fixed predictors."""

from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.scenarios.checkpoint_bank import (
    ROWS_PER_SCENE,
    generate_scene_pool,
    scene_sample_indices,
    select_checkpoints,
    signal_provider_index,
)

OUT = Path(__file__).with_name("results")
DESIGNS = ((24, 8), (192, 1), (192, 8), (1536, 1))


def loss_difference(block, slots, singleton, target):
    frozen = block.P[:, np.arange(len(slots)), slots]
    return 2 * (
        (frozen[:, singleton] - target) ** 2 - (frozen.mean(axis=1) - target) ** 2
    )


def analyze_bank(seed, law):
    bank = generate_scene_pool(seed, law=law)
    slots = select_checkpoints(bank["A"], 8)
    singleton = signal_provider_index(seed)
    fits = loss_difference(bank["F"], slots, singleton, bank["F"].y)
    audit = loss_difference(bank["I"], slots, singleton, bank["I"].posterior).mean()
    by_scene = fits.reshape(-1, ROWS_PER_SCENE)
    common = {"seed": seed, "law": law, "audit_conditional_difference": audit}
    components = {
        **common,
        "within": by_scene.var(axis=1, ddof=1).mean(),
        "between_means": by_scene.mean(axis=1).var(ddof=1),
    }
    margins = []
    for n_scenes, rows in DESIGNS:
        indices = scene_sample_indices(n_scenes, rows)
        margins.append(
            {
                **common,
                "n_scenes": n_scenes,
                "rows_per_scene": rows,
                "fit_difference": fits[indices].mean(),
            }
        )
    return components, margins


def ratio(between, within):
    return ROWS_PER_SCENE * between / (between + (1 - 1 / ROWS_PER_SCENE) * within)


def main():
    components, margins = [], []
    for seed in range(286100, 286132):
        for law in ("equal", "signal"):
            estimates, decisions = analyze_bank(seed, law)
            components.append(estimates)
            margins.extend(decisions)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(components)
    frame.to_csv(OUT / "fixed-variance-components.csv", index=False)
    pd.DataFrame(margins).to_csv(OUT / "fixed-decisions.csv", index=False)
    draws = np.random.default_rng(97823).integers(32, size=(10000, 32))
    summary = []
    for law, rows in frame.groupby("law"):
        within, between = rows.within.to_numpy(), rows.between_means.to_numpy()
        design_effect = ratio(between.mean(), within.mean())
        paired = ratio(between[draws].mean(axis=1), within[draws].mean(axis=1))
        lo, hi = np.quantile(paired, [0.025, 0.975])
        summary.append(
            {
                "law": law,
                "banks": len(rows),
                "variance_ratio": design_effect,
                "ci_low": lo,
                "ci_high": hi,
                "standard_error_ratio": np.sqrt(design_effect),
                "loss_difference_intraclass_correlation": (design_effect - 1)
                / (ROWS_PER_SCENE - 1),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "fixed-variance-summary.csv", index=False)


if __name__ == "__main__":
    main()
