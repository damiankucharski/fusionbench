"""Measure whether a single fit bank reveals known mechanisms in round one."""

from pathlib import Path

import numpy as np
import pandas as pd
from okapi.pool_diagnostics import POOL_FEATURES
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FACTORS = ("h", "c", "s")
# Each directional readout is chosen before seeing any round-one scores.
READOUT = {
    "h": ("quality_spread", +1),
    "c": ("disagreement", -1),
    "s": ("local_complementarity", +1),
}


def pair_effects(focal):
    data = []
    for factor in FACTORS:
        others = [f for f in FACTORS if f != factor]
        for feature in POOL_FEATURES:
            v = focal.pivot(index=["seed", *others], columns=factor, values=feature)
            if len(v.columns) != 2 or v.isna().any().any():
                raise ValueError("Missing paired cell")
            diff = v[1] - v[0]
            paired_by_seed = diff.groupby("seed").mean().to_numpy()
            n = len(paired_by_seed)
            boot = np.random.default_rng(63911).integers(n, size=(10000, n))
            lo, hi = np.quantile(paired_by_seed[boot].mean(1), [0.025, 0.975])
            data.append(
                {
                    "factor": factor,
                    "feature": feature,
                    "high_minus_low": paired_by_seed.mean(),
                    "lo": lo,
                    "hi": hi,
                    "higher_pairs": int((diff > 1e-12).sum()),
                    "lower_pairs": int((diff < -1e-12).sum()),
                    "tied_pairs": int((diff.abs() <= 1e-12).sum()),
                    "matched_pairs": len(diff),
                    "higher_seeds": int((paired_by_seed > 1e-12).sum()),
                    "lower_seeds": int((paired_by_seed < -1e-12).sum()),
                    "tied_seeds": int((np.abs(paired_by_seed) <= 1e-12).sum()),
                    "independent_seeds": n,
                }
            )
    return pd.DataFrame(data)


def threshold(scores, y):
    values = np.unique(scores)
    # Two outer thresholds yield all-positive or all-negative predictions.
    # Within a tie use the first boundary in ascending score order.
    cuts = np.r_[-np.inf, values[:-1] + np.diff(values) / 2, np.inf]
    predictions = scores[:, None] >= cuts
    accuracy = (predictions[y].mean(0) + (~predictions[~y]).mean(0)) / 2
    return cuts[np.argmax(accuracy)]


def held_seed_readout(focal):
    rows = []
    for factor, (feature, sign) in READOUT.items():
        scores = sign * focal[feature].to_numpy()
        labels = focal[factor].to_numpy(dtype=bool)
        seeds = focal.seed.to_numpy()
        predicted = np.zeros(len(labels), dtype=bool)
        for seed in np.unique(seeds):
            held = seeds == seed
            cut = threshold(scores[~held], labels[~held])
            predicted[held] = scores[held] >= cut
        sensitivity = predicted[labels].mean()
        specificity = (~predicted[~labels]).mean()
        rows.append(
            {
                "factor": factor,
                "feature": feature,
                "score_direction": sign,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "descriptive_auc": roc_auc_score(labels, scores),
                "correct": int((predicted == labels).sum()),
                "cell_banks": len(focal),
                "independent_seeds": len(np.unique(seeds)),
            }
        )
    return pd.DataFrame(rows)


def main():
    tab = pd.read_csv(OUT / "r1-fingerprints.csv", float_precision="round_trip")
    focal = tab[tab.cohort == "factorial"].copy()
    if focal.empty or focal.duplicated(["family", "seed"]).any():
        raise ValueError("Expected one source fingerprint per round-one factorial bank")
    pattern = r"^h([01])c([01])s([01])$"
    focal[list(FACTORS)] = focal.family.str.extract(pattern).astype(int)
    if not (focal.groupby("seed").size() == 8).all():
        raise ValueError("Expected eight paired configurations per independent seed")
    pair_effects(focal).to_csv(OUT / "r1-feature-recovery.csv", index=False)
    held_seed_readout(focal).to_csv(OUT / "r1-mechanism-readout.csv", index=False)


if __name__ == "__main__":
    main()
