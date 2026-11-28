"""Choose one fitted native tree by stability over four predictions on source inputs."""

from __future__ import annotations

import hashlib

import numpy as np

from fusionbench.eval.suite_point_gp import _full, _targets
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

BOOTSTRAP_SEEDS = (1061, 1062, 1063)


def stratified_bootstrap(y, seed):
    """Bootstrap within each class and preserve the original sequence of labels."""
    rng = np.random.default_rng(seed)
    rows = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        indices = np.flatnonzero(y == c)
        rows[indices] = rng.choice(indices, size=len(indices), replace=True)
    return rows


def select_medoid(forecasts):
    """Rank by mean squared full-class distance to the mean prediction on source rows."""
    p = np.asarray(forecasts, dtype=float)
    if (
        p.ndim != 3
        or p.shape[0] != 4
        or p.shape[1] < 1
        or p.shape[2] < 2
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "Expected finite forecasts for four models on a common [rows,classes] bank"
        )
    center = p.mean(axis=0)
    distances = np.square(p - center).sum(axis=2).mean(axis=1)
    # Arrays are ordered full-fit first. argmin preserves that ordering on exact ties.
    return int(np.argmin(distances)), distances


class BootstrapAverage:
    """Four-model arithmetic-average diagnostic."""

    def __init__(self, members):
        self.members = tuple(members)

    def predict(self, preds):
        return np.mean([member.predict(preds) for member in self.members], axis=0)


def fit_variants(x, y):
    """Return {regularized, bootstrap_medoid, bootstrap_average} fitted models and metadata.

    A single selected native estimator serves all predictions of the medoid arm.
    The average diagnostic evaluates all four separately fitted estimators.
    """
    x = _full(x)
    n, _, classes = x.shape
    y, _ = _targets(y, n, classes)
    task = "binary" if classes == 2 else "multiclass"
    members = []
    forecasts = []
    unique_counts = []
    row_digests = []
    validation_repeats = []
    for optimizer_seed, bootstrap_seed in enumerate((None, *BOOTSTRAP_SEEDS)):
        rows = (
            np.arange(n, dtype=np.int64)
            if bootstrap_seed is None
            else stratified_bootstrap(y, bootstrap_seed)
        )
        model = RegularizedNativePatternOKAPI(seed=optimizer_seed, split_seed=0)
        model.fit(x[rows], y[rows], task=task, n_classes=classes)
        members.append(model)
        forecasts.append(model.predict(x))
        unique_counts.append(int(np.unique(rows).size))
        # Count validation copies with an original row also in discovery.
        # This describes bootstrap overlap after fitting and never enters selection.
        validation = model.validation_indices_
        discovery = np.ones(n, dtype=bool)
        discovery[validation] = False
        validation_repeats.append(int(np.isin(rows[validation], rows[discovery]).sum()))
        row_digests.append(
            hashlib.sha256(rows.astype("<i8", copy=False).tobytes()).hexdigest()
        )
    chosen, distances = select_medoid(forecasts)
    info = {
        "selected_index": chosen,
        "source_distances": distances.tolist(),
        "bootstrap_seeds": [None, *BOOTSTRAP_SEEDS],
        "optimizer_seeds": list(range(4)),
        "split_seed": 0,
        "n_unique_fit_rows": unique_counts,
        "validation_rows": len(members[0].validation_indices_),
        "validation_rows_repeated_in_discovery": validation_repeats,
        "fit_index_sha256": row_digests,
        "families": [model.chosen_ for model in members],
        "score_calls": [int(model.counts_["total_score_calls"]) for model in members],
        "providers_per_member": [
            len(model.trees_[model.chosen_].unique_value_node_ids) for model in members
        ],
    }
    return {
        "regularized": members[0],
        "bootstrap_medoid": members[chosen],
        "bootstrap_average": BootstrapAverage(members),
    }, info


def summarize(folder):
    """Describe selected members, training cost, and risk relative to averaging all four."""
    import json
    from pathlib import Path

    import pandas as pd

    folder = Path(folder)
    scores = pd.read_csv(folder / "round3-scores.csv")
    with (folder / "round3-fits.jsonl").open() as stream:
        fits = [json.loads(line) for line in stream]
    if len(fits) != 106:
        raise ValueError("Round 3 requires 106 fully evaluated banks")
    decisions = []
    for case in fits:
        v = case["variants"]
        chosen = int(np.argmin(v["source_distances"]))
        if v["selected_index"] != chosen:
            raise ValueError(
                "Archived selection differs from the minimum source distance"
            )
        decisions.append(
            {
                **{k: case[k] for k in ("cohort", "family", "seed")},
                "selected_index": chosen,
                "selected_family": v["families"][chosen],
                "runner_up_distance_margin": float(
                    np.sort(v["source_distances"])[1] - v["source_distances"][chosen]
                ),
                "source_distance": v["source_distances"][chosen],
                "selected_providers": v["providers_per_member"][chosen],
                "fit_objective_calls": sum(v["score_calls"]),
                "single_full_fit_objective_calls": v["score_calls"][0],
                "bootstrap_validation_positions": 3 * v["validation_rows"],
                "bootstrap_validation_copies_seen_in_discovery": sum(
                    v["validation_rows_repeated_in_discovery"][1:]
                ),
            }
        )
    selected = pd.DataFrame(decisions)
    keys = ["cohort", "family", "seed"]
    wide = scores.pivot(index=keys, columns="method", values="test_brier")
    data = selected.set_index(keys).join(wide, validate="one_to_one").sort_index()
    if (
        data[["bootstrap_medoid", "bootstrap_average", "regularized"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Incomplete member decisions or scores")
    selected.to_csv(folder / "round3-decisions.csv", index=False)
    summary = []
    for cohort in ("controlled", "standard", "digits"):
        part = data.xs(cohort, level="cohort")
        groups = (
            {"standard": part}
            if cohort == "standard"
            else dict(tuple(part.groupby(level="family")))
        )
        for label, rows in groups.items():
            families = rows.index.get_level_values("family").nunique()
            n = len(rows) // families
            v = (
                (rows.bootstrap_medoid - rows.bootstrap_average)
                .to_numpy()
                .reshape(families, n)
            )
            if cohort == "digits":
                lo = hi = None
            else:
                draws = np.random.default_rng(89003).integers(
                    n, size=(10000, families, n)
                )
                lo, hi = np.quantile(
                    v[np.arange(families)[None, :, None], draws].mean(axis=(1, 2)),
                    [0.025, 0.975],
                )
            summary.append(
                {
                    "group": label,
                    "banks": len(rows),
                    "selected_full_fit": int((rows.selected_index == 0).sum()),
                    "selected_bootstrap": int((rows.selected_index != 0).sum()),
                    "medoid_minus_average_brier": v.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "medoid_better_than_average_banks": int((v < -1e-12).sum()),
                    "medoid_worse_than_average_banks": int((v > 1e-12).sum()),
                    "fit_objective_calls": int(rows.fit_objective_calls.sum()),
                    "single_full_fit_objective_calls": int(
                        rows.single_full_fit_objective_calls.sum()
                    ),
                    "bootstrap_validation_positions": int(
                        rows.bootstrap_validation_positions.sum()
                    ),
                    "bootstrap_validation_copies_seen_in_discovery": int(
                        rows.bootstrap_validation_copies_seen_in_discovery.sum()
                    ),
                }
            )
    pd.DataFrame(summary).to_csv(folder / "round3-stability.csv", index=False)


if __name__ == "__main__":
    from pathlib import Path

    summarize(Path(__file__).with_name("results"))
