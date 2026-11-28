"""Measure fixed-law generalization, prediction variance, and paired holdout choices."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from gate import gate_choices

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.stratified_pool import StratifiedPool

OUT = Path(__file__).resolve().parent / "results"
SEEDS = tuple(range(255100, 255132))
DELTAS = (0, 0.25, 0.75)
SIZES = (32, 128, 512)
METHODS = ("mean", "top_n", "prob_lr", "regularized", "plain", "guard")


def bank(seed):
    rows, forecasts = [], []
    for d in DELTAS:
        scenario = StratifiedPool(d)
        support, mass, q = scenario.support()
        full_x, full_y, meta = scenario.sample(512, np.random.default_rng([seed, 1]))
        strata = 2 * meta["high_amplitude"].astype(int) + full_y
        for n in SIZES:
            indices = np.sort(
                np.concatenate(
                    [np.flatnonzero(strata == k)[: n // 4] for k in range(4)]
                )
            )
            x, y = full_x[indices], full_y[indices]
            np.testing.assert_array_equal(
                np.bincount(strata[indices], minlength=4), np.full(4, n // 4)
            )
            methods = {
                "mean": SimpleAverage(),
                "top_n": TopN(metric="brier"),
                "prob_lr": StackingLR(c_value=1.0),
                "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0),
            }
            gate = gate_choices(x, y)
            for method in methods.values():
                method.fit(x, y, task="binary", n_classes=2)
            for name in METHODS:
                model = (
                    methods["regularized" if gate[name] else "mean"]
                    if name in ("plain", "guard")
                    else methods[name]
                )
                phat = model.predict(support)[:, 1]
                empirical = 2 * np.mean((model.predict(x)[:, 1] - y) ** 2)
                excess = 2 * np.dot(mass, (phat - q) ** 2)
                bayes = 2 * np.dot(mass, q * (1 - q))
                rows.append(
                    {
                        "seed": seed,
                        "specialization": d,
                        "size": n,
                        "method": name,
                        "fit_brier": empirical,
                        "bayes_brier": bayes,
                        "population_brier": bayes + excess,
                        "excess_brier": excess,
                        "optimism": bayes + excess - empirical,
                        "reg_family": methods["regularized"].chosen_,
                        "gate_gain": gate["gain"],
                        "gate_se": gate["se"],
                        "plain_uses_okapi": gate["plain"],
                        "guard_uses_okapi": gate["guard"],
                    }
                )
                forecasts.append(phat)
    return rows, forecasts


def summarize(scores, forecasts):
    draws = np.random.default_rng(97944).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    parts, contrasts = [], []
    for (d, n), cell in scores.groupby(["specialization", "size"]):
        _, mass, q = StratifiedPool(d).support()
        for name, group in cell.groupby("method"):
            p = forecasts[group.index]
            model_average = p.mean(axis=0)
            bias2 = 2 * np.dot(mass, (model_average - q) ** 2)
            var = 2 * np.dot(mass, p.var(axis=0, ddof=0))
            np.testing.assert_allclose(
                bias2 + var, group.excess_brier.mean(), rtol=0, atol=2e-13
            )
            parts.append(
                {
                    "specialization": d,
                    "size": n,
                    "method": name,
                    "bias_squared": bias2,
                    "variance": var,
                    "excess_brier": group.excess_brier.mean(),
                    "fit_brier": group.fit_brier.mean(),
                    "population_brier": group.population_brier.mean(),
                    "optimism": group.optimism.mean(),
                }
            )
        wide = cell.pivot(
            index="seed", columns="method", values="population_brier"
        ).loc[list(SEEDS)]
        for a, b in (
            ("regularized", "mean"),
            ("regularized", "top_n"),
            ("regularized", "prob_lr"),
            ("guard", "mean"),
            ("guard", "regularized"),
            ("plain", "mean"),
            ("guard", "plain"),
        ):
            v = (wide[a] - wide[b]).to_numpy()
            lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
            contrasts.append(
                {
                    "specialization": d,
                    "size": n,
                    "method": a,
                    "ref": b,
                    "difference": v.mean(),
                    "lo": lo,
                    "hi": hi,
                    "banks_better": int((v < -1e-12).sum()),
                    "banks_worse": int((v > 1e-12).sum()),
                }
            )
    return pd.DataFrame(parts), pd.DataFrame(contrasts)


def main():
    clock = perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(bank, SEEDS))
    scores = pd.DataFrame(row for records, _ in batches for row in records)
    forecasts = np.asarray([p for _, preds in batches for p in preds])
    assert len(scores) == 32 * 3 * 3 * 6 and forecasts.shape == (len(scores), 512)
    parts, contrasts = summarize(scores, forecasts)
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(OUT / "stratified-predictions.npz", p=forecasts)
    scores.to_csv(OUT / "stratified-scores.csv", index=False)
    parts.to_csv(OUT / "stratified-decomposition.csv", index=False)
    contrasts.to_csv(OUT / "stratified-contrasts.csv", index=False)
    (OUT / "stratified-manifest.json").write_text(
        json.dumps(
            {
                "seconds": perf_counter() - clock,
                "banks": 32,
                "strata": 4,
                "workers": 2,
                "search_seed": 0,
                "native_fits_including_outer_holdout": 2 * 32 * 3 * 3,
                "bootstrap_seed": 97944,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
