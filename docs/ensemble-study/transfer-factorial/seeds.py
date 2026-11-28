"""Cross 16 independent, exactly stratified training banks with six fixed optimizer seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.stratified_pool import StratifiedPool

HERE = Path(__file__).resolve().parent
SEEDS = range(257100, 257116)
OPTIMIZERS = range(6)
REFERENCES = ("mean", "top_n", "prob_lr")
METHODS = ("regularized", *REFERENCES)


def training_bank(scenario, seed):
    x, y, latent = scenario.sample(512, np.random.default_rng([seed, 1]))
    strata = 2 * latent["high_amplitude"].astype(int) + y
    rows = np.sort(np.concatenate([np.flatnonzero(strata == j)[:32] for j in range(4)]))
    np.testing.assert_array_equal(
        np.bincount(strata[rows], minlength=4), np.full(4, 32)
    )
    return x[rows], y[rows]


def checksum(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fit_bank(seed, scenario, support, mass, q):
    x, y = training_bank(scenario, seed)
    # Each reference is fitted once to this bank. Only regularized OKAPI varies
    # an optimizer seed. The split on these exact input rows stays fixed.
    plans = [
        ("mean", None, SimpleAverage()),
        ("top_n", None, TopN(metric="brier")),
        ("prob_lr", None, StackingLR(c_value=1.0)),
        *(
            ("regularized", s, RegularizedNativePatternOKAPI(seed=s, split_seed=0))
            for s in OPTIMIZERS
        ),
    ]
    records, forecasts = [], []
    floor = float(2 * np.dot(mass, q * (1 - q)))
    for name, s, model in plans:
        model.fit(x, y, task="binary", n_classes=2)
        p = np.asarray(model.predict(support))[:, 1]
        fit = float(2 * np.mean(np.square(model.predict(x)[:, 1] - y)))
        population = float(2 * np.dot(mass, q * (1 - p) ** 2 + (1 - q) * p**2))
        excess = float(2 * np.dot(mass, (p - q) ** 2))
        np.testing.assert_allclose(population, floor + excess, rtol=0, atol=2e-14)
        records.append(
            {
                "seed": seed,
                "optimizer_seed": s,
                "method": name,
                "size": len(y),
                "specialization": 0.25,
                "fit_brier": fit,
                "population_brier": population,
                "bayes_brier": floor,
                "excess_brier": excess,
                "optimism": population - fit,
                "chosen_family": getattr(model, "chosen_", None),
                "forecast_sha256": checksum(p),
            }
        )
        forecasts.append(p)
    return (
        records,
        forecasts,
        {"seed": seed, "fit_x_sha256": checksum(x), "fit_y_sha256": checksum(y)},
    )


def summarize(data, forecasts, mass, q):
    draws = np.random.default_rng(97946).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    components, by_bank = [], []
    if len(data) != len(SEEDS) * (len(OPTIMIZERS) + len(REFERENCES)):
        raise ValueError("Incomplete fit grid")
    floor = float(2 * np.dot(mass, q * (1 - q)))
    for method in METHODS:
        bank_means, bank_variances = [], []
        for seed in SEEDS:
            frame = data[(data.method == method) & (data.seed == seed)].sort_values(
                "optimizer_seed"
            )
            if method == "regularized":
                np.testing.assert_array_equal(
                    frame.optimizer_seed, np.arange(len(OPTIMIZERS))
                )
            elif len(frame) != 1:
                raise ValueError("Each reference must have exactly one fit per bank")
            sample = forecasts[frame.forecast_row.to_numpy()]
            bank_means.append(sample.mean(axis=0))
            bank_variances.append(sample.var(axis=0, ddof=0))
            row = {
                "seed": seed,
                "method": method,
                "fits": len(frame),
                **frame[
                    [
                        "fit_brier",
                        "population_brier",
                        "bayes_brier",
                        "excess_brier",
                        "optimism",
                    ]
                ]
                .mean()
                .to_dict(),
                "within_bank_optimizer_variance": float(
                    2 * np.dot(mass, bank_variances[-1])
                ),
                "bank_mean_forecast_excess": float(
                    2 * np.dot(mass, (bank_means[-1] - q) ** 2)
                ),
            }
            np.testing.assert_allclose(
                row["within_bank_optimizer_variance"]
                + row["bank_mean_forecast_excess"],
                row["excess_brier"],
                rtol=0,
                atol=2e-14,
            )
            by_bank.append(row)
        means = np.asarray(bank_means)
        bias2 = float(2 * np.dot(mass, (means.mean(axis=0) - q) ** 2))
        between = float(2 * np.dot(mass, means.var(axis=0, ddof=0)))
        within = float(2 * np.dot(mass, np.mean(bank_variances, axis=0)))
        measured = data[data.method == method][
            ["fit_brier", "population_brier", "excess_brier", "optimism"]
        ].mean()
        np.testing.assert_allclose(
            bias2 + between + within, measured.excess_brier, rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(
            floor + bias2 + between + within,
            measured.population_brier,
            rtol=0,
            atol=2e-14,
        )
        components.append(
            {
                "method": method,
                "banks": len(SEEDS),
                "fits_per_bank": len(OPTIMIZERS) if method == "regularized" else 1,
                "bayes_brier": floor,
                **measured.to_dict(),
                "squared_bias": bias2,
                "between_bank_variance": between,
                "within_bank_optimizer_variance": within,
                "total_variance": between + within,
            }
        )
    banks = pd.DataFrame(by_bank)
    comparisons = []
    for stat in ("population_brier", "excess_brier", "fit_brier", "optimism"):
        wide = banks.pivot(index="seed", columns="method", values=stat).loc[list(SEEDS)]
        for a, b in (
            ("regularized", "mean"),
            ("regularized", "top_n"),
            ("regularized", "prob_lr"),
            ("top_n", "mean"),
            ("prob_lr", "mean"),
        ):
            difference = (wide[a] - wide[b]).to_numpy()
            lo, hi = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
            comparisons.append(
                {
                    "method": a,
                    "reference": b,
                    "metric": stat,
                    "banks": len(difference),
                    "estimate": difference.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "banks_lower": int((difference < -1e-12).sum()),
                    "banks_tied": int((np.abs(difference) <= 1e-12).sum()),
                    "banks_higher": int((difference > 1e-12).sum()),
                }
            )
    return pd.DataFrame(components), pd.DataFrame(comparisons), banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    start = perf_counter()
    sc = StratifiedPool(0.25)
    support, mass, q = sc.support()
    records, forecasts, identities = [], [], []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            scores, predictions, keys = fit_bank(seed, sc, support, mass, q)
            records.extend(scores)
            forecasts.extend(predictions)
            identities.append(keys)
            print(
                f"Completed bank {seed} ({len(identities)}/{len(SEEDS)}).", flush=True
            )
    data = pd.DataFrame(records)
    data.insert(0, "forecast_row", np.arange(len(data)))
    forecasts = np.asarray(forecasts)
    parts, comparisons, banks = summarize(data, forecasts, mass, q)
    args.out.mkdir(parents=True, exist_ok=True)
    for label, frame in (
        ("scores", data),
        ("components", parts),
        ("contrasts", comparisons),
        ("bank-means", banks),
    ):
        frame.to_csv(args.out / f"seed-{label}.csv", index=False)
    np.savez_compressed(
        args.out / "seed-predictions.npz", p=forecasts, support=support, mass=mass, q=q
    )
    from okapi import pattern_logit

    sources = {
        "protocol": HERE / "seed-protocol.md",
        "runner": Path(__file__),
        "native_pattern_logit": Path(pattern_logit.__file__),
    }
    info = {
        "seconds": perf_counter() - start,
        "workers": 1,
        "numerical_threads_per_worker": 1,
        "banks": len(SEEDS),
        "native_fits": len(SEEDS) * len(OPTIMIZERS),
        "reference_fits": len(SEEDS) * len(REFERENCES),
        "optimizer_seeds": list(OPTIMIZERS),
        "split_seed": 0,
        "bootstrap_seed": 97946,
        "bootstrap_draws": 10000,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "training_banks": identities,
        "source_sha256": {
            k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in sources.items()
        },
    }
    (args.out / "seed-manifest.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )
    print(f"Completed {len(data)} fits in {info['seconds']:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
