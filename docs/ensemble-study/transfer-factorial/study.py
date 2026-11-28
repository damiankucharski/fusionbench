"""Fit on 24 paired source banks and integrate Brier risk on finite source and target laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR
from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

HERE = Path(__file__).resolve().parent
SEEDS = range(251100, 251124)
REGIONS = (0.5, 0.05)
SHORTCUTS = (0.0, 1.0)
NOISE = (0.0, 0.2)
METHODS = ("regularized", "mean", "top_n", "prob_lr")
AXES = ("region_fit", "shortcut_fit", "noise_fit", "noise_target")
LEVELS = (REGIONS, SHORTCUTS, NOISE, NOISE)
IDS = ["seed", *AXES, "method"]
PATTERNS = np.arange(16)
CONTRASTS = {
    "A_rare_source": ((0.05, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0)),
    "B_source_shortcut": ((0.5, 1.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0)),
    "C_train_noise_only": ((0.5, 0.0, 0.2, 0.0), (0.5, 0.0, 0.0, 0.0)),
    "D_target_noise_only": ((0.5, 0.0, 0.0, 0.2), (0.5, 0.0, 0.2, 0.2)),
    "E_joint_mismatch": ((0.05, 1.0, 0.0, 0.2), (0.5, 0.0, 0.2, 0.2)),
}
MEASURES = (
    "fit_brier",
    "source_brier",
    "target_brier",
    "target_excess",
    "target_rare_brier",
    "target_common_brier",
    "target_rare_excess",
    "target_common_excess",
    "signed_gap",
    "source_optimism",
    "domain_shift",
    "target_bayes_brier",
    "source_bayes_on_target_brier",
    "source_bayes_on_target_excess",
)


def scenario(r, s, noise_fit, noise_target):
    return RareShortcutNoise(
        region_fit=r,
        region_test=0.5,
        shortcut_fit=s,
        shortcut_test=0.0,
        provider_noise_fit=noise_fit,
        provider_noise_test=noise_target,
        label_noise_fit=0.0,
        label_noise_test=0.0,
    )


def pattern_ids(preds):
    """Lexicographic signs in public generalist/specialist/routing-cue/shortcut order."""
    return (np.asarray(preds)[:, :, 1] > 0.5).astype(np.int64) @ np.array([8, 4, 2, 1])


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def risk(p, mass, q):
    # Binary class-summed Brier, integrated over X,Y for the frozen predictor p(X).
    return float(2 * np.dot(mass, q * np.square(1 - p) + (1 - q) * np.square(p)))


def excess(p, mass, q):
    return float(2 * np.dot(mass, np.square(p - q)))


def domain(sc, split, prevalence):
    distributions = {}
    for sub, region in (("all", None), ("rare", 1), ("common", 0)):
        points, mass, q = sc.support(split, region=region)
        mass, q = np.asarray(mass), np.asarray(q)
        if (
            mass.shape != (16,)
            or q.shape != (16,)
            or not np.array_equal(pattern_ids(points), PATTERNS)
            or np.any(mass < 0)
            or not np.isclose(mass.sum(), 1, rtol=0, atol=1e-13)
            or not np.all((q >= 0) & (q <= 1))
        ):
            raise ValueError(
                "Expected a probability distribution in public sixteen-pattern order"
            )
        if sub == "all":
            support = points
        else:
            np.testing.assert_array_equal(points, support)
        distributions[sub] = (mass, q)
    m, q = distributions["all"]
    mr, qr = distributions["rare"]
    mc, qc = distributions["common"]
    np.testing.assert_allclose(
        m, prevalence * mr + (1 - prevalence) * mc, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        m * q, prevalence * mr * qr + (1 - prevalence) * mc * qc, rtol=0, atol=1e-14
    )
    return support, distributions


def law_map(metadata, points, distribution, prevalence):
    all_mass, all_q = distribution["all"]
    rare_mass, rare_q = distribution["rare"]
    common_mass, common_q = distribution["common"]
    rows = []
    for j in PATTERNS:
        rows.append(
            {
                **metadata,
                "pattern_id": int(j),
                "pattern": f"{j:04b}",
                **{
                    name: float(points[j, i, 1])
                    for i, name in enumerate(
                        ("generalist", "specialist", "routing_cue", "shortcut")
                    )
                },
                "mass": all_mass[j],
                "q": all_q[j],
                "rare_joint_mass": prevalence * rare_mass[j],
                "common_joint_mass": (1 - prevalence) * common_mass[j],
                "rare_given_pattern": prevalence * rare_mass[j] / all_mass[j]
                if all_mass[j] > 0
                else None,
                "mass_given_rare": rare_mass[j],
                "mass_given_common": common_mass[j],
                "q_given_rare": rare_q[j],
                "q_given_common": common_q[j],
            }
        )
    return rows


def fit_models(x, y):
    """Fit methods without access to target maps, source posteriors or latent region labels."""
    models = {
        "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0),
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
    }
    for model in models.values():
        model.fit(x, y, task="binary", n_classes=2)
    return models


def method_description(name, model):
    if name == "regularized":
        chosen = model.chosen_
        return chosen, len(model.trees_[chosen].unique_value_node_ids)
    if name == "top_n":
        return name, int(model.n_selected_)
    return name, 4


def evaluate(p, fit_loss, source, target):
    ms, qs = source["all"]
    mt, qt = target["all"]
    risk_s, risk_t = risk(p, ms, qs), risk(p, mt, qt)
    bayes_s, bayes_t = risk(qs, ms, qs), risk(qt, mt, qt)
    excess_t = excess(p, mt, qt)
    domain_shift = risk_t - risk_s
    optimism = risk_s - fit_loss
    gap = risk_t - fit_loss
    source_oracle_target_risk = risk(qs, mt, qt)
    transport = excess(qs, mt, qt)
    rare, common = (target[key] for key in ("rare", "common"))
    rb, cb = (risk(p, *split) for split in (rare, common))
    re, ce = (excess(p, *split) for split in (rare, common))
    np.testing.assert_allclose(
        [risk_t, risk_s, gap, risk_t, excess_t, source_oracle_target_risk],
        [
            bayes_t + excess_t,
            bayes_s + excess(p, ms, qs),
            optimism + domain_shift,
            0.5 * (rb + cb),
            0.5 * (re + ce),
            bayes_t + transport,
        ],
        rtol=0,
        atol=2e-13,
    )
    overlap = (ms > 0) & (mt > 0)
    return {
        "fit_brier": fit_loss,
        "source_brier": risk_s,
        "target_brier": risk_t,
        "source_bayes_brier": bayes_s,
        "source_excess": excess(p, ms, qs),
        "target_bayes_brier": bayes_t,
        "target_excess": excess_t,
        "signed_gap": gap,
        "source_optimism": optimism,
        "domain_shift": domain_shift,
        "target_rare_brier": rb,
        "target_common_brier": cb,
        "target_rare_excess": re,
        "target_common_excess": ce,
        "target_rare_bayes_brier": risk(rare[1], *rare),
        "target_common_bayes_brier": risk(common[1], *common),
        "source_bayes_on_target_brier": source_oracle_target_risk,
        "source_bayes_on_target_excess": transport,
        "target_mass_outside_source_support": float(np.sum(mt[ms == 0])),
        "max_posterior_change_on_overlap": float(
            np.max(np.abs(qs[overlap] - qt[overlap]))
        )
        if overlap.any()
        else None,
    }


def score_bank(seed):
    scores, predictions, fit_banks = [], [], []
    for r, s, nf in product(REGIONS, SHORTCUTS, NOISE):
        source_sc = scenario(r, s, nf, 0.0)
        x, y, latent = source_sc.sample_split(
            512, np.random.default_rng([seed, 1]), "fit"
        )
        frozen = fit_models(x, y)
        fit_region = np.asarray(latent["region"], dtype=bool)
        obs = pattern_ids(x)
        counts = np.bincount(obs, minlength=16)
        positives = np.bincount(obs, weights=y, minlength=16).astype(int)
        rare_counts = np.bincount(obs, weights=fit_region, minlength=16).astype(int)
        features, source = domain(source_sc, "fit", r)
        targets = {}
        for nt in NOISE:
            points, targets[nt] = domain(scenario(r, s, nf, nt), "test", 0.5)
            np.testing.assert_array_equal(features, points)
        cell = {"seed": seed, "region_fit": r, "shortcut_fit": s, "noise_fit": nf}
        fit_banks.append(
            {
                **cell,
                "n_fit": len(y),
                "n_rare_fit": int(fit_region.sum()),
                "x_sha256": digest(x),
                "y_sha256": digest(y),
                "region_sha256": digest(fit_region),
            }
        )
        for name, model in frozen.items():
            p = np.asarray(model.predict(features))[:, 1]
            empirical = np.asarray(model.predict(x))[:, 1]
            np.testing.assert_allclose(empirical, p[obs], rtol=0, atol=1e-12)
            fit_loss = float(2 * np.square(empirical - y).mean())
            hist_loss = float(
                2 * np.dot(positives, (1 - p) ** 2)
                + 2 * np.dot(counts - positives, p**2)
            ) / len(y)
            np.testing.assert_allclose(fit_loss, hist_loss, rtol=0, atol=1e-13)
            family, n_used = method_description(name, model)
            preds_key = digest(p)
            for j in PATTERNS:
                predictions.append(
                    {
                        **cell,
                        "method": name,
                        "family": family,
                        "pattern_id": int(j),
                        "pattern": f"{j:04b}",
                        "positive_probability": p[j],
                        "fit_count": int(counts[j]),
                        "positive_fit_count": int(positives[j]),
                        "rare_fit_count": int(rare_counts[j]),
                    }
                )
            for nt, destination in targets.items():
                values = evaluate(p, fit_loss, source, destination)
                scores.append(
                    {
                        **cell,
                        "noise_target": nt,
                        "method": name,
                        "family": family,
                        "n_providers": n_used,
                        "n_fit": len(y),
                        "n_rare_fit": int(fit_region.sum()),
                        "prediction_sha256": preds_key,
                        **values,
                    }
                )
    return scores, predictions, fit_banks


def interval(x, draws):
    arr = np.asarray(x, dtype=float)
    if arr.shape != (len(SEEDS),) or not np.isfinite(arr).all():
        raise ValueError("Expected one finite estimate per independent training bank")
    bounds = np.quantile(arr[draws].mean(axis=1), [0.025, 0.975])
    return {
        "estimate": float(arr.mean()),
        "ci_low": float(bounds[0]),
        "ci_high": float(bounds[1]),
        "banks": len(arr),
    }


def analyze(folder):
    scores = pd.read_csv(folder / "main-scores.csv")
    if len(scores) != len(SEEDS) * len(METHODS) * 16 or scores.duplicated(IDS).any():
        raise ValueError("Incomplete or duplicated factorial score grid")
    draws = np.random.default_rng(97941).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    summaries, comparisons, factorial, noise_changes, bank_contrasts = (
        [],
        [],
        [],
        [],
        [],
    )
    for method in METHODS:
        data = scores[scores.method == method].set_index([*AXES, "seed"]).sort_index()
        cells = {
            key: data.xs(key, level=list(AXES)).loc[list(SEEDS)]
            for key in product(*LEVELS)
        }
        for key, frame in cells.items():
            for metric in MEASURES:
                summaries.append(
                    {
                        "method": method,
                        **dict(zip(AXES, key, strict=True)),
                        "metric": metric,
                        **interval(frame[metric], draws),
                    }
                )
        for effect, (key, base) in CONTRASTS.items():
            for metric in MEASURES:
                vector = (cells[key][metric] - cells[base][metric]).to_numpy()
                comparisons.append(
                    {
                        "method": method,
                        "comparison": effect,
                        "metric": metric,
                        **interval(vector, draws),
                    }
                )
                if metric in (
                    "target_brier",
                    "target_excess",
                    "signed_gap",
                    "source_optimism",
                    "domain_shift",
                ):
                    bank_contrasts.extend(
                        {
                            "method": method,
                            "comparison": effect,
                            "metric": metric,
                            "seed": seed,
                            "difference": v,
                        }
                        for seed, v in zip(SEEDS, vector, strict=True)
                    )
        for r, s, nf in product(REGIONS, SHORTCUTS, NOISE):
            clean, noisy = (cells[r, s, nf, nt] for nt in NOISE)
            if not np.array_equal(
                clean["prediction_sha256"].to_numpy(),
                noisy["prediction_sha256"].to_numpy(),
            ):
                raise ValueError(
                    "Target evaluations used different trained predictions"
                )
            for metric in (
                "target_brier",
                "target_bayes_brier",
                "target_excess",
                "target_rare_brier",
                "target_common_brier",
                "target_rare_excess",
                "target_common_excess",
                "signed_gap",
            ):
                delta = noisy[metric] - clean[metric]
                noise_changes.append(
                    {
                        "method": method,
                        "region_fit": r,
                        "shortcut_fit": s,
                        "noise_fit": nf,
                        "metric": metric,
                        **interval(delta, draws),
                    }
                )
        # Signed contrasts are high minus low in the declared LEVELS, averaging
        # across every combination of the remaining factors. Thus order-two and
        # higher terms are ordinary differences of differences in Brier units.
        for order in range(1, 5):
            for axes in combinations(range(4), order):
                effect = "*".join(AXES[i] for i in axes)
                for metric in (
                    "target_brier",
                    "target_excess",
                    "target_rare_brier",
                    "target_common_brier",
                    "signed_gap",
                    "domain_shift",
                    "source_optimism",
                ):
                    vector = np.zeros(len(SEEDS))
                    for key, frame in cells.items():
                        sign = np.prod(
                            [1 if key[i] == LEVELS[i][1] else -1 for i in axes]
                        )
                        vector += sign * frame[metric].to_numpy() / 2 ** (4 - order)
                    factorial.append(
                        {
                            "method": method,
                            "effect": effect,
                            "order": order,
                            "metric": metric,
                            **interval(vector, draws),
                        }
                    )
    for name, rows in (
        ("cells", summaries),
        ("contrasts", comparisons),
        ("factorial-effects", factorial),
        ("frozen-target-noise", noise_changes),
        ("bank-contrasts", bank_contrasts),
    ):
        pd.DataFrame(rows).to_csv(folder / f"main-{name}.csv", index=False)
    counts = (
        scores[scores.noise_target == 0]
        .groupby(
            ["method", "region_fit", "shortcut_fit", "noise_fit", "family"],
            as_index=False,
        )
        .size()
    )
    counts.to_csv(folder / "main-family-counts.csv", index=False)


def all_maps():
    rows = []
    for r, s, nf in product(REGIONS, SHORTCUTS, NOISE):
        points, maps = domain(scenario(r, s, nf, 0.0), "fit", r)
        rows.extend(
            law_map(
                {
                    "domain": "source",
                    "region_rate": r,
                    "shortcut_rate": s,
                    "provider_noise": nf,
                },
                points,
                maps,
                r,
            )
        )
    for nt in NOISE:
        points, maps = domain(scenario(0.5, 0.0, 0.0, nt), "test", 0.5)
        rows.extend(
            law_map(
                {
                    "domain": "target",
                    "region_rate": 0.5,
                    "shortcut_rate": 0.0,
                    "provider_noise": nt,
                },
                points,
                maps,
                0.5,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.out)
        return
    utc = datetime.now(UTC).isoformat()
    started = perf_counter()
    results = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            results.append(score_bank(seed))
            print(f"Completed {len(results)}/{len(SEEDS)} paired banks.", flush=True)
    elapsed = perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(
        ("main-scores.csv", "main-support-predictions.csv.gz", "main-fit-banks.csv")
    ):
        table = pd.DataFrame(row for batch in results for row in batch[idx])
        table.to_csv(
            args.out / name,
            index=False,
            compression={"method": "gzip", "mtime": 0}
            if name.endswith(".gz")
            else None,
        )
    pd.DataFrame(all_maps()).to_csv(args.out / "main-law-maps.csv", index=False)
    analyze(args.out)
    import okapi
    from okapi import pattern_logit

    import fusionbench.scenarios.rare_shortcut_noise as generator

    source_files = {
        "runner": Path(__file__),
        "protocol": HERE / "protocol.md",
        "scenario": Path(generator.__file__),
        "okapi": Path(okapi.__file__),
        "pattern_logit": Path(pattern_logit.__file__),
    }
    info = {
        "started_utc": utc,
        "fit_and_score_seconds": elapsed,
        "total_seconds": perf_counter() - started,
        "workers": 1,
        "numerical_threads_per_worker": 1,
        "banks": len(SEEDS),
        "fits": len(SEEDS) * 8 * len(METHODS),
        "fit_rows_per_bank": 512,
        "search_seed": 0,
        "split_seed": 0,
        "bootstrap_seed": 97941,
        "bootstrap_draws": 10000,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "sources_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in source_files.items()
        },
    }
    (args.out / "main-run.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"Wrote {info['fits']} fitted models and {2 * info['fits']} target evaluations in {info['total_seconds']:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
