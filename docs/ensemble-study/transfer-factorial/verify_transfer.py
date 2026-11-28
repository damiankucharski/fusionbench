"""Independently integrate every archived forecast and refit the first and last banks."""

import hashlib
import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

OUT = Path(__file__).resolve().parent / "results"
SOURCE = ["seed", "region_fit", "shortcut_fit", "noise_fit"]
POLICIES = ("mean", "top_n", "prob_lr", "regularized")
BANKS = range(251100, 251124)
ROLES = np.array([8, 4, 2, 1])
SIGNS = np.array(list(product((0, 1), repeat=4)))
CONF = np.array([0.75, 0.75, 0.75, 0.9])
P1 = np.where(SIGNS, CONF, 1 - CONF)
POINTS = np.stack([1 - P1, P1], axis=-1)
TOL = 2e-12


@lru_cache
def joint(region_rate, share, provider_noise, label_noise=0):
    """Construct P(observed reports, observed label) by forward enumeration."""
    clean = np.zeros((16, 2))
    for idx, (g, s, r, h) in enumerate(SIGNS):
        for y in (0, 1):
            ghit = 0.75 if g == y else 0.25
            sh = 0.95 if r else 0.5
            shit = sh if s == y else 1 - sh
            shortcut = share * (0.9 if h == y else 0.1) + (1 - share) / 2
            clean[idx, y] = (
                0.5 * (region_rate if r else 1 - region_rate) * ghit * shit * shortcut
            )
    result = np.zeros_like(clean)
    for fg, fs, fy in product((0, 1), repeat=3):
        weight = (
            (provider_noise if fg else 1 - provider_noise)
            * (provider_noise if fs else 1 - provider_noise)
            * (label_noise if fy else 1 - label_noise)
        )
        result += (
            weight * clean[np.arange(16) ^ (fg * 8 + fs * 4)][:, np.arange(2) ^ fy]
        )
    np.testing.assert_allclose(result.sum(), 1, rtol=0, atol=1e-14)
    return result


def posterior(j):
    return np.divide(
        j[:, 1], j.sum(axis=1), out=np.full(16, 0.5), where=j.sum(axis=1) > 0
    )


def risk(p, j):
    return float(2 * np.sum(j * (p[:, None] - np.arange(2)) ** 2))


def excess(p, j):
    return float(2 * np.dot(j.sum(axis=1), (p - posterior(j)) ** 2))


def replicate_fit(seed, pi, share, noise):
    # Replay original primitive streams directly, including perfect cue and clean-label draws.
    rng = np.random.default_rng([int(seed), 1])
    n = 512
    clean = rng.integers(0, 2, size=n)
    region = rng.random(n) < pi
    g = clean ^ (rng.random(n) >= 0.75)
    s = clean ^ (rng.random(n) >= (0.5 + region * 0.45))
    cue = np.where(rng.random(n) < 1, region, ~region)
    flip = rng.random((n, 2)) < noise
    on = rng.random(n) < share
    signal = clean ^ (rng.random(n) >= 0.9)
    h = np.where(on, signal, rng.integers(2, size=n))
    y = clean ^ (rng.random(n) < 0)
    index = (g ^ flip[:, 0]) * 8 + (s ^ flip[:, 1]) * 4 + cue * 2 + h
    return POINTS[index], y, region, index


def quantities(p, emp, j, t):
    src, dst = risk(p, j), risk(p, t)
    rare = t.copy()
    rare[SIGNS[:, 2] == 0] = 0
    rare /= rare.sum()
    common = t.copy()
    common[SIGNS[:, 2] == 1] = 0
    common /= common.sum()
    qs = posterior(j)
    values = {
        "fit_brier": emp,
        "source_brier": src,
        "target_brier": dst,
        "source_bayes_brier": risk(qs, j),
        "source_excess": excess(p, j),
        "target_bayes_brier": risk(posterior(t), t),
        "target_excess": excess(p, t),
        "source_optimism": src - emp,
        "domain_shift": dst - src,
        "signed_gap": dst - emp,
        "source_bayes_on_target_brier": risk(qs, t),
        "source_bayes_on_target_excess": excess(qs, t),
        "target_mass_outside_source_support": float(t[j.sum(axis=1) == 0].sum()),
        "max_posterior_change_on_overlap": np.abs(qs - posterior(t)).max(),
    }
    for key, group in (("rare", rare), ("common", common)):
        values.update(
            {
                f"target_{key}_brier": risk(p, group),
                f"target_{key}_excess": excess(p, group),
                f"target_{key}_bayes_brier": risk(posterior(group), group),
            }
        )
    np.testing.assert_allclose(
        dst, values["target_excess"] + values["target_bayes_brier"], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        dst - emp, values["source_optimism"] + values["domain_shift"], rtol=0, atol=TOL
    )
    return values


def main():
    def load(name):
        return pd.read_csv(OUT / name, float_precision="round_trip")

    preds = load("main-support-predictions.csv.gz")
    scores = load("main-scores.csv").set_index(
        [*SOURCE, "noise_target", "method"], verify_integrity=True
    )
    fits = load("main-fit-banks.csv").set_index(SOURCE, verify_integrity=True)
    maps = load("main-law-maps.csv")
    for (r, s, n), frame in maps.groupby(
        ["region_rate", "shortcut_rate", "provider_noise"]
    ):
        j = joint(r, s, n)
        for _, sub in frame.groupby("domain"):
            m = sub.sort_values("pattern_id")
            np.testing.assert_array_equal(m.pattern_id, np.arange(16))
            np.testing.assert_allclose(m.mass, j.sum(axis=1), rtol=0, atol=TOL)
            np.testing.assert_allclose(m.q, posterior(j), rtol=0, atol=TOL)
    rebuilt = []
    max_refit = max_risk = 0.0
    fields = models = count = 0
    for keys, rows in preds.groupby(SOURCE, sort=True):
        seed, pi, share, n = keys
        x, y, region, idx = replicate_fit(*keys)
        for name, array in (("x", x), ("y", y), ("region", region)):
            assert (
                fits.loc[keys, name + "_sha256"]
                == hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            )
        assert int(region.sum()) == fits.loc[keys, "n_rare_fit"]
        for method, data in rows.groupby("method"):
            data = data.sort_values("pattern_id")
            np.testing.assert_array_equal(data.pattern_id, np.arange(16))
            np.testing.assert_array_equal(
                data.fit_count, np.bincount(idx, minlength=16)
            )
            np.testing.assert_array_equal(
                data.positive_fit_count, np.bincount(idx, weights=y, minlength=16)
            )
            np.testing.assert_array_equal(
                data.rare_fit_count, np.bincount(idx, weights=region, minlength=16)
            )
            frozen = data.positive_probability.to_numpy()
            if seed in (BANKS[0], BANKS[-1]):
                model = {
                    "mean": SimpleAverage,
                    "top_n": lambda: TopN(metric="brier"),
                    "prob_lr": lambda: StackingLR(c_value=1.0),
                    "regularized": lambda: RegularizedNativePatternOKAPI(
                        seed=0, split_seed=0
                    ),
                }[method]()
                model.fit(x, y, task="binary", n_classes=2)
                delta = abs(model.predict(POINTS)[:, 1] - frozen).max()
                max_refit = max(max_refit, delta)
                assert delta <= TOL, (keys, method, delta)
                if method == "regularized":
                    assert model.chosen_ == data.family.iloc[0]
                models += 1
            for target_noise in (0.0, 0.2):
                archived = scores.loc[(*keys, target_noise, method)]
                expected = quantities(
                    frozen,
                    float(np.mean(2 * (frozen[idx] - y) ** 2)),
                    joint(pi, share, n),
                    joint(0.5, 0.0, target_noise),
                )
                assert (
                    archived.prediction_sha256
                    == hashlib.sha256(frozen.tobytes()).hexdigest()
                )
                residual = np.abs(
                    archived[list(expected)].to_numpy(float) - list(expected.values())
                )
                assert residual.max() <= TOL, (keys, target_noise, method, residual)
                fields += len(expected)
                count += 1
                max_risk = max(max_risk, residual.max())
                rebuilt.append(
                    {
                        "seed": seed,
                        "cell": (pi, share, n, target_noise),
                        "method": method,
                        **expected,
                    }
                )

    result = pd.DataFrame(rebuilt)
    assert count == len(scores) == 1536 and len(preds) == 12288 and models == 64
    means = result.pivot_table(index="cell", columns="method", values="target_brier")
    ranks = means.rank(axis=1)
    cells = [
        {
            "cell": list(key),
            "regularized_rank": float(ranks.at[key, "regularized"]),
            "mean_target_brier": {
                name: float(means.at[key, name]) for name in POLICIES
            },
        }
        for key in means.index
    ]
    draws = np.random.default_rng(97941).integers(0, 24, (10000, 24))
    refs = (
        load("main-contrasts.csv")
        .query("method=='regularized'")
        .set_index(["comparison", "metric"])
    )
    pairs = {
        "A_rare_source": ((0.05, 0, 0, 0), (0.5, 0, 0, 0)),
        "B_source_shortcut": ((0.5, 1, 0, 0), (0.5, 0, 0, 0)),
        "C_train_noise_only": ((0.5, 0, 0.2, 0), (0.5, 0, 0, 0)),
        "D_target_noise_only": ((0.5, 0, 0, 0.2), (0.5, 0, 0.2, 0.2)),
        "E_joint_mismatch": ((0.05, 1, 0, 0.2), (0.5, 0, 0.2, 0.2)),
    }
    contrasts = {}
    for (compare, metric), row in refs.iterrows():
        table = (
            result[result.method == "regularized"]
            .pivot(index="seed", columns="cell", values=metric)
            .reindex(BANKS)
        )
        high, baseline = pairs[compare]
        diff = (table[high] - table[baseline]).to_numpy()
        bounds = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
        triple = np.r_[diff.mean(), bounds]
        np.testing.assert_allclose(
            triple, [row.estimate, row.ci_low, row.ci_high], rtol=0, atol=TOL
        )
        if metric == "target_brier":
            contrasts[compare] = dict(
                zip(("estimate", "ci_low", "ci_high"), triple.tolist(), strict=True)
            )
    verified = {
        "training_banks": len(fits),
        "predictions": len(preds),
        "risk_rows": count,
        "risk_fields": fields,
        "max_absolute_risk_discrepancy": max_risk,
        "reference_refits": models,
        "replay_seeds": [BANKS[0], BANKS[-1]],
        "max_absolute_prediction_discrepancy": max_refit,
        "bootstrap_contrasts_verified": len(refs),
        "target_brier_contrasts_regularized": contrasts,
        "rank_method": "rank of mean population Brier among four methods per cell",
        "mean_rank_across_16_cells": ranks.mean().to_dict(),
        "regularized_lower_than": {
            "vs_" + m: int((means.regularized < means[m] - 1e-12).sum())
            for m in POLICIES[:-1]
        },
        "cell_order": ["region_fit", "shortcut_fit", "noise_fit", "noise_target"],
        "cells": cells,
    }
    (OUT / "verification-transfer.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in verified.items() if k not in ("cells", "cell_order")},
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
