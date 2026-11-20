"""Falsify a single-fault context observer on independent GP seed banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import (
    disagreement_features,
    observations,
    search_champion,
    subset_weights,
)

# name, frequency of known channel-0 inversion, extra channel, extra frequency,
# temperature exponent for channel 0, independent probability of label reversal
CONDITIONS = [
    ("source", 0.1, None, 0.0, 1.0, 0.0),
    ("known", 0.5, None, 0.0, 1.0, 0.0),
    ("weak_new", 0.1, 3, 0.15, 1.0, 0.0),
    ("new", 0.1, 2, 0.5, 1.0, 0.0),
    ("joint", 0.5, 4, 0.5, 1.0, 0.0),
    ("calibration", 0.1, None, 0.0, 0.1, 0.0),
    ("hidden_label", 0.1, None, 0.0, 1.0, 0.40),
]


def features(x):
    votes, _ = disagreement_features(x)
    return np.column_stack([votes, abs(2 * x[:, :, 1] - 1)])


def modified(x, channel=None, temperature=1.0):
    if channel is None and temperature == 1:
        return x
    out = x.copy()
    if channel is not None:
        out[:, channel] = x[:, channel, ::-1]
    if temperature != 1:
        base = out[:, 0] ** temperature
        out[:, 0] = base / base.sum(1, keepdims=True)
    return out


def transfer_surface(target_mean, reference, perturbations, losses, perturbed_losses):
    """Fit a known inversion plus one unknown independent inversion.

    For fixed candidate channel and p(known), the vector of target means is
    affine in p(extra), allowing a clipped least-squares estimate. A grid
    over p(known) completes the search. The same weights mix conditional risks.
    """
    grid = np.linspace(0, 1, 41)
    original = (1 - grid[:, None]) * reference[0] + grid[:, None] * reference[1]
    best = None
    for j, perturbed in enumerate(perturbations, start=1):
        shifted = (1 - grid[:, None]) * perturbed[0] + grid[:, None] * perturbed[1]
        delta = shifted - original
        q = np.clip(
            ((target_mean - original) * delta).sum(1) / np.maximum((delta**2).sum(1), 1e-12), 0, 1
        )
        residual = original + q[:, None] * delta - target_mean
        sse = (residual**2).sum(1)
        i = int(np.argmin(sse))
        candidate = (sse[i], j, grid[i], q[i])
        if best is None or candidate < best:
            best = candidate
    residual, j, p, q = best
    a = (1 - p) * losses[0] + p * losses[1]
    b = (1 - p) * perturbed_losses[j - 1, 0] + p * perturbed_losses[j - 1, 1]
    return (1 - q) * a + q * b, j, p, q, residual


def test_losses(bank, ds, size):
    clean, known, y = observations(ds, 2, size)
    cache = {}
    for condition, p, j, q, temp, wrong in CONDITIONS:
        total = np.zeros(len(bank))
        for base, cweight in [(clean, 1 - p), (known, p)]:
            for extra, eweight in [(None, 1 - q), (j, q)]:
                if eweight == 0:
                    continue
                key = (base is known, extra, temp)
                if key not in cache:
                    z = _predict_trees(bank, modified(base, extra, temp))
                    cache[key] = ((z - y) ** 2).mean(1), ((z - (1 - y)) ** 2).mean(1)
                good, bad = cache[key]
                total += cweight * eweight * ((1 - wrong) * good + wrong * bad)
        yield condition, total


def run_study(
    output,
    data_seeds,
    gp_seeds,
    *,
    repeats=12,
    val_n=256,
    target_n=512,
    test_n=8192,
    fit_n=384,
    generations=10,
    draws=399,
):
    from okapi.context_diagnostics import assess_context_mixture, invariant_mean_pvalue

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data_seeds = list(data_seeds)
    gp_seeds = list(gp_seeds)
    records = []
    risk_rows = []
    with threadpool_limits(limits=1):
        for ds in data_seeds:
            a, b, y = observations(ds, 0, fit_n)
            mask = np.random.default_rng([ds, 11]).random(fit_n) < 0.1
            train = np.where(mask[:, None, None], b, a)
            bank = [search_champion(train, y, g, generations) for g in gp_seeds]
            tests = dict(test_losses(bank, ds, test_n))
            for condition, rr in tests.items():
                risk_rows.extend(
                    dict(data_seed=ds, gp_seed=g, condition=condition, test_brier=score)
                    for g, score in zip(gp_seeds, rr)
                )
            for repeat in range(repeats):
                a, b, y = observations(ds, 100 + repeat, val_n)
                mask = np.random.default_rng([ds, 13, repeat]).random(val_n) < 0.1
                if min(mask.sum(), (~mask).sum()) < 2:
                    raise ValueError("need two calibration examples per context")
                source = np.where(mask[:, None, None], b, a)
                f = features(source)
                n_pairs = f.shape[1] - source.shape[1]
                bit = np.argmax(abs(f[mask, :n_pairs].mean(0) - f[~mask, :n_pairs].mean(0)))
                # Under a single polarity inversion, pairwise disagreements that
                # exclude the faulty channel and all confidence magnitudes
                # are invariant. Use independent rows to choose the channel
                # and test these invariants, keeping target observations held out.
                train_idx = np.arange(val_n) < val_n // 2
                cmask = mask[train_idx]
                aidx, bidx = np.triu_indices(source.shape[1], 1)
                if cmask.any() and (~cmask).any():
                    first = f[train_idx, :n_pairs]
                    gap = abs(first[cmask].mean(0) - first[~cmask].mean(0))
                    suspect = int(
                        np.argmax([gap[(aidx == j) | (bidx == j)].sum() for j in range(source.shape[1])])
                    )
                else:
                    suspect = -1
                invariant = np.r_[
                    (aidx != suspect) & (bidx != suspect), np.ones(source.shape[1], dtype=bool)
                ]
                ref = np.stack([f[~mask].mean(0), f[mask].mean(0)])
                cv = (_predict_trees(bank, source) - y) ** 2
                erisk = cv.mean(1)
                crisk = np.stack([cv[:, ~mask].mean(1), cv[:, mask].mean(1)])
                minimax = np.maximum(0.9 * crisk[0] + 0.1 * crisk[1], 0.5 * (crisk[0] + crisk[1]))
                v_features = []
                v_risks = []
                for j in range(1, source.shape[1]):
                    simulated = modified(source, j)
                    ff = features(simulated)
                    ll = (_predict_trees(bank, simulated) - y) ** 2
                    v_features.append([ff[~mask].mean(0), ff[mask].mean(0)])
                    v_risks.append([ll[:, ~mask].mean(1), ll[:, mask].mean(1)])
                v_features = np.array(v_features)
                v_risks = np.array(v_risks)
                ta, tb, _ = observations(ds, 4000 + repeat, target_n)
                rng = np.random.default_rng([ds, 17, repeat])
                flags, extras = rng.random((2, target_n))
                for condition, p, j, q, temp, wrong in CONDITIONS:
                    raw = np.where((flags < p)[:, None, None], tb, ta)
                    xt = modified(raw, temperature=temp)
                    if j is not None:
                        xt = np.where((extras < q)[:, None, None], modified(xt, j), xt)
                    ft = features(xt)
                    absolute = tests[condition]
                    fits = {}
                    for name, idx in [
                        ("bit", [bit]),
                        ("votes", slice(0, n_pairs)),
                        ("all", slice(None)),
                    ]:
                        fits[name] = assess_context_mixture(
                            ft[:, idx],
                            f[~mask][:, idx],
                            f[mask][:, idx],
                            0.1,
                            draws=draws,
                            seed=10_000 * ds + repeat,
                        )
                    # The deployed two-context weight uses the original 21 moments.
                    weight = fits["votes"].prevalence
                    adapted = (1 - weight) * crisk[0] + weight * crisk[1]
                    alert = fits["all"].status != "compatible"
                    invariant_p = (
                        invariant_mean_pvalue(
                            f[~train_idx][:, invariant],
                            ft[:, invariant],
                            draws=draws,
                            seed=10_000 * ds + repeat,
                        )
                        if suspect >= 0
                        else np.nan
                    )
                    invariant_alert = not invariant_p > 0.05
                    family, channel, known_weight, extra_weight, sse = transfer_surface(
                        ft.mean(0), ref, v_features, crisk, v_risks
                    )
                    broad = np.maximum.reduce(
                        [
                            minimax,
                            *[
                                0.5 * (0.9 * crisk[0] + 0.1 * crisk[1]) + 0.5 * (0.9 * r[0] + 0.1 * r[1])
                                for r in v_risks
                            ],
                        ]
                    )
                    for name, score in [
                        ("validation", erisk),
                        ("minimax", minimax),
                        ("fault_minimax", broad),
                        ("two_context", adapted),
                        ("alert_to_validation", erisk if alert else adapted),
                        ("alert_to_minimax", minimax if alert else adapted),
                        ("family", family),
                        ("invariant_to_family", family if invariant_alert else adapted),
                        ("alert_to_family", family if alert else adapted),
                    ]:
                        weights = subset_weights(score, len(bank))
                        loss = weights @ absolute
                        records.append(
                            dict(
                                data_seed=ds,
                                repeat=repeat,
                                condition=condition,
                                selector=name,
                                test_brier=loss,
                                regret=loss - absolute.min(),
                                known_p=weight,
                                extra_channel=channel,
                                family_known_p=known_weight,
                                family_extra_p=extra_weight,
                                family_sse=sse,
                                bit_alert=fits["bit"].status != "compatible",
                                votes_alert=fits["votes"].status != "compatible",
                                all_alert=alert,
                                invariant_alert=invariant_alert,
                                invariant_pvalue=invariant_p,
                                source_suspect=suspect,
                                votes_pvalue=fits["votes"].p_value,
                                full_pvalue=fits["all"].p_value,
                                full_residual=fits["all"].max_residual,
                            )
                        )
            pd.DataFrame(records).to_csv(output / "stress.csv", index=False)
            pd.DataFrame(risk_rows).to_csv(output / "candidates.csv", index=False)
    config = dict(
        data_seeds=data_seeds,
        gp_seeds=gp_seeds,
        repeats=repeats,
        val_n=val_n,
        target_n=target_n,
        test_n=test_n,
        fit_n=fit_n,
        generations=generations,
        draws=draws,
        conditions=CONDITIONS,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-seeds", required=True, nargs="+", type=int)
    parser.add_argument("--gp-seeds", required=True, nargs="+", type=int)
    args = parser.parse_args()
    run_study(args.out, args.data_seeds, args.gp_seeds)
