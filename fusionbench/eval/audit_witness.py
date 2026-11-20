"""Measure residual label information when the validation-winning OKAPI tree is uncertain."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import search_champion
from fusionbench.eval.sentinel_audit import source_sample
from fusionbench.eval.stratified_audit import sample_indices


def permute_within(scores, blocks, cuts, rng):
    """Shuffle whole provider vectors inside prespecified winner-score bins."""
    bins = np.searchsorted(cuts, scores, side="right")
    result = np.empty_like(blocks)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        result[idx] = blocks[rng.permutation(idx)]
    return result


def designs(raw, trees, winner, cuts, seed):
    provider = logit(np.clip(raw[:, :, 1], 1e-8, 1 - 1e-8))
    bank = logit(np.clip(trees.T, 1e-8, 1 - 1e-8))
    w = bank[:, winner : winner + 1]
    shuffled = permute_within(w[:, 0], provider, cuts, np.random.default_rng(seed))
    return {
        "winner": w,
        "tree_bank": bank,
        "providers": provider,
        "winner_providers": np.column_stack([w, provider]),
        "within_bin_permutation": np.column_stack([w, shuffled]),
    }


def run_study(out):
    from okapi.sentinel import upper_tail_pvalues

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    powers = []
    clean_risks = []
    models = []
    cohorts = []
    with threadpool_limits(limits=1):
        for phase, seeds, gp_seeds in [
            ("development", range(1200, 1204), range(240, 248)),
            ("confirmation", range(1300, 1308), range(260, 268)),
        ]:
            for seed in seeds:
                x, y = source_sample(seed, 0, 384)
                bank = [search_champion(x, y, g, 10) for g in gp_seeds]
                xv, yv = source_sample(seed, 1, 256)
                scores = _predict_trees(bank, xv)
                winner = np.argmin(((scores - yv) ** 2).mean(1))
                xf, yf = source_sample(seed, 2, 2048)
                tc = _predict_trees(bank, xf)
                cuts = np.quantile(logit(np.clip(tc[winner], 1e-8, 1 - 1e-8)), np.arange(1, 40) / 40)
                design = designs(xf, tc, winner, cuts, [seed, 2, 9])
                predictors = {
                    name: LogisticRegression(C=1.0, max_iter=1000).fit(z, yf)
                    for name, z in design.items()
                }
                win_conf = abs(predictors["winner"].predict_proba(design["winner"])[:, 1] - 0.5)
                low, high = np.quantile(win_conf, [0.15, 0.85])
                models.append(
                    dict(
                        phase=phase,
                        data_seed=seed,
                        winner_gp_seed=gp_seeds[winner],
                        unique_trees=len({repr(tree) for tree in bank}),
                        low=low,
                        high=high,
                    )
                )

                def read_batches(split, nbatches):
                    raw, y = source_sample(seed, split, nbatches * 256)
                    tt = _predict_trees(bank, raw)
                    mats = designs(raw, tt, winner, cuts, [seed, split, 9])
                    probs = {
                        name: fit.predict_proba(mats[name])[:, 1].reshape(nbatches, 256)
                        for name, fit in predictors.items()
                    }
                    sample, groups = sample_indices(
                        probs["winner"], (low, high), 16, np.random.default_rng([seed, split, 30])
                    )
                    idx = sample["fixed_allocation"]
                    return probs, y.reshape(nbatches, 256), groups, idx

                ref, cy, cgroups, ci = read_batches(3, 399)
                test, y, groups, idx = read_batches(4, 512)
                low_group = groups == 0
                changed = np.random.default_rng([seed, 5]).random(y.shape) < 0.15
                ranked_y = np.take_along_axis(y, idx, 1)
                src_sel_y = np.take_along_axis(cy, ci, 1)
                low_s = np.take_along_axis(cgroups == 0, ci, 1)
                low_t = np.take_along_axis(low_group, idx, 1)
                within = groups == 0
                base_decision = test["winner"] >= 0.5
                for name, tp in test.items():
                    calpred = np.take_along_axis(ref[name], ci, 1) >= 0.5
                    pred = np.take_along_axis(tp, idx, 1) >= 0.5
                    all_decision = tp >= 0.5
                    for g, mask in [
                        ("all", np.ones_like(y, dtype=bool)),
                        ("low", within),
                        ("non_low", ~within),
                    ]:
                        clean_risks.append(
                            dict(
                                phase=phase,
                                data_seed=seed,
                                predictor=name,
                                group=g,
                                n=int(mask.sum()),
                                accuracy=(all_decision[mask] == y[mask]).mean(),
                                brier=((tp[mask] - y[mask]) ** 2).mean(),
                                disagreement=(all_decision[mask] != base_decision[mask]).mean(),
                                repairs=(
                                    (base_decision[mask] != y[mask]) & (all_decision[mask] == y[mask])
                                ).mean(),
                                harms=(
                                    (base_decision[mask] == y[mask]) & (all_decision[mask] != y[mask])
                                ).mean(),
                            )
                        )
                    for condition, flip in [
                        ("clean", np.zeros_like(y, dtype=bool)),
                        ("low_full", low_group),
                        ("independent_15", changed),
                    ]:
                        obs = ranked_y ^ np.take_along_axis(flip, idx, 1)
                        err = pred != obs
                        clean_err = calpred != src_sel_y
                        for scope, c_mask, t_mask in [
                            ("all_16", np.ones_like(low_s), np.ones_like(low_t)),
                            ("low_only", low_s, low_t),
                        ]:
                            # Missing a low case can only occur if a pool lacks enough examples.
                            # The denominator is the actual number of audited cases in the scope.
                            baseline = (clean_err * c_mask).sum(1) / c_mask.sum(1).clip(min=1)
                            current = (err * t_mask).sum(1) / t_mask.sum(1).clip(min=1)
                            u = np.random.default_rng([seed, 7, 16]).random(len(current))
                            pvalue = upper_tail_pvalues(baseline, current, u)
                            powers.append(
                                dict(
                                    phase=phase,
                                    data_seed=seed,
                                    predictor=name,
                                    condition=condition,
                                    scope=scope,
                                    detections=int((pvalue <= 0.05).sum()),
                                    n=len(pvalue),
                                    power=(pvalue <= 0.05).mean(),
                                )
                            )
                cohorts.append(
                    dict(
                        phase=phase,
                        data_seed=seed,
                        min_low_labels=low_t.sum(1).min(),
                        max_low_labels=low_t.sum(1).max(),
                    )
                )
                pd.DataFrame(powers).to_csv(out / "power.csv", index=False)
    risk = pd.DataFrame(clean_risks)
    risk.to_csv(out / "predictive.csv", index=False)
    summary = risk.groupby(["phase", "group", "predictor"])[
        ["accuracy", "brier", "disagreement", "repairs", "harms"]
    ].mean()
    summary.to_csv(out / "predictive_means.csv")
    pd.DataFrame(powers).groupby(["phase", "scope", "condition", "predictor"]).power.mean().to_csv(
        out / "power_means.csv"
    )
    pd.DataFrame(models).to_csv(out / "models.csv", index=False)
    pd.DataFrame(cohorts).to_csv(out / "cohorts.csv", index=False)
    p = risk.query("phase=='confirmation' and group=='low'").pivot(
        index="data_seed", columns="predictor", values="accuracy"
    )
    delta = p.winner_providers - p.winner
    radius = t.ppf(0.975, len(delta) - 1) * delta.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                source_fit_n=384,
                select_n=256,
                witness_fit_n=2048,
                reference_batches=399,
                test_batches=512,
                pool_size=256,
                quotas=[8, 4, 4],
                alpha=0.05,
                primary=dict(
                    difference=delta.mean(), lo=delta.mean() - radius, hi=delta.mean() + radius
                ),
                compute_seconds=time.perf_counter() - started,
                compute_started_utc=start_utc,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run_study(parser.parse_args().out)
