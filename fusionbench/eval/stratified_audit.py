"""Equal-label-budget batch monitoring with probability and confidence-stratified sampling."""

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
from fusionbench.eval.sentinel_audit import batch_scores, source_sample


BUDGETS = (8, 16, 32)
POLICIES = ("random", "uncertainty_pps", "random_allocation", "fixed_allocation")
METHODS = ("error", "likelihood_ratio")


def sample_indices(probs, cutoffs, budget, rng):
    """Return nested equal-budget samples for each policy; labels are unavailable."""
    p = np.asarray(probs)
    if p.ndim != 2 or not 0 < budget <= p.shape[1] or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("expected valid pools of probabilities and a feasible budget")
    low, high = cutoffs
    if not 0 <= low < high <= 0.5:
        raise ValueError("expected ordered absolute confidence cutoffs")
    confidence = abs(p - 0.5)
    strata = np.where(confidence <= low, 0, np.where(confidence >= high, 2, 1))
    # A common exponential key ordering couples simple random sampling and PPS.
    keys = rng.exponential(size=p.shape)
    random_order = np.argsort(keys, axis=1)
    result = {
        "random": random_order[:, :budget],
        "uncertainty_pps": np.argsort(keys / (0.01 + p * (1 - p)), axis=1)[:, :budget],
    }
    pools = len(p)
    targets = {
        "random_allocation": rng.choice(3, size=(pools, budget), p=(0.5, 0.25, 0.25)),
        "fixed_allocation": np.broadcast_to(np.resize([0, 1, 0, 2], budget), (pools, budget)),
    }
    for name, schedule in targets.items():
        chosen = np.empty((pools, budget), dtype=int)
        for i, order in enumerate(random_order):
            ordered_strata = strata[i, order]
            group_order = [order[ordered_strata == g] for g in range(3)]
            offset = [0, 0, 0]
            used = np.zeros(len(order), dtype=bool)
            fallback = 0
            for k, g in enumerate(schedule[i]):
                options = group_order[g]
                while offset[g] < len(options) and used[options[offset[g]]]:
                    offset[g] += 1
                if offset[g] < len(options):
                    pick = options[offset[g]]
                    offset[g] += 1
                else:
                    while used[order[fallback]]:
                        fallback += 1
                    pick = order[fallback]
                chosen[i, k] = pick
                used[pick] = True
        result[name] = chosen
    return result, strata


def run_study(out, phases=("development", "confirmation")):
    from okapi.sentinel import upper_tail_pvalues

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    all_rows = []
    coverage = []
    models = []
    splits = {
        "development": (range(1000, 1004), range(200, 208)),
        "confirmation": (range(1100, 1108), range(220, 228)),
    }
    with threadpool_limits(limits=1):
        for phase in phases:
            data_seeds, gp_seeds = splits[phase]
            phase_start = time.perf_counter()
            for seed in data_seeds:
                train, y = source_sample(seed, 0, 384)
                forest = [search_champion(train, y, g, 10) for g in gp_seeds]
                validation, yv = source_sample(seed, 1, 256)
                winner = np.argmin(((_predict_trees(forest, validation) - yv) ** 2).mean(1))
                tree = forest[winner]

                def raw_logits(x):
                    return logit(np.clip(_predict_trees([tree], x)[0], 1e-8, 1 - 1e-8)).reshape(-1, 1)

                xc, yc = source_sample(seed, 2, 2048)
                log_odds = raw_logits(xc)
                cal = LogisticRegression(C=1.0, max_iter=1000).fit(log_odds, yc)
                lower, mid_lo, mid_hi, upper = np.quantile(
                    abs(cal.predict_proba(log_odds)[:, 1] - 0.5), [0.15, 0.425, 0.575, 0.85]
                )
                models.append(
                    dict(
                        phase=phase,
                        data_seed=seed,
                        gp_seed=gp_seeds[winner],
                        nodes=tree.nodes_count,
                        tree=repr(tree),
                        low=lower,
                        middle_low=mid_lo,
                        middle_high=mid_hi,
                        high=upper,
                    )
                )

                def pools(split, n):
                    raw, labels = source_sample(seed, split, n * 256)
                    pp = np.clip(cal.predict_proba(raw_logits(raw))[:, 1], 1e-10, 1 - 1e-10).reshape(
                        n, 256
                    )
                    samples, groups = sample_indices(
                        pp, (lower, upper), max(BUDGETS), np.random.default_rng([seed, split, 30])
                    )
                    return pp, labels.reshape(n, 256), samples, groups

                ref_p, ref_y, ref_idx, _ = pools(3, 399)
                tp, ty, test_idx, test_strata = pools(4, 512)
                change = np.random.default_rng([seed, 5]).random(ty.shape)
                c = abs(tp - 0.5)
                shifts = {
                    "clean": np.zeros_like(ty, dtype=bool),
                    "independent_15": change < 0.15,
                    "low_full": c <= lower,
                    "low_half": (c <= lower) & (change < 0.5),
                    "high_full": c >= upper,
                    "middle_band": (c >= mid_lo) & (c <= mid_hi),
                }
                for policy in POLICIES:
                    cp = np.take_along_axis(ref_p, ref_idx[policy], 1)
                    cy = np.take_along_axis(ref_y, ref_idx[policy], 1)
                    chosen = test_idx[policy]
                    p = np.take_along_axis(tp, chosen, 1)
                    g = np.take_along_axis(test_strata, chosen, 1)
                    for n in BUDGETS:
                        reference = batch_scores(cp, cy, n)
                        uniforms = np.random.default_rng([seed, 6, n]).random(len(tp))
                        coverage.append(
                            dict(
                                phase=phase,
                                data_seed=seed,
                                policy=policy,
                                budget=n,
                                low=(g[:, :n] == 0).mean(),
                                middle=(g[:, :n] == 1).mean(),
                                high=(g[:, :n] == 2).mean(),
                                variance_low_count=(g[:, :n] == 0).sum(1).var(),
                            )
                        )
                        for condition, changed in shifts.items():
                            y = np.take_along_axis(ty ^ changed, chosen, 1)
                            scores = batch_scores(p, y, n)
                            for method in METHODS:
                                pv = upper_tail_pvalues(reference[method], scores[method], uniforms)
                                all_rows.append(
                                    dict(
                                        phase=phase,
                                        data_seed=seed,
                                        policy=policy,
                                        budget=n,
                                        condition=condition,
                                        method=method,
                                        detections=int(np.sum(pv <= 0.05)),
                                        batches=len(pv),
                                        power=(pv <= 0.05).mean(),
                                    )
                                )
                pd.DataFrame(all_rows).to_csv(out / "power.csv", index=False)
            (out / f"{phase}-time.json").write_text(
                json.dumps(dict(seconds=time.perf_counter() - phase_start), indent=2)
            )
    data = pd.DataFrame(all_rows)
    data.groupby(["phase", "policy", "budget", "condition", "method"]).power.mean().to_csv(
        out / "means.csv"
    )
    pd.DataFrame(coverage).to_csv(out / "sampling.csv", index=False)
    pd.DataFrame(models).to_csv(out / "models.csv", index=False)
    delta = data.query(
        "phase=='confirmation' and budget==16 and condition=='low_full' and method=='error'"
    )
    if len(delta):
        pivot = delta.pivot(index="data_seed", columns="policy", values="power")
        d = pivot.fixed_allocation - pivot.random
        radius = t.ppf(0.975, len(d) - 1) * d.sem()
        primary = dict(difference=d.mean(), lo=d.mean() - radius, hi=d.mean() + radius)
    else:
        primary = {}
    (out / "config.json").write_text(
        json.dumps(
            dict(
                phases=list(phases),
                pool_size=256,
                budgets=BUDGETS,
                clean_batches=399,
                test_batches=512,
                alpha=0.05,
                compute_seconds=time.perf_counter() - started,
                compute_started_utc=start_utc,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
                primary=primary,
            ),
            indent=2,
        )
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run_study(parser.parse_args().out)
