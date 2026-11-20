"""Power of small random label audits under otherwise invisible conditional shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import observations, search_champion


BUDGETS = (4, 8, 16, 32, 64)
SCENARIOS = (
    "clean",
    "independent_05",
    "independent_15",
    "independent_40",
    "full",
    "high_confidence",
    "low_confidence",
)


def source_sample(seed, split, size):
    clean, changed, labels = observations(seed, split, size)
    switch = np.random.default_rng([seed, split, 721]).random(size) < 0.1
    return np.where(switch[:, None, None], changed, clean), labels


def true_class_probability(probability, label):
    return np.where(label, probability, 1 - probability)


def batch_scores(probability, label, budget):
    """All labeled scores use identical records and identical recalibration."""
    p, y = probability[:, :budget], label[:, :budget]
    assigned = true_class_probability(p, y)
    alt = 0.85 * assigned + 0.15 * (1 - assigned)
    return {
        "error": ((p >= 0.5) != y).mean(1),
        "brier": ((p - y) ** 2).mean(1),
        "log_loss": -np.log(assigned).mean(1),
        "likelihood_ratio": (np.log(alt) - np.log(assigned)).sum(1),
        "unlabeled_uncertainty": np.minimum(p, 1 - p).mean(1),
    }


def run_study(out):
    from okapi.sentinel import upper_tail_pvalues

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data_seeds = list(range(900, 908))
    gp_seeds = list(range(110, 122))
    cal_batches, test_batches = 399, 512
    power, risks, models = [], [], []
    with threadpool_limits(limits=1):
        for seed in data_seeds:
            train, y = source_sample(seed, 0, 384)
            bank = [search_champion(train, y, g, 10) for g in gp_seeds]
            val, val_y = source_sample(seed, 1, 256)
            losses = ((_predict_trees(bank, val) - val_y) ** 2).mean(1)
            best = np.argmin(losses)
            tree = bank[best]

            def raw_logits(scores):
                pred = _predict_trees([tree], scores)[0]
                return logit(np.clip(pred, 1e-8, 1 - 1e-8)).reshape(-1, 1)

            source, source_y = source_sample(seed, 2, 2048)
            source_logit = raw_logits(source)
            calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(source_logit, source_y)
            probs = calibrator.predict_proba(source_logit)[:, 1]
            lower, upper = np.quantile(abs(probs - 0.5), [0.15, 0.85])
            models.append(
                dict(
                    data_seed=seed,
                    gp_seed=gp_seeds[best],
                    nodes=tree.nodes_count,
                    tree=repr(tree),
                    selection_brier=losses[best],
                    calibration_slope=calibrator.coef_[0, 0],
                    calibration_intercept=calibrator.intercept_[0],
                    low_confidence_cutoff=lower,
                    high_confidence_cutoff=upper,
                )
            )

            def batches(split, count):
                raw, y = source_sample(seed, split, count * max(BUDGETS))
                p = calibrator.predict_proba(raw_logits(raw))[:, 1]
                return np.clip(p, 1e-10, 1 - 1e-10).reshape(count, -1), y.reshape(count, -1)

            clean_p, clean_y = batches(3, cal_batches)
            p, y = batches(4, test_batches)
            confidence = abs(p - 0.5)
            flip = np.random.default_rng([seed, 5]).random(y.shape)
            changes = [
                np.zeros_like(y, dtype=bool),
                flip < 0.05,
                flip < 0.15,
                flip < 0.40,
                np.ones_like(y, dtype=bool),
                confidence >= upper,
                confidence <= lower,
            ]
            baseline = {}
            ref = {n: batch_scores(clean_p, clean_y, n) for n in BUDGETS}
            for name, mask in zip(SCENARIOS, changes):
                target_y = np.logical_xor(y, mask)
                risks.append(
                    dict(
                        data_seed=seed,
                        scenario=name,
                        flip_fraction=mask.mean(),
                        error=np.mean((p >= 0.5) != target_y),
                        brier=np.mean((p - target_y) ** 2),
                        log_loss=-np.mean(np.log(true_class_probability(p, target_y))),
                    )
                )
                for n in BUDGETS:
                    uniforms = np.random.default_rng([seed, 6, n]).random(test_batches)
                    for method, score in batch_scores(p, target_y, n).items():
                        pvalue = upper_tail_pvalues(ref[n][method], score, uniforms)
                        alarm = pvalue <= 0.05
                        if method == "unlabeled_uncertainty":
                            if name == "clean":
                                baseline[n] = alarm
                            else:
                                np.testing.assert_array_equal(alarm, baseline[n])
                        power.append(
                            dict(
                                data_seed=seed,
                                scenario=name,
                                budget=n,
                                method=method,
                                detections=alarm.sum(),
                                batches=test_batches,
                                power=alarm.mean(),
                            )
                        )
            pd.DataFrame(power).to_csv(out / "power.csv", index=False)
            pd.DataFrame(risks).to_csv(out / "risks.csv", index=False)
            pd.DataFrame(models).to_csv(out / "models.csv", index=False)
    d = pd.DataFrame(power)
    d.groupby(["scenario", "budget", "method"]).power.mean().to_csv(out / "means.csv")
    primary = d.query("scenario=='independent_15' and budget==16").pivot(
        index="data_seed", columns="method", values="power"
    )
    delta = primary.likelihood_ratio - primary.brier
    half = t.ppf(0.975, len(delta) - 1) * delta.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                data_seeds=data_seeds,
                gp_seeds=gp_seeds,
                generations=10,
                fit_n=384,
                select_n=256,
                recalibrate_n=2048,
                clean_reference_batches=cal_batches,
                test_batches=test_batches,
                budgets=BUDGETS,
                scenarios=SCENARIOS,
                alpha=0.05,
                primary_lr_minus_brier=dict(
                    mean=delta.mean(), lo=delta.mean() - half, hi=delta.mean() + half
                ),
            ),
            indent=2,
        )
    )
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run_study(parser.parse_args().out)
