"""Compare label-prior and score location estimators on polarity_fault and UCI HAR."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.eval.covariance_transport import (
    FAULT,
    Population,
    draw,
    posterior,
    projected_correlation,
    source_parameters,
)
from fusionbench.scenarios.polarity import PolarityFault
from okapi.score_shift import ReferenceScoreCDF
from real_transfer import populations


def estimates(reference, scores):
    methods = {"mean": (reference.prior_from_mean(scores), 0.0, np.nan)}
    for name, allow in [("cdf_fixed", False), ("cdf_joint", True)]:
        fit = reference.fit(scores, allow_offset=allow)
        methods[name] = (fit.prior, fit.offset, fit.cdf_rmse)
    return methods


def synthetic():
    s_true = PolarityFault().strengths
    conditions = {
        "iid": (Population(), 1),
        "prior": (Population(prior=0.8), 1),
        "offset": (Population(stable_offset=0.6, anchor_offset=0.6), 1),
        "both": (Population(prior=0.8, stable_offset=0.6, anchor_offset=0.6), 1),
        "scale": (Population(), 1.7),
    }
    rows = []
    for phase, seeds in [("development", range(1800, 1804)), ("confirmation", range(1900, 1912))]:
        for seed in seeds:
            x, _, y = draw(seed, 0, 768, Population(), s_true)
            s, frozen = source_parameters(x, y)
            v, _, vy = draw(seed, 1, 768, Population(), s_true)
            ref = ReferenceScoreCDF(v[vy == 0, 0], v[vy == 1, 0])
            for condition, (population, scale) in conditions.items():
                adapt = draw(seed, 2, 1024, population, s_true)[0] * scale
                xt, _, yt = draw(seed, 3, 8192, population, s_true)
                oracle = posterior(xt, s_true, population)
                xt *= scale
                true_offset = population.anchor_offset * (1 - 2 * FAULT) * s_true[0]
                for n in (128, 1024):
                    a = adapt[:n]
                    rho = projected_correlation(a[:, 1:], s[1:])
                    stable_mean = a[:, 1:].mean(0) @ s[1:] / (s[1:] @ s[1:])
                    methods = estimates(ref, a[:, 0])
                    methods["frozen"] = (frozen.prior, 0.0, np.nan)
                    for method, (prior, offset, residual) in methods.items():
                        stable_offset = stable_mean - (2 * prior - 1) if method == "cdf_joint" else 0.0
                        fitted = Population(rho, prior, stable_offset, offset / ((1 - 2 * FAULT) * s[0]))
                        prob = posterior(xt, s, frozen if method == "frozen" else fitted)
                        rows.append(
                            dict(
                                phase=phase,
                                seed=seed,
                                condition=condition,
                                budget=n,
                                method=method,
                                estimated_prior=prior,
                                true_prior=population.prior,
                                estimated_offset=offset,
                                true_offset=true_offset,
                                prior_abs_error=abs(prior - population.prior),
                                offset_abs_error=abs(offset - true_offset),
                                cdf_rmse=residual,
                                excess_brier=np.mean((prob - oracle) ** 2),
                                brier=np.mean((prob - yt) ** 2),
                            )
                        )
    return pd.DataFrame(rows)


def temporal_pools(rows, y, person_seeds):
    early, late = [[], []], [[], []]
    for label in (0, 1):
        segment = rows[y[rows] == label]
        side = (len(segment) - 5) // 2
        if side < 2:
            raise ValueError("Insufficient windows for temporal separation")
        early[label] = person_seeds.permutation(segment[:side])
        late[label] = person_seeds.permutation(segment[-side:])
        assert early[label].max() + 5 < late[label].min()
    return early, late


def choose_rows(pool, prior):
    n = min(int(len(pool[1]) / 0.8), int(len(pool[0]) / 0.5))
    positive = round(n * prior)
    return np.concatenate([pool[0][: n - positive], pool[1][:positive]])


def human_activity(archive):
    _, x, classes, groups, (base, a, b, test) = next(
        item for item in populations(archive) if item[0] == "har"
    )
    base = base[np.isin(classes[base], (3, 4))]
    cal = np.r_[a, b]
    cal = cal[np.isin(classes[cal], (3, 4))]
    test = test[np.isin(classes[test], (3, 4))]
    y = (classes == 4).astype(int)
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    classifier.fit(x[base], y[base])
    logits = classifier.decision_function(x)[:, None]
    calibrator = LogisticRegression(class_weight="balanced", max_iter=1000).fit(logits[cal], y[cal])
    scores = calibrator.decision_function(logits)
    ref = ReferenceScoreCDF(scores[cal][y[cal] == 0], scores[cal][y[cal] == 1])
    rows = []
    for person in np.unique(groups[test]):
        indices = test[groups[test] == person]
        early, late = temporal_pools(indices, y, np.random.default_rng([6411, int(person)]))
        for prior in (0.5, 0.8):
            adapt, trial = [choose_rows(pool, prior) for pool in (early, late)]
            for offset_sd in (0.0, 0.6):
                offset = offset_sd * ref.scale
                methods = estimates(ref, scores[adapt] + offset)
                methods["frozen"] = (0.5, 0, np.nan)
                methods["known_intervention"] = (prior, offset, np.nan)
                for method, (prior_hat, offset_hat, residual) in methods.items():
                    p = expit(scores[trial] + offset - offset_hat + logit(prior_hat))
                    rows.append(
                        dict(
                            person=int(person),
                            prior=prior,
                            offset_sd=offset_sd,
                            method=method,
                            adapt_n=len(adapt),
                            test_n=len(trial),
                            adapt_prior=y[adapt].mean(),
                            test_prior=y[trial].mean(),
                            prior_hat=prior_hat,
                            offset_hat_sd=offset_hat / ref.scale,
                            prior_abs_error=abs(prior_hat - y[adapt].mean()),
                            cdf_rmse=residual,
                            accuracy=np.mean((p >= 0.5) == y[trial]),
                            brier=np.mean((p - y[trial]) ** 2),
                        )
                    )
    return pd.DataFrame(rows)


def paired(frame, index, column, a, b):
    p = frame.pivot(index=index, columns="method", values=column)
    diff = p[a] - p[b]
    rad = t.ppf(0.975, len(diff) - 1) * diff.sem()
    return dict(reduction=diff.mean(), lo=diff.mean() - rad, hi=diff.mean() + rad)


def run(archive, out):
    start = time.perf_counter()
    began = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = synthetic()
    sim_time = time.perf_counter() - start
    people = human_activity(archive)
    for name, frame, by in [
        ("synthetic", sim, ["phase", "condition", "budget", "method"]),
        ("har", people, ["prior", "offset_sd", "method"]),
    ]:
        frame.to_csv(out / (name + ".csv"), index=False)
        frame.groupby(by).mean(numeric_only=True).drop(
            columns=["seed", "person"], errors="ignore"
        ).to_csv(out / (name + "_means.csv"))
    cfg = dict(
        primary=paired(
            sim.query('phase=="confirmation" and condition=="offset" and budget==1024'),
            "seed",
            "excess_brier",
            "mean",
            "cdf_joint",
        ),
        har_mean=paired(
            people.query("prior==.5 and offset_sd>.5"), "person", "brier", "mean", "cdf_joint"
        ),
        har_no_adaptation=paired(
            people.query("prior==.5 and offset_sd>.5"), "person", "brier", "frozen", "cdf_joint"
        ),
        synthetic_seconds=sim_time,
        compute_seconds=time.perf_counter() - start,
        compute_started_utc=began,
        compute_ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.har, args.out)
