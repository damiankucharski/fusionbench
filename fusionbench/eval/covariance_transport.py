"""Test whether unlabeled target moments transport stable Gaussian sensor weights and GP seed ranks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.special import expit, logit
from scipy.stats import spearmanr, t
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import search_champion
from fusionbench.scenarios.polarity import PolarityFault

FAULT = 0.1


@dataclass(frozen=True)
class Population:
    correlation: float = 0.65
    prior: float = 0.5
    stable_offset: float = 0.0
    anchor_offset: float = 0.0


CONDITIONS = {
    "iid": Population(),
    "decorrelated": Population(0.0),
    "stronger_correlation": Population(0.9),
    "prior": Population(prior=0.8),
    "joint": Population(0.0, 0.8),
    "stable_offset": Population(stable_offset=0.6),
    "all_offset": Population(stable_offset=0.6, anchor_offset=0.6),
}


def draw(seed, split, n, population, strengths):
    rng = np.random.default_rng([seed, split])
    y = rng.uniform(size=n) < population.prior
    common = rng.normal(size=(n, 1))
    noise = rng.normal(size=(n, len(strengths)))
    rho = population.correlation
    noise[:, 1:] = np.sqrt(rho) * common + np.sqrt(1 - rho) * noise[:, 1:]
    raw = (2 * y - 1)[:, None] * strengths + noise
    raw[rng.uniform(size=n) < FAULT, 0] *= -1
    raw[:, 1:] += population.stable_offset * strengths[1:]
    raw[:, 0] += population.anchor_offset * (1 - 2 * FAULT) * strengths[0]
    logits = 2 * strengths * raw
    return raw, np.stack([expit(-logits), expit(logits)], axis=-1), y.astype(int)


def stable_weights(s, correlation):
    covariance = np.full((len(s), len(s)), correlation, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    return 2 * np.linalg.solve(covariance, s)


def posterior(raw, strengths, population):
    p = population
    anchor = raw[:, 0] - p.anchor_offset * (1 - 2 * FAULT) * strengths[0]
    unstable = logit(FAULT + (1 - 2 * FAULT) * expit(2 * strengths[0] * anchor))
    stable = raw[:, 1:] - p.stable_offset * strengths[1:]
    return expit(logit(p.prior) + unstable + stable @ stable_weights(strengths[1:], p.correlation))


def projected_correlation(x, s):
    """Contrast directions remove class, along-s offsets and common-mode noise."""
    basis = null_space(np.stack([s, np.ones_like(s)]))
    return float(np.clip(1 - np.mean((x @ basis) ** 2), 0.0, 0.98))


def source_parameters(x, y):
    s = np.mean((2 * y - 1)[:, None] * x, axis=0)
    s[0] /= 1 - 2 * FAULT
    return s, Population(projected_correlation(x[:, 1:], s[1:]), y.mean())


def fit_populations(x, s, frozen):
    stable = x[:, 1:]
    n = len(x)
    k = stable.shape[1]
    conditional_second = stable.T @ stable / n - np.outer(s[1:], s[1:])
    rho_moment = np.clip(np.mean(conditional_second[np.triu_indices(k, k=1)]), 0.0, 0.98)
    rho_project = projected_correlation(stable, s[1:])
    mean_along_s = stable.mean(axis=0) @ s[1:] / (s[1:] @ s[1:])
    mean_anchor = x[:, 0].mean() / ((1 - 2 * FAULT) * s[0])
    stable_prior = np.clip((1 + mean_along_s) / 2, 0.02, 0.98)
    anchor_prior = np.clip((1 + mean_anchor) / 2, 0.02, 0.98)
    return {
        "frozen": frozen,
        "second_moment": Population(rho_moment, stable_prior),
        "projected": Population(rho_project, stable_prior),
        "anchor": Population(rho_project, anchor_prior),
        "anchor_offset": Population(rho_project, anchor_prior, mean_along_s - (2 * anchor_prior - 1)),
    }


def seed_metrics(risk, ranking):
    best = np.isclose(risk, min(risk), rtol=0, atol=1e-14)
    if ranking is None:
        return risk.mean() - min(risk), np.nan, best.mean()
    tied = np.isclose(ranking, min(ranking), rtol=0, atol=1e-14)
    rho = spearmanr(ranking, risk).statistic if np.ptp(ranking) and np.ptp(risk) else np.nan
    return risk[tied].mean() - min(risk), rho, best[tied].mean()


def run_study(directory):
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    truth = PolarityFault().strengths
    models, selectors, candidates = [], [], []
    for phase, bank_seeds, gp_seeds in [
        ("development", range(1600, 1604), range(320, 328)),
        ("confirmation", range(1700, 1708), range(340, 348)),
    ]:
        for seed in bank_seeds:
            x, train, y = draw(seed, 0, 768, CONDITIONS["iid"], truth)
            _, valid, vy = draw(seed, 1, 256, CONDITIONS["iid"], truth)
            s, frozen = source_parameters(x, y)
            trees = [search_champion(train, y, g, 10) for g in gp_seeds]
            validation_loss = ((_predict_trees(trees, valid) - vy) ** 2).mean(axis=1)
            for name, population in CONDITIONS.items():
                x, adapt, _ = draw(seed, 2, 1024, population, truth)
                test_x, test, labels = draw(seed, 3, 8192, population, truth)
                test_pred = _predict_trees(trees, test)
                adapt_pred = _predict_trees(trees, adapt)
                oracle = posterior(test_x, truth, population)
                candidate_risk = ((test_pred - oracle) ** 2).mean(axis=1)
                for i, gp in enumerate(gp_seeds):
                    candidates.append(
                        dict(
                            phase=phase,
                            seed=seed,
                            condition=name,
                            gp_seed=gp,
                            excess_brier=candidate_risk[i],
                            source_brier=validation_loss[i],
                        )
                    )
                for n in (256, 1024):
                    keys = dict(phase=phase, seed=seed, condition=name, budget=n)
                    fits = fit_populations(x[:n], s, frozen)
                    fits["oracle"] = population
                    ranks = {
                        "source_validation": validation_loss,
                        "random": None,
                        "hindsight": candidate_risk,
                    }
                    for method, fit in fits.items():
                        strengths = truth if method == "oracle" else s
                        pred = posterior(test_x, strengths, fit)
                        surrogate = posterior(x[:n], strengths, fit)
                        ranks[method] = ((adapt_pred[:, :n] - surrogate) ** 2).mean(axis=1)
                        weights = stable_weights(strengths[1:], fit.correlation)
                        actual_sign = np.sign(stable_weights(truth[1:], population.correlation))
                        models.append(
                            dict(
                                **keys,
                                method=method,
                                excess_brier=np.mean((pred - oracle) ** 2),
                                accuracy=np.mean((pred >= 0.5) == labels),
                                estimated_rho=fit.correlation,
                                estimated_prior=fit.prior,
                                estimated_offset=fit.stable_offset,
                                sign_match=np.mean(np.sign(weights) == actual_sign),
                            )
                        )
                    for method, ranks_by_seed in ranks.items():
                        regret, rho, top = seed_metrics(candidate_risk, ranks_by_seed)
                        selectors.append(
                            dict(**keys, method=method, regret=regret, spearman=rho, top=top)
                        )
    m = pd.DataFrame(models)
    r = pd.DataFrame(selectors)
    for frame, name in [(m, "models"), (r, "selectors"), (pd.DataFrame(candidates), "candidates")]:
        frame.to_csv(out / (name + ".csv"), index=False)
    for frame, name in [(m, "models"), (r, "selectors")]:
        frame.groupby(["phase", "condition", "budget", "method"]).mean(numeric_only=True).drop(
            columns="seed"
        ).to_csv(out / (name + "_means.csv"))
    p = m.query("phase=='confirmation' and condition=='joint' and budget==1024").pivot(
        index="seed", columns="method", values="excess_brier"
    )
    diff = p.frozen - p.projected
    rad = t.ppf(0.975, len(diff) - 1) * diff.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                primary_reduction=diff.mean(),
                lo=diff.mean() - rad,
                hi=diff.mean() + rad,
                strengths_source="estimated; unstable channel identity and q=.1 known",
                search_count=96,
                source_train=768,
                source_validation=256,
                adapt_budgets=[256, 1024],
                test=8192,
                compute_seconds=time.perf_counter() - start,
                compute_started_utc=started,
                compute_ended_utc=datetime.now(timezone.utc).isoformat(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    with threadpool_limits(limits=1):
        run_study(parser.parse_args().out)
