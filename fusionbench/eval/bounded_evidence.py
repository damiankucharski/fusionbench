"""Learn whether to cap one raw sensor and compare with the exact Gaussian oracle."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from fusionbench.eval.search_stability import _predict_trees
from fusionbench.eval.seed_selection import search_champion
from fusionbench.scenarios.polarity import PolarityFault


def capped(x, j, threshold):
    if j is None:
        return x
    transformed = x.copy()
    transformed[:, j] = np.clip(x[:, j], -threshold, threshold)
    return transformed


def raw_cap_parameters(choice):
    _, channel, limit, model = choice
    return dict(channel=channel, limit=limit, weights=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]))


def predict_raw_cap(x, *, channel, limit, weights, intercept):
    """Evaluate the exported single raw-input network without provider models."""
    values = capped(np.asarray(x, dtype=float), channel, limit)
    return expit(values @ np.asarray(weights) + intercept)


def fit_clipped(x, y, val, labels):
    options = []
    for j in [None, *range(x.shape[1])]:
        for limit in [None] if j is None else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            model = LogisticRegression(C=1.0, max_iter=400).fit(capped(x, j, limit), y)
            prob = model.predict_proba(capped(val, j, limit))[:, 1]
            options.append((np.mean((prob - labels) ** 2), j, limit, model))

    def best(candidates):
        return min(candidates, key=lambda row: row[0])

    return {
        "linear": options[0],
        "learned_cap": best(options),
        "only_correct_cap": best([r for r in options if r[1] == 0]),
        "only_wrong_cap": best([r for r in options if r[1] == 1]),
    }


def run_study(out):
    start = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    compact_models = []
    with threadpool_limits(limits=1):
        for phase, seeds, gp_seeds in [
            ("development", range(1400, 1404), range(280, 288)),
            ("confirmation", range(1500, 1508), range(300, 308)),
        ]:
            for seed in seeds:
                for q in (0.0, 0.1):
                    for corr in (0.0, 0.65):
                        gen = PolarityFault(q, corr)
                        splits = [
                            gen._generate(n, np.random.default_rng([seed, i]))
                            for i, n in enumerate([768, 256, 16384])
                        ]
                        (p, y, l), (v, vy, lv), (s, sy, ls) = splits
                        x, vt, xt = l["raw"], lv["raw"], ls["raw"]
                        oracle = gen.bayes_oracle(s, ls)[:, 1]
                        choices = fit_clipped(x, y, vt, vy)
                        params = raw_cap_parameters(choices["learned_cap"])
                        compact_models.append(dict(phase=phase, seed=seed, fault_rate=q, correlation=corr, parameters=params))
                        trees = [search_champion(p, y, g, 10) for g in gp_seeds]
                        vpred = _predict_trees(trees, v)
                        gidx = np.argmin(((vpred - vy) ** 2).mean(1))
                        predictions = {
                            "okapi": _predict_trees([trees[gidx]], s)[0],
                            "oracle": oracle,
                            "ignore_fault": gen.posterior(s, fault_rate=0, correlation=corr),
                            "ignore_correlation": gen.posterior(s, fault_rate=q, correlation=0.0),
                            "ignore_both": gen.posterior(s, fault_rate=0.0, correlation=0.0),
                        }
                        for name, (_, j, lim, fit) in choices.items():
                            predictions[name] = fit.predict_proba(capped(xt, j, lim))[:, 1]
                        nn = []
                        with warnings.catch_warnings(record=True) as logs:
                            warnings.simplefilter("always", ConvergenceWarning)
                            for width in (4, 8):
                                for alpha in (1.0, 10.0):
                                    for init in (0, 1):
                                        net = MLPClassifier(
                                            hidden_layer_sizes=(width,),
                                            activation="tanh",
                                            solver="lbfgs",
                                            alpha=alpha,
                                            max_iter=400,
                                            random_state=seed * 20 + init,
                                        )
                                        net.fit(x, y)
                                        prob = net.predict_proba(vt)[:, 1]
                                        nn.append((np.mean((prob - vy) ** 2), width, alpha, init, net))
                        _, width, alpha, init, net = min(nn, key=lambda r: r[0])
                        predictions["mlp"] = net.predict_proba(xt)[:, 1]
                        for name, post in predictions.items():
                            j = choices[name][1] if name in choices else None
                            limit = choices[name][2] if name in choices else None
                            records.append(
                                dict(
                                    phase=phase,
                                    seed=seed,
                                    fault_rate=q,
                                    correlation=corr,
                                    model=name,
                                    excess_brier=np.mean((post - oracle) ** 2),
                                    brier=np.mean((post - sy) ** 2),
                                    accuracy=np.mean((post >= 0.5) == sy),
                                    channel=j,
                                    raw_limit=limit,
                                    logodds_cap=(
                                        abs(choices[name][3].coef_[0, j]) * limit
                                        if j is not None
                                        else None
                                    ),
                                    gp_seed=gp_seeds[gidx] if name == "okapi" else None,
                                    gp_nodes=trees[gidx].nodes_count if name == "okapi" else None,
                                    width=width if name == "mlp" else None,
                                    mlp_convergence_warnings=len(logs) if name == "mlp" else None,
                                )
                            )
                pd.DataFrame(records).to_csv(out / "models.csv", index=False)
    (out / "cap_models.json").write_text(json.dumps(compact_models, indent=2))
    df = pd.DataFrame(records)
    df.groupby(["phase", "fault_rate", "correlation", "model"])[
        ["excess_brier", "brier", "accuracy"]
    ].mean().to_csv(out / "means.csv")
    p = df.query("phase=='confirmation' and fault_rate==.1 and correlation==.65").pivot(
        index="seed", columns="model", values="excess_brier"
    )
    d = p.linear - p.learned_cap
    rad = t.ppf(0.975, len(d) - 1) * d.sem()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                primary=dict(reduction=d.mean(), lo=d.mean() - rad, hi=d.mean() + rad),
                train=768,
                selection=256,
                test=16384,
                thresholds=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                compute_seconds=time.perf_counter() - start,
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
