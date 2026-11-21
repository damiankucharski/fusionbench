"""Reproduce the two-provider calibrated sign-reversal study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.correlated import CalibratedSuppressor


class NonnegativeLogit(StackingLogit):
    """Match StackingLogit's redundant two-class design and C=1 penalty."""

    name = "nonnegative_logit"

    def fit(self, fit_preds, fit_y, *, task, n_classes):
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(self._features(fit_preds))
        y = np.asarray(fit_y)

        def objective(params):
            weights = params[:-1]
            z = x @ weights + params[-1]
            residual = expit(z) - y
            loss = np.logaddexp(0., (1 - 2 * y) * z).sum() + 0.5 * weights @ weights
            gradient = np.r_[x.T @ residual + weights, residual.sum()]
            return loss, gradient

        # Each pair consists of opposite standardized columns. Bounding their
        # signs enforces a nonnegative effective slope without changing the ridge.
        bounds = [(None, 0.), (0., None)] * fit_preds.shape[1] + [(None, None)]
        fit = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, bounds=bounds,
                       method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-8})
        if not fit.success:
            raise RuntimeError(fit.message)
        self.params = fit.x
        return self

    def predict(self, preds):
        z = self.scaler.transform(self._features(preds)) @ self.params[:-1] + self.params[-1]
        return np.column_stack((expit(-z), expit(z)))


def coefficients(model):
    if isinstance(model, NonnegativeLogit):
        w = model.params[:-1] / model.scaler.scale_
    else:
        scale, logistic = model.model.steps
        w = logistic[1].coef_[0] / scale[1].scale_
    return w[1::2] - w[::2]


def run_bank(seed, corr):
    law = CalibratedSuppressor(corr=corr)
    shifted = CalibratedSuppressor(corr=0. if corr == .85 else .85)
    streams = np.random.SeedSequence(seed).spawn(5)
    order = np.random.default_rng(streams[0]).permutation(2)
    fit_p, fit_y, fit_latent = law._generate(384, np.random.default_rng(streams[1]))
    fit_input = fit_p[:, order]
    models = [SimpleAverage(), TopN(metric="brier"), StackingLR(), StackingLogit(),
              NonnegativeLogit()]
    choices = []
    for model in models:
        start = perf_counter()
        model.fit(fit_input, fit_y, task="binary", n_classes=2)
        elapsed = perf_counter() - start
        row = {"seed": seed, "rho": corr, "method": model.name, "fit_seconds": elapsed,
               "first_source": int(order[0]), "n_selected": getattr(model, "n_selected_", None)}
        if isinstance(model, StackingLogit):
            w = coefficients(model)[np.argsort(order)]
            row.update(strong_weight=w[0], weak_weight=w[1])
        choices.append(row)

    splits = [("fit", law, (fit_p, fit_y, fit_latent))]
    for i, (name, domain, n) in enumerate([("val", law, 256), ("iid", law, 8192),
                                           ("shift", shifted, 8192)], start=2):
        splits.append((name, domain, domain._generate(n, np.random.default_rng(streams[i]))))
    scores = []
    for split, domain, (p, y, latent) in splits:
        predicted = {m.name: m.predict(p[:, order])[:, 1] for m in models}
        predicted["strong_source"] = p[:, 0, 1]
        predicted["bayes"] = domain.bayes_oracle(p, latent)[:, 1]
        for method, q in predicted.items():
            scores.append({"seed": seed, "rho": corr, "split": split, "method": method,
                           "brier": np.mean((q - y) ** 2)})
    return scores, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    start = perf_counter()
    scores, fits = [], []
    for seed in range(41000, 41024):
        for corr in [0., .5, .85]:
            measured, choices = run_bank(seed, corr)
            scores.extend(measured)
            fits.extend(choices)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(args.out / "scores.csv", index=False)
    pd.DataFrame(fits).to_csv(args.out / "fits.csv", index=False)
    timing = {"elapsed_seconds": perf_counter() - start}
    (args.out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
