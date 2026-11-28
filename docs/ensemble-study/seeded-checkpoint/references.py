"""Deployed fusion references for seeded binary checkpoint banks."""

from numbers import Integral

import numpy as np
from sklearn.metrics import f1_score

from fusionbench.eval.metrics import brier_score
from fusionbench.eval.suite_point_gp import _full
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR


def _matches(fitted, candidate):
    return all(
        np.allclose(observed, expected, rtol=0, atol=1e-12)
        for observed, expected in zip(fitted, candidate, strict=True)
    )


def score_references(
    fit_x, fit_y, audit_x, audit_y, *, seed, audit_q, signal_provider=None
):
    """Fit on the supplied checkpoint columns and compare frozen predictions.

    Binary inputs can be (rows, providers) positive probabilities or canonical
    (rows, providers, 2) scores. audit_q contains one latent positive-class
    probability per audit row. selected_is_* tests equality on all fit AND
    audit inputs without using their labels. n_providers counts distinct inputs
    referenced by the deployed computation, including zero-weight tree leaves.
    """
    fit_x, audit_x = _full(fit_x), _full(audit_x)
    if fit_x.shape[-1] != 2 or fit_x.shape[1:] != audit_x.shape[1:]:
        raise ValueError(
            "Supply binary fit and audit banks with matching provider axes"
        )
    n_providers = fit_x.shape[1]
    if signal_provider is not None and (
        not isinstance(signal_provider, Integral)
        or isinstance(signal_provider, (bool, np.bool_))
        or not 0 <= signal_provider < n_providers
    ):
        raise ValueError("signal_provider must identify a provider in the bank")
    q = np.asarray(audit_q, dtype=float)
    if q.shape != (len(audit_x),) or not np.all((q >= 0) & (q <= 1)):
        raise ValueError(
            "audit_q must contain one finite positive-class probability per audit row"
        )

    models = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
        "regularized": RegularizedNativePatternOKAPI(seed=seed, split_seed=seed),
    }
    for model in models.values():
        model.fit(fit_x, fit_y, task="binary", n_classes=2)

    banks = (fit_x, audit_x)
    arithmetic = [x[:, :, 1].mean(axis=1) for x in banks]
    signal = (
        [x[:, signal_provider, 1] for x in banks]
        if signal_provider is not None
        else None
    )
    results = {}
    for name, model in models.items():
        predictions = [model.predict(x) for x in banks]
        positive = [p[:, 1] for p in predictions]
        row = {
            "audit_excess": float(2.0 * np.square(positive[1] - q).mean()),
            "selected_is_mean": _matches(positive, arithmetic),
            "selected_is_signal": _matches(positive, signal)
            if signal is not None
            else None,
            "n_providers": n_providers,
        }
        for split, y, p in zip(
            ("fit", "audit"), (fit_y, audit_y), predictions, strict=True
        ):
            row[f"{split}_brier"] = brier_score(y, p, n_classes=2)
            row[f"{split}_f1"] = float(
                f1_score(y, p[:, 1] > 0.5, pos_label=1, zero_division=0)
            )
        if name == "top_n":
            row["n_selected"] = row["n_providers"] = int(model.n_selected_)
        elif name == "regularized":
            row["family"] = model.chosen_
            row["n_providers"] = len(model.trees_[model.chosen_].unique_value_node_ids)
        results[name] = row
    return results
