"""Source-only choice between the native regularized tree and two arithmetic pools."""

from time import perf_counter

import numpy as np

from fusionbench.methods.auto_configuration import _folds, _inputs, _task
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI

ACTIONS = ("mean", "top_n", "native")
RULES = ("cv", "paired", "sign", "stress", "one_se")


def fit_action(x, y, action, *, seed=0):
    """Fit the chosen action on every supplied source row."""
    x, y, _ = _inputs(x, y, None)
    if action == "mean":
        model = SimpleAverage()
    elif action == "top_n":
        model = TopN(metric="brier")
    elif action == "native":
        model = RegularizedNativePatternOKAPI(
            seed=seed, split_seed=0, gp_validation_top_k=8
        )
    else:
        raise ValueError(f"Unknown action {action!r}")
    return model.fit(x, y, task=_task(x), n_classes=x.shape[2])


def _summarize(unit_loss, unit_fold, unit_disagreement):
    """Summarize paired Brier differences across equally weighted source units."""
    n = len(unit_fold)
    unit_stratum = (unit_disagreement > np.median(unit_disagreement)).astype(int)
    risk = {name: float(np.mean(unit_loss[name])) for name in ACTIONS}
    paired_se = {
        a: {
            b: float(np.std(unit_loss[a] - unit_loss[b], ddof=1) / np.sqrt(n))
            for b in ACTIONS
        }
        for a in ACTIONS
    }
    # Position in ACTIONS decides exact ties, with simpler procedures first.
    best_simple = min(ACTIONS[:2], key=risk.get)
    gain = unit_loss["native"] - unit_loss[best_simple]
    return {
        "n_units": n,
        "unit_loss": {name: unit_loss[name].tolist() for name in ACTIONS},
        "unit_fold": unit_fold.tolist(),
        "unit_disagreement": unit_disagreement.tolist(),
        "unit_stratum": unit_stratum.tolist(),
        "oof_brier": risk,
        "paired_se": paired_se,
        "best": min(ACTIONS, key=risk.get),
        "best_simple": best_simple,
        "simple_gain": float(gain.mean()),
        "simple_gain_by_fold": [
            float(gain[unit_fold == fold].mean()) for fold in range(3)
        ],
        "simple_gain_by_stratum": [
            float(gain[unit_stratum == side].mean()) for side in np.unique(unit_stratum)
        ],
        "fallback_reason": None,
    }


def source_evidence(x, y, groups=None, seed=0):
    """Measure all three actions using the same three out-of-fold source samples.

    A unit is a scene, or a row if no scene identifiers are available. The
    unit Brier loss averages the sum of squared class errors over its rows.
    The paired standard error describes variation over these units. Folds
    share training examples, so the one-SE thresholds are selection heuristics.
    """
    start = perf_counter()
    x, y, groups = _inputs(x, y, groups)
    units = (
        np.arange(len(y))
        if groups is None
        else np.unique(groups, return_inverse=True)[1]
    )
    counts = np.bincount(units)
    n = len(counts)
    if n < 3 or x.shape[1] < 2:
        return {
            "fallback_reason": "single_provider"
            if x.shape[1] < 2
            else "fewer_than_three_units",
            "n_units": n,
            "oof_fits": 0,
            "folds": [],
            "seconds": perf_counter() - start,
        }
    # Average squared distance of providers from the row's consensus. Neither
    # labels nor an action's own fitted predictions enter the stress strata.
    row_disagreement = (
        np.square(x - x.mean(axis=1, keepdims=True)).sum(axis=2).mean(axis=1)
    )
    disagreement = np.bincount(units, weights=row_disagreement) / counts
    loss = {name: np.zeros(n) for name in ACTIONS}
    unit_fold = np.full(n, -1, dtype=int)
    folds = []
    for fold, (train, held) in enumerate(_folds(y, groups)):
        held_units = np.unique(units[held])
        unit_fold[held_units] = fold
        targets = np.eye(x.shape[2])[y[held]]
        for name in ACTIONS:
            model = fit_action(x[train], y[train], name, seed=seed)
            p = model.predict(x[held])
            if p.shape != targets.shape or not np.isfinite(p).all():
                raise ValueError(f"Invalid out-of-fold predictions for {name}")
            row_loss = np.square(p - targets).sum(axis=1)
            loss[name] += np.bincount(units[held], weights=row_loss, minlength=n)
        folds.append(
            {
                "train_rows": len(train),
                "held_rows": len(held),
                "held_units": len(held_units),
            }
        )
    loss = {name: values / counts for name, values in loss.items()}
    evidence = _summarize(loss, unit_fold, disagreement)
    evidence.update(
        oof_fits=len(ACTIONS) * len(folds), folds=folds, seconds=perf_counter() - start
    )
    return evidence


def routes(evidence):
    """Apply five prespecified decision rules to the shared source measurements."""
    if evidence["fallback_reason"]:
        return dict.fromkeys(RULES, "mean")
    best, simple = evidence["best"], evidence["best_simple"]
    risk, se = evidence["oof_brier"], evidence["paired_se"]
    paired = evidence["simple_gain"] >= se["native"][simple]
    favorable_folds = all(gain > 0 for gain in evidence["simple_gain_by_fold"])
    nonworse_strata = all(gain >= 0 for gain in evidence["simple_gain_by_stratum"])
    return {
        "cv": best,
        "paired": simple if paired else "native",
        "sign": simple if paired and favorable_folds else "native",
        "stress": simple if paired and nonworse_strata else "native",
        "one_se": next(a for a in ACTIONS if risk[a] - risk[best] <= se[a][best]),
    }


def fit_source_choice(x, y, groups=None, *, rule, seed=0):
    """Choose once using source OOF losses and refit on the entire source bank.

    Pass a rule from RULES. Return the fitted FusionBench method, selected
    action name and source evidence. Its predict method accepts canonical
    probability tensors with shape [row, provider, class].
    """
    if rule not in RULES:
        raise ValueError(f"Expected a configuration rule from {RULES}")
    evidence = source_evidence(x, y, groups, seed=seed)
    action = routes(evidence)[rule]
    model = fit_action(x, y, action, seed=seed)
    return model, action, evidence
