"""Source-only configuration of fusion on a fixed prediction bank.

Inputs are soft predictions [row, provider, class]. Selection compares full-class
Brier losses (the sum of squared class errors, including both binary channels).
Every deployable configuration refits on all rows after the choice is made.
"""

from collections.abc import Mapping
from copy import copy
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from fusionbench.eval.suite_point_gp import _bind, _full, _targets, predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.base import Method
from fusionbench.methods.native_pattern import _brier_skill, _full_probability, _head
from fusionbench.methods.okapi import OKAPIMethod

ACTIONS = ("mean", "top_n", "cal_mean", "flat", "cal_pull")
_FOLD_SEED = 3107


def _inputs(x, y, groups):
    x = _full(x)
    if not np.all((x >= 0) & (x <= 1)) or not np.allclose(
        x.sum(2), 1, rtol=0, atol=1e-6
    ):
        raise ValueError(
            "Each provider must supply finite probabilities summing to one"
        )
    y, _ = _targets(y, len(x), x.shape[-1])
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("Supply one group identifier per source row")
        if np.issubdtype(groups.dtype, np.number) and not np.isfinite(groups).all():
            raise ValueError("Supply finite group identifiers")
    return x, y, groups


def _task(x):
    return "binary" if x.shape[-1] == 2 else "multiclass"


class _FlatLogit(Method):
    """Native variance-penalized, derivative-free logit head with no basis search."""

    def fit(self, x, y, *, task, n_classes):
        from okapi.globals import set_backend

        super().fit(x, y, task=task, n_classes=n_classes)
        set_backend("numpy")
        self.input_shape_ = x.shape[1:]

        def objective(p, labels):
            return _brier_skill(_full_probability(p), labels)

        self.best_tree, self.fit_info_ = _head(
            _bind(x), y, (), objective, n_classes > 2, 0.25
        )
        return self

    def predict(self, preds):
        x = _full(preds)
        if x.shape[1:] != self.input_shape_:
            raise ValueError("Query provider and class counts must match training")
        return predict_tree(self.best_tree, x)


class _PredictionPullGP(OKAPIMethod):
    """Bounded native search seeded at, and penalized by distance to, the mean."""

    def __init__(self, n_providers):
        super().__init__(
            seed=0,
            generations=8,
            n_restarts=1,
            n_seeds=0,
            population_size=8,
            population_multiplier=2,
            tournament_size=3,
            max_nodes=max(31, n_providers + 1),
            max_depth=7,
            fitness="brier",
        )

    def _seed_population(self, okapi, rng):
        from okapi.paired_fitness import arithmetic_mean_tree
        from okapi.prediction_pull import PredictionPullBrier

        super()._seed_population(okapi, rng)
        labels = okapi.gt_tensor
        seed = arithmetic_mean_tree(okapi.train_tensors)
        okapi.objective_functions = (
            PredictionPullBrier(
                seed.predict(), labels, strength=np.sqrt(8 / len(labels))
            ),
        )
        okapi.population.append(seed)
        okapi.population_size += 1
        okapi.fitnesses = None


def _calibrated(base, x, y, groups):
    from okapi.globals import set_backend
    from okapi.paired_fitness import arithmetic_mean_tree
    from okapi.temperature import fit_brier_temperature

    set_backend("numpy")
    if isinstance(base, SimpleAverage):
        adapter = OKAPIMethod()
        adapter.n_classes = x.shape[-1]
        adapter.task = _task(x)
        values = adapter._to_okapi(np.asarray(x, dtype=float))
        tree = arithmetic_mean_tree(
            {f"m{i}.npy": values[:, i] for i in range(x.shape[1])}
        )
    else:
        adapter = copy(base)
        tree = base.best_tree
    tree, info = fit_brier_temperature(
        tree, base.predict(x), y, groups=groups, penalized=True
    )
    if isinstance(base, SimpleAverage) and info["beta"] == 1:
        adapter = copy(base)
    adapter.best_tree = tree
    adapter.best_nodes = tree.nodes_count
    adapter.calibration_ = info
    return adapter


def fit_actions(x, y, groups=None, actions=ACTIONS):
    """Fit requested configurations on all source rows and return predictors by name."""
    x, y, groups = _inputs(x, y, groups)
    actions = tuple(actions)
    unknown = set(actions) - set(ACTIONS) - {"pull_mean"}
    if unknown:
        raise ValueError(f"Unknown fusion configurations: {sorted(unknown)}")
    options = {"task": _task(x), "n_classes": x.shape[-1]}
    models = {}
    if "mean" in actions or "cal_mean" in actions:
        models["mean"] = SimpleAverage().fit(x, y, **options)
    if "top_n" in actions:
        models["top_n"] = TopN(metric="brier").fit(x, y, **options)
    if "flat" in actions:
        models["flat"] = _FlatLogit().fit(x, y, **options)
    if "cal_mean" in actions:
        models["cal_mean"] = _calibrated(models["mean"], x, y, groups)
    if "cal_pull" in actions or "pull_mean" in actions:
        if x.shape[1] < 2:
            raise ValueError(
                "Native prediction-pull search requires at least two providers"
            )
        models["pull_mean"] = _PredictionPullGP(x.shape[1]).fit(x, y, **options)
        if "cal_pull" in actions:
            models["cal_pull"] = _calibrated(models["pull_mean"], x, y, groups)
    return {name: models[name] for name in actions}


def _folds(y, groups):
    """Split whole groups when supplied. Otherwise stratify if every label permits."""
    if groups is not None:
        _, units = np.unique(groups, return_inverse=True)
        order = np.random.default_rng(_FOLD_SEED).permutation(units.max() + 1)
        for held in np.array_split(order, 3):
            mask = np.isin(units, held)
            yield np.flatnonzero(~mask), np.flatnonzero(mask)
    elif np.unique(y, return_counts=True)[1].min() >= 3:
        splitter = StratifiedKFold(3, shuffle=True, random_state=_FOLD_SEED)
        yield from splitter.split(np.arange(len(y)), y)
    else:
        splitter = KFold(3, shuffle=True, random_state=_FOLD_SEED)
        yield from splitter.split(np.arange(len(y)))


def cv_choice(x, y, groups=None):
    """Choose the first action within one paired standard error of the OOF winner.

    Group loss is the mean row loss in one group. Grouped comparisons weight
    groups equally; otherwise each row is one unit. The standard error is a
    selection heuristic on shared folds, not an independent-sample interval.
    Fewer than three units or fewer than two providers selects the simple mean.
    """
    start = perf_counter()
    x, y, groups = _inputs(x, y, groups)
    units = (
        np.arange(len(y))
        if groups is None
        else np.unique(groups, return_inverse=True)[1]
    )
    counts = np.bincount(units)
    n_units = len(counts)
    if n_units < 3 or x.shape[1] < 2:
        reason = "fewer_than_three_units" if n_units < 3 else "single_provider"
        return "mean", {
            "best": None,
            "eligible": ["mean"],
            "oof_brier": {},
            "paired_se": {},
            "n_units": n_units,
            "folds": [],
            "oof_fits": 0,
            "fit_seconds": perf_counter() - start,
            "fallback_reason": reason,
        }
    unit_losses = np.zeros((len(ACTIONS), n_units), dtype=float)
    folds = []
    for train, held in _folds(y, groups):
        models = fit_actions(
            x[train], y[train], None if groups is None else groups[train]
        )
        truth = np.eye(x.shape[-1])[y[held]]
        for i, name in enumerate(ACTIONS):
            p = models[name].predict(x[held])
            loss = np.square(p - truth).sum(axis=1)
            unit_losses[i] += np.bincount(units[held], weights=loss, minlength=n_units)
        folds.append(
            {
                "train_rows": len(train),
                "held_rows": len(held),
                "held_units": len(np.unique(units[held])),
                "train_classes": np.unique(y[train]).tolist(),
            }
        )
    unit_losses /= counts
    if not np.isfinite(unit_losses).all():
        raise ValueError("Nonfinite out-of-fold Brier loss")
    risks = unit_losses.mean(axis=1)
    best = np.argmin(risks)
    differences = unit_losses - unit_losses[best]
    paired_se = differences.std(axis=1, ddof=1) / np.sqrt(n_units)
    eligible = [
        name
        for i, name in enumerate(ACTIONS)
        if risks[i] - risks[best] <= paired_se[i] + 1e-12
    ]
    return eligible[0], {
        "best": ACTIONS[best],
        "eligible": eligible,
        "oof_brier": dict(zip(ACTIONS, risks.tolist())),
        "paired_se": dict(zip(ACTIONS, paired_se.tolist())),
        "n_units": n_units,
        "folds": folds,
        "oof_fits": len(folds) * len(ACTIONS),
        "fit_seconds": perf_counter() - start,
    }


class ConfiguredOKAPI(Method):
    """Fit one configuration chosen by source CV or a supplied OKAPI policy mapping.

    Supply the string 'source_cv' or a JSON-decoded policy returned by
    okapi.configuration.fit_configuration_policy. A source bank provides only
    probabilities, labels and optional group identifiers. After fit, inspect
    action_ and selection_ to see the choice; predict uses the refitted model.
    """

    name = "configured_okapi"

    @classmethod
    def available(cls):
        try:
            from okapi.configuration import recommend_for_pool  # noqa: F401
            from okapi.pattern_logit import fit_pattern_tree  # noqa: F401
            from okapi.pool_diagnostics import pool_fingerprint  # noqa: F401
            from okapi.prediction_pull import PredictionPullBrier  # noqa: F401
            from okapi.temperature import fit_brier_temperature  # noqa: F401
        except ImportError:
            return False
        return OKAPIMethod.available()

    def __init__(self, policy):
        if not isinstance(policy, Mapping) and not (
            isinstance(policy, str) and policy == "source_cv"
        ):
            raise ValueError(
                "Supply 'source_cv' or a fitted OKAPI configuration policy mapping"
            )
        self.policy = policy

    def fit(self, fit_preds, fit_y, *, task, n_classes, groups=None):
        x, y, groups = _inputs(fit_preds, fit_y, groups)
        if x.shape[-1] != n_classes or task != _task(x):
            raise ValueError("Class axis must match the supplied classification task")
        if self.policy == "source_cv":
            action, details = cv_choice(x, y, groups)
        else:
            from okapi.configuration import recommend_for_pool

            details = recommend_for_pool(x, y, self.policy, groups=groups)
            action = details["action"]
        if action not in ACTIONS:
            raise ValueError(f"Unsupported selected action {action!r}")
        model = fit_actions(x, y, groups, actions=(action,))[action]
        super().fit(x, y, task=task, n_classes=n_classes)
        self.action_, self.selection_, self.model_ = action, details, model
        self.input_shape_ = x.shape[1:]
        return self

    def predict(self, preds):
        if not hasattr(self, "model_"):
            raise ValueError("Fit ConfiguredOKAPI before prediction")
        x = _full(preds)
        if x.shape[1:] != self.input_shape_:
            raise ValueError("Query provider and class counts must match training")
        return self.model_.predict(x)
