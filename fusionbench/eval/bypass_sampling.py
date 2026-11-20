"""Can decision changes under an OKAPI stage bypass guide supervised data selection?

All four encodings of a selected commutative two-operator formula are evaluated.
Selection sees expert probabilities on an unlabeled pool. A fresh MLP then sees
only the selected raw examples and their ground-truth labels.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def sample(rng, size, prevalence):
    factors = rng.random((size, 3)) < prevalence
    # Two sensor readings per factor and two nuisance coordinates, all available
    # to the student. The experts were built for separate pairs of sensors.
    signal = np.repeat(2 * factors.astype(float) - 1, 2, axis=1)
    x = np.column_stack((signal + rng.normal(0, 1.05, signal.shape),
                         rng.normal(size=(size, 2))))
    primary, rescue, safe = factors.T
    return x, (primary | rescue) & safe, factors


def expert_predictions(experts, x):
    return np.column_stack([expert.predict_proba(x[:, 2*i:2*i+2])[:, 1]
                            for i, expert in enumerate(experts)])


def program(scores, pair, outer_id, ops, reverse_inner=False, reverse_outer=False):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    operators = {"min": MinNode, "max": MaxNode}
    def leaf(index):
        return ValueNode(None, scores[:, index], index)

    left, right = pair[::-1] if reverse_inner else pair
    inner = leaf(left)
    inner.add_child(operators[ops[0]]([leaf(right)]))
    outer = leaf(outer_id)
    if reverse_outer:
        outer.add_child(operators[ops[1]]([inner]))
        return Tree(outer)
    inner.add_child(operators[ops[1]]([outer]))
    return Tree(inner)


def choose_formula(scores, labels):
    candidates = []
    for pair in combinations(range(3), 2):
        outer, = set(range(3)).difference(pair)
        for ops in product(("min", "max"), repeat=2):
            pred = program(scores, pair, outer, ops).predict()
            candidates.append((np.mean((pred - labels) ** 2), pair, outer, ops))
    _, pair, outer, ops = min(candidates)
    return pair, outer, ops


def stratified_order(signatures, random_order):
    """Equal allocations across observed signatures, then redistribute shortfalls."""
    signatures = signatures[random_order]
    _, groups = np.unique(signatures, axis=0, return_inverse=True)
    ranks = np.empty(len(groups), dtype=int)
    for group in np.unique(groups):
        mask = groups == group
        ranks[mask] = np.arange(mask.sum())
    return random_order[np.argsort(ranks, kind="stable")]


def policies(scores, formula, order):
    from okapi.interventions import operator_paths, predict_with_bypass

    signatures = []
    rankings = {"random": order,
                "vote_patterns": stratified_order(scores >= 0.5, order),
                "disagreement": stratified_order((scores >= 0.5).any(1) != (scores >= 0.5).all(1), order)}
    intact = None
    for inner, outer in product((False, True), repeat=2):
        tree = program(scores, *formula, inner, outer)
        pred = tree.predict()
        if intact is not None:
            np.testing.assert_array_equal(pred, intact)
        intact = pred
        changes = np.column_stack([(predict_with_bypass(tree, [path]) >= 0.5) != (pred >= 0.5)
                                   for path, _ in operator_paths(tree)])
        signatures.append(changes)
        rankings[f"bypass_{int(inner)}{int(outer)}"] = stratified_order(changes, order)
    rankings["all_encodings"] = stratified_order(np.column_stack(signatures), order)
    rankings["uncertainty"] = order[np.argsort(np.abs(intact[order] - 0.5), kind="stable")]
    # Half random exploration mitigates concentration near a single boundary.
    uncertain = rankings["uncertainty"]
    # Interleave both lists, keeping the first occurrence of each pool example.
    mixed = np.column_stack((order, uncertain)).ravel()
    _, first = np.unique(mixed, return_index=True)
    rankings["mixed_uncertainty"] = mixed[np.sort(first)]
    return rankings


def run_study(out, seeds, budgets=(96, 256)):
    from threadpoolctl import threadpool_limits

    if not seeds or not budgets or min(budgets) < 1 or max(budgets) > 1024:
        raise ValueError("Supply seeds and budgets between 1 and 1024")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        source, calibration, pool_rng, iid, balanced, lottery = [
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
        x0, _, targets = sample(source, 1024, [.5]*3)
        experts = [LogisticRegression().fit(x0[:, 2*i:2*i+2], targets[:, i]) for i in range(3)]
        xc, yc, cal_factors = sample(calibration, 1024, [.5]*3)
        formula = choose_formula(expert_predictions(experts, xc), yc)
        prevalence = [.48, .045, .9]
        x, y, factors = sample(pool_rng, 8192, prevalence)
        ranks = policies(expert_predictions(experts, x), formula, lottery.permutation(len(y)))
        tests = {"pool_like": sample(iid, 8192, prevalence),
                 "balanced": sample(balanced, 8192, [.5]*3)}
        for budget in budgets:
            # Reusing balanced source calibration labels is possible when retained.
            choices = {policy: (x, y, factors, order) for policy, order in ranks.items()}
            choices["source_labels"] = (xc, yc, cal_factors, np.arange(len(yc)))
            for policy, (available_x, available_y, available_factors, order) in choices.items():
                subset = order[:budget]
                # Same training order for identical sets and same model initialisation.
                subset = np.sort(subset)
                with threadpool_limits(limits=1):
                    net = MLPClassifier(hidden_layer_sizes=(12,), solver="lbfgs",
                                        activation="tanh", alpha=1., random_state=seed,
                                        max_iter=500).fit(available_x[subset], available_y[subset])
                for domain, (xt, yt, factors_t) in tests.items():
                    pred = net.predict(xt)
                    oracle_pred = program(expert_predictions(experts, xt), *formula).predict() >= .5
                    primary, rescue, safe = factors_t.T
                    rare = ~primary & rescue & safe
                    veto = (primary | rescue) & ~safe
                    core_train, rescue_train, safe_train = available_factors[subset].T
                    rows.append(dict(seed=seed, budget=budget, policy=policy, domain=domain,
                                     formula=str(formula), accuracy=np.mean(pred == yt),
                                     rescue_recall=pred[rare].mean(),
                                     veto_specificity=(~pred[veto]).mean(),
                                     ensemble_accuracy=np.mean(oracle_pred == yt),
                                     n_rescue=int((~core_train & rescue_train & safe_train).sum()),
                                     n_veto=int(((core_train | rescue_train) & ~safe_train).sum())))
    result = pd.DataFrame(rows)
    result.to_csv(out / "bypass_sampling.csv", index=False)
    return result
