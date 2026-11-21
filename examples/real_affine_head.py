"""Out-of-fold portability test for signed binary OKAPI heads (frozen WDBC protocol).

Ting and Witten (1999) trained a linear combiner on cross-validated class
probabilities. Here the same out-of-fold principle supplies inputs for
regularized logistic stacking and a signed OKAPI head with a maximum feature.
See ting1999stacked in docs/ensemble-study/literature/README.md.
"""
import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.stacking import StackingLR, StackingLogit
from okapi.affine import stacking_tree
from okapi.node import MaxNode, ValueNode

BANKS = (87300, 87301, 87302)
PROVIDERS = ("lr", "knn", "naive_bayes", "forest", "boost", "lda")


def base_scores(X_train, y_train, X_query, seed):
    estimators = [make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000)),
                  make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
                  GaussianNB(), RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                       random_state=seed, n_jobs=1),
                  HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, l2_regularization=1.,
                                                 early_stopping=False, random_state=seed),
                  LinearDiscriminantAnalysis()]
    scores = []
    for est in estimators:
        est.fit(X_train, y_train)
        np.testing.assert_array_equal(est.classes_, [0, 1])
        scores.append(est.predict_proba(X_query))
    return np.stack(scores, axis=1)


def cross_fit(X, y, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.empty((len(y), len(PROVIDERS), 2))
    filled = np.zeros(len(y), int)
    for fold, (train, val) in enumerate(folds.split(X, y)):
        assert not np.intersect1d(train, val).size
        p[val] = base_scores(X[train], y[train], X[val], seed+fold)
        filled[val] += 1
    np.testing.assert_array_equal(filled, np.ones(len(y)))
    return p


def augmented(preds, pair):
    if pair is None:
        return preds
    a, b = pair
    p = np.maximum(preds[:, a, 1], preds[:, b, 1])
    return np.concatenate((preds, np.stack([1-p, p], axis=-1)[:, None]), axis=1)


def make_tree(preds, stack, transform, pair):
    roots = [ValueNode(None, preds[:, i, 1], i) for i in range(preds.shape[1])]
    if pair is not None:
        a, b = pair
        operand = roots[a].copy_subtree()
        operand.add_child(MaxNode([roots[b].copy_subtree()]))
        roots.append(operand)
    return stacking_tree(roots, stack.model, transform=transform)


def fit_bank(preds, y):
    timing = []
    references = {
        "mean": SimpleAverage(), "top_n": TopN("brier"),
        "ordinary_gp": OKAPIMethod(seed=61, n_seeds=10, n_restarts=1,
                                   max_nodes=11, generations=16, fitness="brier"),
    }
    for name, model in references.items():
        start = perf_counter()
        model.fit(preds, y, task="binary", n_classes=2)
        timing.append(dict(model=name, seconds=perf_counter()-start,
                           nodes=getattr(model, "best_nodes", None),
                           n_selected=getattr(model, "n_selected_", None)))
    pairs = list(combinations(range(len(PROVIDERS)), 2))
    discord = [np.mean(np.abs(preds[:, i, 1] - preds[:, j, 1])) for i, j in pairs]
    pair = pairs[int(np.argmax(discord))]
    heads = {}
    for transform, stack_cls in (("probability", StackingLR), ("logit", StackingLogit)):
        for flavor, operation in (("flat", None), ("duplicate", (pair[0], pair[0])), ("max", pair)):
            start = perf_counter()
            stack = stack_cls().fit(augmented(preds, operation), y, task="binary", n_classes=2)
            tree = make_tree(preds, stack, transform, operation)
            key = transform + "_" + flavor
            heads[key] = stack, tree, operation
            timing.append(dict(model=key, seconds=perf_counter()-start, nodes=tree.nodes_count,
                               iterations=int(stack.model[-1].n_iter_[0]),
                               coefficients=stack.model[-1].coef_.size,
                               native_slots=preds.shape[1] + int(operation is not None)))
    return references, heads, pair, timing


def assess(bank, fitted, p, y, split):
    references, heads, _, _ = fitted
    outcomes = {name: obj.predict(p)[:, 1] for name, obj in references.items()}
    discrepancies = {}
    tensors = {i: p[:, i, 1] for i in range(p.shape[1])}
    for name, (stack, tree, pair) in heads.items():
        ref = stack.predict(augmented(p, pair))[:,1]
        out = tree.do_pred_on_another_tensors(current_tensors=tensors).ravel()
        outcomes[name] = out
        discrepancies[name] = np.max(np.abs(out - ref))
        if pair is None:
            outcomes[stack.name] = ref
    rows = []
    for name, prob in outcomes.items():
        assert prob.shape == y.shape and np.isfinite(prob).all()
        rows.append(dict(bank=bank, split=split, model=name,
                         binary_brier=np.square(prob-y).mean(), accuracy=np.mean((prob>.5)==y),
                         max_transfer_error=discrepancies.get(name, np.nan)))
    return outcomes, rows


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    X, y = load_breast_cancer(return_X_y=True)
    dev, test = train_test_split(np.arange(len(X)), test_size=.4, stratify=y, random_state=87299)
    fits, rows, timing, banks, contrasts = {}, [], [], [], []
    with threadpool_limits(limits=1):
        for bank in BANKS:
            tick = perf_counter()
            p = cross_fit(X[dev], y[dev], bank)
            timing.append(dict(bank=bank, model="30_oof_base_fits", seconds=perf_counter()-tick))
            fits[bank] = fit_bank(p, y[dev])
            _, _, pair, costs = fits[bank]
            banks.append(dict(bank=bank, pair=[PROVIDERS[i] for i in pair]))
            timing.extend([dict(bank=bank, **r) for r in costs])
            _, records = assess(bank, fits[bank], p, y[dev], "oof_fit")
            rows.extend(records)
        # Every pair, coefficient and GP tree has been frozen prior to this call.
        tick = perf_counter()
        test_p = base_scores(X[dev], y[dev], X[test], 87310)
        timing.append(dict(model="6_outer_base_fits", seconds=perf_counter()-tick))
        b = np.random.default_rng(87320).integers(0,len(test),size=(20000,len(test)))
        for bank in BANKS:
            outputs, records = assess(bank, fits[bank], test_p, y[test], "outer_test")
            rows.extend(records)
            if bank == BANKS[0]:
                for control in ("flat", "duplicate"):
                    delta = (outputs["logit_"+control]-y[test])**2 - (outputs["logit_max"]-y[test])**2
                    lb = np.quantile(delta[b].mean(1), .025)
                    contrasts.append(dict(control=control, risk_reduction=delta.mean(),
                                          lower_onesided=lb, passes=bool(lb>0)))
            fits[bank][0]["ordinary_gp"]._tmp.cleanup()
    pd.DataFrame(rows).to_csv(out/"scores.csv",index=False)
    pd.DataFrame(contrasts).to_csv(out/"contrasts.csv",index=False)
    pd.DataFrame(timing).to_csv(out/"costs.csv",index=False)
    info = dict(n_dev=len(dev), n_test=len(test), seconds=perf_counter()-start, banks=banks,
                test_index_sha256=hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
                sklearn=sklearn.__version__, numpy=np.__version__)
    (out/"config.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
