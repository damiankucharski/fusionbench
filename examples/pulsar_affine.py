"""Evaluate the preregistered HTRU2 experiment on the original UCI ZIP."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.stacking import StackingLR, StackingLogit
from okapi.affine import fit_stacking_tree
from okapi.node import MaxNode, ValueNode
from real_affine_head import PROVIDERS, augmented, base_scores, cross_fit


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        data = archive.read("HTRU_2.csv")
    fingerprint = hashlib.sha256(data).hexdigest()
    if fingerprint != "b13b4d8929e96ecd196e464c1c8a454c3ac2ffa631015f6388957531a9923f59":
        raise ValueError("CSV differs from preregistered UCI file")
    data = np.loadtxt(data.splitlines(), delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(int)
    assert X.shape == (17898, 8) and len(np.unique(X, axis=0)) == len(X)
    return X, y


def select_pair(preds):
    pairs = list(combinations(range(len(PROVIDERS)), 2))
    differences = [np.abs(preds[:, a, 1] - preds[:, b, 1]).mean() for a, b in pairs]
    return pairs[int(np.argmax(differences))]


def native_roots(preds, operation):
    roots = [ValueNode(None, preds[:, i, 1], i) for i in range(preds.shape[1])]
    if operation is not None:
        a, b = operation
        feature = roots[a].copy_subtree()
        feature.add_child(MaxNode([roots[b].copy_subtree()]))
        roots.append(feature)
    return roots


def project(preds, pair):
    scaler = StandardScaler().fit(preds.reshape(len(preds), -1))
    Z = scaler.transform(preds.reshape(len(preds), -1))
    a, b = pair
    maximum = np.maximum(preds[:, a, 1], preds[:, b, 1])
    g_scaler = StandardScaler().fit(maximum[:, None])
    h = g_scaler.transform(maximum[:, None]).ravel()
    q = np.column_stack((np.ones(len(Z)), Z[:, 1::2]))
    coef = np.linalg.lstsq(q, h, rcond=None)[0]
    return scaler, g_scaler, coef


def designs(preds, projectors, pair):
    scaler, g_scaler, coef = projectors
    Z = scaler.transform(preds.reshape(len(preds), -1))
    a, b = pair
    maximum = np.maximum(preds[:, a, 1], preds[:, b, 1])
    h = g_scaler.transform(maximum[:, None]).ravel()
    s = coef[0] + Z[:, 1::2] @ coef[1:]
    residual = h-s
    scaled = Z.copy()
    scaled[:, 2*a:2*a+2] *= np.sqrt(2)
    zeros = np.zeros((len(Z), 2))
    X = {"flat": np.column_stack((Z, zeros)),
         "duplicate": np.column_stack((Z, Z[:, 2*a:2*a+2])),
         "scaled": scaled, "full": np.column_stack((Z, -h, h)),
         "projected": np.column_stack((Z, -s, s)),
         "residual": np.column_stack((Z, -residual, residual))}
    return X, float(np.var(residual)/np.var(h))


def fit_fusions(preds, y):
    start = perf_counter()
    pair = select_pair(preds)
    projectors = project(preds, pair)
    matrices, _ = designs(preds, projectors, pair)
    costs = [dict(model="pair_selection_and_projection", seconds=perf_counter()-start)]
    heads, native, numerical = {}, {}, {}
    methods = {"mean": (SimpleAverage(), None), "top_n": (TopN("brier"), None),
               "stack_p": (StackingLR(), None), "stack_logit": (StackingLogit(), None),
               "stack_p_duplicate": (StackingLR(), (pair[0], pair[0])),
               "stack_p_max": (StackingLR(), pair)}
    for name, (head, operation) in methods.items():
        start = perf_counter()
        head.fit(augmented(preds, operation), y, task="binary", n_classes=2)
        heads[name] = (head, operation)
        parameters, iterations = None, None
        if hasattr(head, "model"):
            parameters = head.model[-1].coef_.size
            iterations = int(head.model[-1].n_iter_[0])
        costs.append(dict(model=name, seconds=perf_counter()-start, coefficients=parameters,
                          iterations=iterations, n_selected=getattr(head, "n_selected_", None)))
    for name, operation in (("stack_p", None), ("stack_p_duplicate", (pair[0], pair[0])),
                            ("stack_p_max", pair)):
        start = perf_counter()
        roots = native_roots(preds, operation)
        tree = fit_stacking_tree(roots, y, transform="probability")
        native[name] = tree
        costs.append(dict(model="native_"+name, seconds=perf_counter()-start, nodes=tree.nodes_count,
                          coefficients=2*len(roots)))
    for name, matrix in matrices.items():
        start = perf_counter()
        model = LogisticRegression(solver="lbfgs", C=1., tol=1e-10, max_iter=2000).fit(matrix, y)
        numerical[name] = model
        costs.append(dict(model="geometry_"+name, seconds=perf_counter()-start,
                          coefficients=matrix.shape[1], iterations=int(model.n_iter_[0])))
    return pair, projectors, heads, native, numerical, costs


def predict(fit, preds):
    pair, projectors, heads, native, numerical, _ = fit
    matrices, residual_fraction = designs(preds, projectors, pair)
    output = dict(zip(PROVIDERS, preds[:, :, 1].T))
    for key, (model, op) in heads.items():
        output[key] = model.predict(augmented(preds, op))[:, 1]
    tensors = dict(enumerate(preds[:, :, 1].T))
    for key, tree in native.items():
        output["native_"+key] = tree.do_pred_on_another_tensors(current_tensors=tensors).ravel()
    for key, model in numerical.items():
        output["geometry_"+key] = model.predict_proba(matrices[key])[:, 1]
    return output, residual_fraction


def bootstrap(output, y):
    pairs = [("projected", "full"), ("flat", "residual"), ("flat", "duplicate")]
    difference = np.column_stack([(output["geometry_"+control]-y)**2 -
                                  (output["geometry_"+method]-y)**2 for control, method in pairs])
    rng = np.random.default_rng(94120)
    boot = np.empty((10000, len(pairs)))
    for start in range(0, len(boot), 100):
        indices = rng.integers(0, len(y), size=(100, len(y)))
        boot[start:start+100] = difference[indices].mean(axis=1)
    records = []
    for i, (control, method) in enumerate(pairs):
        low, high = np.quantile(boot[:, i], [.025, .975])
        records.append(dict(control=control, method=method, gain=difference[:, i].mean(),
                            q025=low, q975=high, co_primary=i<2))
    return records


def run(archive, out):
    start = perf_counter()
    X, y = load_archive(archive)
    dev, test = train_test_split(np.arange(len(y)), test_size=1/3, random_state=94107, stratify=y)
    tick = perf_counter()
    p = cross_fit(X[dev], y[dev], 94108)
    costs = [dict(model="30_oof_base_fits", seconds=perf_counter()-tick)]
    fit = fit_fusions(p, y[dev])
    costs.extend(fit[-1])
    single = PROVIDERS[int(np.argmin(np.square(p[:, :, 1] - y[dev, None]).mean(axis=0)))]
    tick = perf_counter()
    # All methods, the pair, the projection and the single comparator are fixed.
    external = base_scores(X[dev], y[dev], X[test], 94119)
    costs.append(dict(model="6_outer_base_fits", seconds=perf_counter()-tick))
    scores, diagnostics = [], []
    for split, indices, preds in (("oof_fit", dev, p), ("outer_test", test, external)):
        output, fraction = predict(fit, preds)
        target = y[indices]
        for name, values in output.items():
            assert values.shape == (len(target),) and np.isfinite(values).all()
            scores.append(dict(split=split, model=name, brier=np.square(values-target).mean(),
                               average_precision=average_precision_score(target, values)))
        diagnostics.append(dict(split=split, residual_variance_fraction=fraction,
                                max_duplicate_vs_scaled=np.max(np.abs(output["geometry_duplicate"]-
                                                                      output["geometry_scaled"])),
                                max_native_vs_public=max(np.max(np.abs(output["native_"+k]-
                                                                       output[k])) for k in fit[3]),
                                max_public_vs_tight_full=np.max(np.abs(output["stack_p_max"]-
                                                                       output["geometry_full"]))))
    contrasts = bootstrap(output, y[test])
    out.mkdir(exist_ok=True, parents=True)
    for name, records in (("scores", scores), ("contrasts", contrasts),
                          ("diagnostics", diagnostics), ("costs", costs)):
        pd.DataFrame(records).to_csv(out/(name+".csv"), index=False)
    info = dict(n_dev=len(dev), n_dev_positive=int(y[dev].sum()), n_test=len(test),
                n_test_positive=int(y[test].sum()), single=single,
                selected_pair=[PROVIDERS[i] for i in fit[0]], top_n=fit[2]["top_n"][0].n_selected_,
                passes=all(r["q025"]>0 for r in contrasts if r["co_primary"]),
                test_index_sha256=hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
                sklearn=sklearn.__version__, numpy=np.__version__, seconds=perf_counter()-start)
    (out/"config.json").write_text(json.dumps(info, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.archive, args.out)
