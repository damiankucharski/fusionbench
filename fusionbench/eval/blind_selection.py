"""Check an adaptive choice of a blind circuit, a lookup table or an RBF SVM.

Thresholds and code lengths are fixed before evaluation. The default rotations
100--111 are disjoint from the exploratory runs of blind_circuit.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import betaln
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from fusionbench.eval.blind_circuit import (
    DIM, TASKS, RawCircuit, alignment, cell_id, draw_sources, fit_rule, fit_table,
    library, make_y,
)

NAMES = ("circuit", "rbf", "table")


def null_quantile():
    rng = np.random.default_rng(940_531)
    rotations, _ = np.linalg.qr(rng.normal(size=(4096, DIM, DIM)))
    return float(np.quantile([alignment(np.eye(DIM), q) for q in rotations], .95))


def fit_axes(raw, seed):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = FastICA(n_components=DIM, whiten="unit-variance", random_state=seed,
                      max_iter=256, tol=1e-4, whiten_solver="svd").fit(raw)
    converged = not any(issubclass(msg.category, ConvergenceWarning) for msg in messages)
    matrix = fit.components_
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True), converged


def residual_bits(errors, total):
    # Universal binary code for the ordered residuals under iid label flips;
    # the Jeffreys mixture integrates out the unknown error probability.
    return float((betaln(.5, .5) - betaln(errors + .5, total - errors + .5)) / np.log(2))


def competitors(x, y, query, projection, offset, scale):
    z = (x - offset) @ projection.T
    rule, errs = fit_rule(z, y)
    table = fit_table(z, y)
    table_errs = int(np.sum(table[cell_id(z)] != y))
    net = RawCircuit.from_axes(rule, projection, offset)
    if np.unique(y).size == 1:
        rbf = np.full(len(query), y[0])
    else:
        svm = SVC(C=4., gamma="scale").fit((x-offset)/scale, y)
        rbf = svm.predict((query-offset)/scale)
    estimates = {"circuit": net.score(query) > 0, "rbf": rbf, "table": table[cell_id((query-offset) @ projection.T)]}
    circuit_bits = np.log2(len(library()[1])) + residual_bits(errs, len(y))
    table_bits = (1 << DIM) + residual_bits(table_errs, len(y))
    return estimates, circuit_bits, table_bits


def run_seed(seed, cutoff, sizes, n_unlabelled, n_test):
    rotation_rng = np.random.default_rng([seed, 1701])
    q, _ = np.linalg.qr(rotation_rng.normal(size=(DIM, DIM)))
    rows, geometry = [], []
    for domain, mixture in (("bimodal", True), ("gaussian", False)):
        rng = np.random.default_rng([seed, 1722])
        unlab = draw_sources(rng, n_unlabelled, mixture) @ q
        train_sources = draw_sources(rng, max(sizes), mixture)
        test_sources = draw_sources(rng, n_test, mixture)
        y_fit = make_y(train_sources, rng.random(len(train_sources)) < .04)
        y_test = make_y(test_sources, rng.random(len(test_sources)) < .04)
        x_train, x_test = train_sources @ q, test_sources @ q
        axis, _ = fit_axes(unlab, seed)
        left, lc = fit_axes(unlab[:len(unlab)//2], seed+101)
        right, rc = fit_axes(unlab[len(unlab)//2:], seed+107)
        agreement = alignment(left, right)
        stable = agreement > cutoff
        geometry.append(dict(seed=seed, domain=domain, agreement=agreement, cutoff=cutoff,
                             stable=stable, left_converged=lc, right_converged=rc,
                             axis_cosine_truth=alignment(axis, q)))
        offset, scale = unlab.mean(axis=0), unlab.std(axis=0)
        for n in sizes:
            # Training observations are iid and their order is independent of
            # all labels. Fit the plain model chooser on its fixed prefix.
            split = int(3*n/4)
            for task in TASKS:
                labels = y_fit[task][:n]
                candidates, cb, tb = competitors(x_train[:n], labels, x_test, axis, offset, scale)
                val, _, _ = competitors(x_train[:split], labels[:split], x_train[split:n], axis, offset, scale)
                val_choice = min(NAMES, key=lambda name: np.mean(val[name] != labels[split:]))
                choice = ("circuit" if cb <= tb else "table") if stable else "rbf"
                methods = [(name, name) for name in NAMES] + [("adaptive", choice), ("holdout", val_choice)]
                for method, fitted in methods:
                    pred = candidates[fitted]
                    rows.append(dict(seed=seed, domain=domain, task=task, n_fit=n, n_test=n_test,
                                     method=method, choice=fitted, correct=int(np.sum(pred == y_test[task])),
                                     accuracy=float(np.mean(pred == y_test[task])), circuit_bits=cb, table_bits=tb))
    return rows, geometry


def run(out, start=100, seeds=12, sizes=(32, 128, 512), n_unlabelled=4096, n_test=4096):
    if min(sizes) < 4 or n_unlabelled < 2*(DIM+1) or min(seeds, n_test) < 1:
        raise ValueError("use at least four labels, two estimable ICA halves and positive seed/test counts")
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    cutoff = null_quantile()
    (out/"config.json").write_text(json.dumps(dict(start=start, seeds=seeds, sizes=list(sizes),
        n_unlabelled=n_unlabelled, n_test=n_test, cutoff=cutoff, grammar_size=len(library()[1]),
        null_draws=4096, quantile=.95, label_validation_fraction=.25), indent=2) + "\n")
    with threadpool_limits(limits=1), (out/"runs.csv").open("w", newline="") as runs, (out/"geometry.csv").open("w", newline="") as geom:
        for seed in range(start, start + seeds):
            rows, geometry = run_seed(seed, cutoff, sizes, n_unlabelled, n_test)
            if seed == start:
                writer = csv.DictWriter(runs, fieldnames=rows[0])
                geo_writer = csv.DictWriter(geom, fieldnames=geometry[0])
                writer.writeheader()
                geo_writer.writeheader()
            writer.writerows(rows)
            geo_writer.writerows(geometry)
            runs.flush()
            geom.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512])
    args = parser.parse_args()
    run(args.out, args.start, args.seeds, args.sizes)
