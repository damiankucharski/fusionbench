"""Fit a raw-input circuit from unlabelled mixtures and a separate labelled sample.

The network is one affine projection followed by signed min/max operators.
It receives neither expert outputs nor concept labels. Class tasks below have
the same rule across independently redrawn rotations. The fitted networks do
not see any labelled or unlabelled samples from the held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

DIM = 4
TASKS = ("compound", "two_pathways", "parity")
BITS = np.arange(1 << DIM)[:, None] >> np.arange(DIM) & 1
CELL_WEIGHTS = 1 << np.arange(DIM)


@dataclass(frozen=True)
class Rule:
    """A signed affine coordinate at a leaf or an idempotent min/max update."""
    name: str
    leaves: int
    args: tuple

    def score(self, z):
        if self.name == "axis":
            axis, sign = self.args
            return sign * z[:, axis]
        left, right = self.args
        op = np.minimum if self.name == "min" else np.maximum
        return op(left.score(z), right.score(z))

    def compact(self):
        if self.name == "axis":
            axis, sign = self.args
            return ("+" if sign == 1 else "-") + str(axis)
        return self.name + "(" + ",".join(arg.compact() for arg in self.args) + ")"


@dataclass(frozen=True)
class RawCircuit:
    """Portable single affine layer with a binary min/max reduction and zero threshold."""
    weights: np.ndarray
    bias: np.ndarray
    reduction: int | tuple

    @classmethod
    def from_axes(cls, rule, matrix, offset):
        rows = []

        def compile_node(node):
            if node.name == "axis":
                axis, sign = node.args
                rows.append(sign * matrix[axis])
                return len(rows) - 1
            left, right = node.args
            return node.name, compile_node(left), compile_node(right)

        reduction = compile_node(rule)
        weights = np.stack(rows)
        return cls(weights, -(weights @ offset), reduction)

    def score(self, x):
        values = x @ self.weights.T + self.bias

        def reduce_node(node):
            if isinstance(node, int):
                return values[:, node]
            op, left, right = node
            return (np.minimum if op == "min" else np.maximum)(reduce_node(left), reduce_node(right))

        return reduce_node(self.reduction)

    def to_dict(self):
        return dict(weights=self.weights.tolist(), bias=self.bias.tolist(), reduction=self.reduction)


@lru_cache(maxsize=1)
def library():
    """Enumerate read-once signed min/max functions; deduplicate by truth table."""
    groups = {}
    for axis in range(DIM):
        group = {}
        for sign in (1, -1):
            mask = sum((int(v) << i) for i, v in enumerate(BITS[:, axis] == (sign == 1)))
            group[mask] = Rule("axis", 1, (axis, sign))
        groups[1 << axis] = group
    for n in range(2, DIM + 1):
        for subset in itertools.combinations(range(DIM), n):
            target = sum(1 << axis for axis in subset)
            group = {}
            for first in range(1, target):
                second = target ^ first
                if first >= second or (first & target) != first:
                    continue
                for a, left in groups[first].items():
                    for b, right in groups[second].items():
                        for op, mask in (("min", a & b), ("max", a | b)):
                            if mask not in group:
                                group[mask] = Rule(op, n, (left, right))
            groups[target] = group
    entries = sorted(
        ((rule.leaves, mask, rule) for group in groups.values() for mask, rule in group.items()),
        key=lambda item: (item[0], item[1]),
    )
    masks = np.array([mask for _, mask, _ in entries], dtype=np.int32)
    tables = (masks[:, None] >> np.arange(1 << DIM)) & 1
    return tables, [rule for _, _, rule in entries]


def cell_id(z):
    return (z > 0).astype(int) @ CELL_WEIGHTS


def fit_rule(z, y):
    tables, rules = library()
    idx = cell_id(z)
    yes = np.bincount(idx[y], minlength=1 << DIM)
    no = np.bincount(idx[~y], minlength=1 << DIM)
    errors = yes.sum() + tables @ (no - yes)
    # Ties prefer fewer leaves, then a deterministic truth-table code.
    which = int(errors.argmin())
    return rules[which], int(errors[which])


def draw_sources(rng, n, mixture):
    # Use the *same* mixture Rademacher signs and noise draws in the paired
    # Gaussian control. Both populations have zero mean and identity covariance.
    atom = rng.choice([-1., 1.], size=(n, DIM))
    noise = rng.normal(size=(n, DIM))
    return (atom + .5 * noise) / np.sqrt(1.25) if mixture else noise


def make_y(s, flip):
    a, b, c, d = (s > 0).T
    return {"compound": ((a | b) & c) ^ flip, "two_pathways": ((a & b) | (c & d)) ^ flip,
            "parity": (a ^ b) ^ flip}


def learn_axes(x, q, seed):
    """Rows of each returned matrix are projections in raw coordinates."""
    pca = PCA(n_components=DIM, whiten=True, svd_solver="full").fit(x)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always", ConvergenceWarning)
        ica = FastICA(n_components=DIM, whiten="unit-variance", random_state=seed,
                      max_iter=256, tol=1e-4, whiten_solver="svd").fit(x)
    converged = not any(issubclass(msg.category, ConvergenceWarning) for msg in log)
    return {"ica": (ica.components_, converged), "pca": (pca.components_, True),
            "raw": (np.eye(DIM), True), "given": (q, True)}


def alignment(matrix, q):
    normed = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cosines = abs(normed @ q.T)
    i, j = linear_sum_assignment(-cosines)
    return float(cosines[i, j].mean())


def fit_table(z, y):
    idx = cell_id(z)
    n = np.bincount(idx, minlength=1 << DIM)
    positives = np.bincount(idx[y], minlength=1 << DIM)
    # One pseudo-observation at global prevalence; also handles unseen cells.
    return (positives + y.mean()) / (n + 1) >= .5


def summarize(pred, target):
    return dict(accuracy=float(np.mean(pred == target)), correct=int(np.sum(pred == target)))


def run_seed(seed, sizes, n_unlabelled, n_test, export_path=None):
    rotation_rng = np.random.default_rng([seed, 1701])
    q, _ = np.linalg.qr(rotation_rng.normal(size=(DIM, DIM)))
    rows, axis_rows = [], []
    for domain, mixture in [("bimodal", True), ("gaussian", False)]:
        rng = np.random.default_rng([seed, 1722])
        unlab = draw_sources(rng, n_unlabelled, mixture) @ q
        source_fit = draw_sources(rng, max(sizes), mixture)
        source_test = draw_sources(rng, n_test, mixture)
        y_fit = make_y(source_fit, rng.random(len(source_fit)) < .04)
        y_test = make_y(source_test, rng.random(len(source_test)) < .04)
        x_fit, x_test = source_fit @ q, source_test @ q
        offset = unlab.mean(axis=0)
        axes = learn_axes(unlab, q, seed)
        for axis_name, (matrix, converged) in axes.items():
            # Scale rows from unlabelled data to make classifier regularization
            # comparable. The orientation and centring determine circuit decisions.
            scale = (unlab @ matrix.T).std(axis=0)
            matrix = matrix / scale[:, None]
            z_fit, z_test = (x_fit - offset) @ matrix.T, (x_test - offset) @ matrix.T
            axis_rows.append(dict(seed=seed, domain=domain, axes=axis_name, converged=converged,
                                  abs_cosine=alignment(matrix, q)))
            for n in sizes:
                for task in TASKS:
                    labels = y_fit[task][:n]
                    z = z_fit[:n]
                    rule, errors = fit_rule(z, labels)
                    info = dict(seed=seed, domain=domain, n_fit=n, n_test=n_test, axes=axis_name,
                                task=task)
                    network = RawCircuit.from_axes(rule, matrix, offset)
                    predictions = network.score(x_test) > 0
                    np.testing.assert_array_equal(predictions, rule.score(z_test) > 0)
                    if (export_path is not None and axis_name == "ica" and domain == "bimodal"
                            and n == min(sizes) and task == "two_pathways"):
                        export_path.write_text(json.dumps(dict(**info, **network.to_dict()), indent=2) + "\n")
                    rows.append(dict(**info, model="circuit", rule=rule.compact(),
                                     n_coefficients=rule.leaves * (DIM + 1), fit_errors=errors,
                                     **summarize(predictions, y_test[task])))
                    table = fit_table(z, labels)
                    rows.append(dict(**info, model="table", rule="", n_coefficients=DIM*(DIM+1) + len(table),
                                     fit_errors=np.sum(table[cell_id(z)] != labels),
                                     **summarize(table[cell_id(z_test)], y_test[task])))
                    if axis_name not in ("raw", "ica"):
                        continue
                    for name, model in [
                        ("rbf", SVC(C=4., gamma="scale")),
                        ("mlp", MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", solver="lbfgs",
                                              alpha=1., max_iter=250, random_state=seed)),
                    ]:
                        if labels.all() or not labels.any():
                            pred = np.full(n_test, labels[0])
                            fit_errors = 0
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", ConvergenceWarning)
                                model.fit(z, labels)
                            pred = model.predict(z_test)
                            fit_errors = int(np.sum(model.predict(z) != labels))
                        n_coeff = 193 if name == "mlp" else None  # 4x32, 32 biases, 32x1, 1 bias
                        rows.append(dict(**info, model=name, rule="", n_coefficients=n_coeff,
                                         fit_errors=fit_errors, **summarize(pred, y_test[task])))
    return rows, axis_rows


def run(out, seeds=12, start=0, sizes=(32, 128, 512), n_unlabelled=4096, n_test=4096):
    if min(seeds, *sizes, n_test) < 1 or n_unlabelled <= DIM:
        raise ValueError("use positive seed, label and test counts and more unlabelled examples than dimensions")
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    (out / "config.json").write_text(json.dumps(dict(seeds=seeds, start=start, sizes=list(sizes),
        n_unlabelled=n_unlabelled, n_test=n_test, dim=DIM, label_flip=.04, rules=len(library()[1])), indent=2) + "\n")
    with threadpool_limits(limits=1), (out / "runs.csv").open("w", newline="") as runs, (out / "axes.csv").open("w", newline="") as axes:
        for seed in range(start, start + seeds):
            example = out / "example_model.json" if seed == start else None
            records, orientations = run_seed(seed, sizes, n_unlabelled, n_test, example)
            if seed == start:
                writer = csv.DictWriter(runs, fieldnames=records[0])
                axis_writer = csv.DictWriter(axes, fieldnames=orientations[0])
                writer.writeheader()
                axis_writer.writeheader()
            writer.writerows(records)
            axis_writer.writerows(orientations)
            runs.flush()
            axes.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--unlabelled", type=int, default=4096)
    parser.add_argument("--test", type=int, default=4096)
    args = parser.parse_args()
    run(args.out, args.seeds, args.start, args.sizes, args.unlabelled, args.test)
