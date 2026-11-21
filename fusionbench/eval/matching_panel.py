"""Paired hidden-matching data, exhaustive test supports and native stock GP."""

from __future__ import annotations

import hashlib
import itertools
import struct
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from okapi.tree import Tree


LOW, HIGH = 1 / 64, 63 / 64


@dataclass(frozen=True)
class MatchingInputs:
    bits: np.ndarray

    @property
    def scores(self):
        return LOW + (HIGH - LOW) * self.bits

    @property
    def preds(self):
        p = self.scores
        return np.stack((1 - p, p), axis=-1)


@dataclass(frozen=True)
class LabeledInputs(MatchingInputs):
    q: dict[str, np.ndarray]
    u: np.ndarray
    y: dict[str, np.ndarray]


@dataclass(frozen=True)
class MatchingPanel:
    matching: tuple[tuple[int, int], ...]
    roles: tuple[int, ...]
    pi: float
    fit: LabeledInputs
    validation: LabeledInputs
    unlabeled: MatchingInputs
    test: dict[int, LabeledInputs]
    weights: dict[str, np.ndarray]


def _bits(active, n_providers):
    flags = np.zeros((len(active), n_providers), dtype=bool)
    flags[np.arange(len(flags))[:, None], active] = True
    return flags


def _labeled(bits, matching, roles, uniforms):
    pairs = np.asarray(matching)
    paired = (bits[:, pairs[:, 0]] & bits[:, pairs[:, 1]]).any(axis=1)
    a, b, c, d = roles
    score = (
        0.2 + 1.1 * bits[:, a] + 0.6 * bits[:, b] - 1.2 * bits[:, c] - 0.7 * bits[:, d]
    )
    q = {"matching": np.where(paired, HIGH, LOW), "logistic": 1 / (1 + np.exp(-score))}
    y = {
        law: (uniforms < (p if uniforms.ndim == 1 else p[:, None])).astype(np.int64)
        for law, p in q.items()
    }
    return LabeledInputs(bits, q, uniforms, y)


def make_panel(seed, k, design):
    """Generate one anonymous provider bank and its two paired binary label laws.

    Streams use default_rng([seed,k,code]); codes 0/1 independently permute
    matching and logistic roles, codes 52/53/54 give 64 paired test uniforms
    per lexicographic support state of size 2/3/4. These are common to designs.
    Development streams append 0 (balanced) or 1 (uniform) to that seed:
    10/11 are fit features/label uniforms, 20/21 validation, 30 unlabeled X.
    The uniform feature generator never consults the matching. The balanced
    generator draws group indicators, all within-group indices, then all cross
    indices. All groups use lexicographically ordered unordered pairs.

    Fit/validation labels have one draw per row. Test labels have shape
    [state,64], while each test state appears only once among the features.
    Each weight vector sums to one over the corresponding support.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 2:
        raise ValueError("k must be an integer >= 2")
    if design not in ("balanced", "uniform"):
        raise ValueError("design must be 'balanced' or 'uniform'")
    k = int(k)
    n = 2 * k
    design_id = int(design == "uniform")
    perm = np.random.default_rng([seed, k, 0]).permutation(n).reshape(k, 2)
    matching = tuple(sorted(tuple(sorted(map(int, edge))) for edge in perm))
    roles = tuple(map(int, np.random.default_rng([seed, k, 1]).permutation(n)[:4]))
    all_edges = tuple(itertools.combinations(range(n), 2))
    edge_array = np.asarray(all_edges)
    matched = np.asarray(matching)
    inside = set(matching)
    cross = np.asarray([edge for edge in all_edges if edge not in inside])
    pi = 0.5 if design_id == 0 else 1 / (n - 1)

    def features(size, code):
        rng = np.random.default_rng([seed, k, code, design_id])
        if design_id:
            active = edge_array[rng.integers(len(edge_array), size=size)]
        else:
            within = rng.random(size) < pi
            positive = matched[rng.integers(len(matched), size=size)]
            negative = cross[rng.integers(len(cross), size=size)]
            active = np.where(within[:, None], positive, negative)
        return _bits(active, n)

    def sample(size, code):
        u = np.random.default_rng([seed, k, code + 1, design_id]).random(size)
        return _labeled(features(size, code), matching, roles, u)

    test, weights = {}, {}
    for size, name in ((2, "two_uniform"), (3, "three_uniform"), (4, "four_uniform")):
        subsets = np.asarray(list(itertools.combinations(range(n), size)))
        bits = _bits(subsets, n)
        u = np.random.default_rng([seed, k, 50 + size]).random((len(bits), 64))
        test[size] = _labeled(bits, matching, roles, u)
        weights[name] = np.full(len(bits), 1 / len(bits))
    weights["two_balanced"] = np.array(
        [0.5 / k if e in inside else 0.5 / len(cross) for e in all_edges]
    )
    return MatchingPanel(
        matching,
        roles,
        pi,
        sample(64, 10),
        sample(128, 20),
        MatchingInputs(features(128, 30)),
        test,
        weights,
    )


@dataclass(frozen=True)
class StockResult:
    fit_tree: Tree
    validation_tree: Tree
    calls: int
    archive_trees: int
    path_digest: str
    fit_brier: float
    validation_brier: float
    elapsed_seconds: float
    max_generation: int


def _positive(probs, labels):
    x = np.asarray(probs)
    if x.ndim == 3 and x.shape[2] == 2:
        x = x[:, :, 1]
    if x.ndim != 2 or x.shape[0] != len(labels) or not len(x) or x.shape[1] < 2:
        raise ValueError(
            "Expected [row,provider] scores or [row,provider,2] probabilities and matching labels"
        )
    return x


def stock_winners(fit_probs, y, val_probs, val_y, gp_seed):
    """Select both archive champions after exactly 600 true native Brier calls.

    n singleton providers seed the standard six-operator NumPy optimizer.
    Fitness passes use the actual native batched routine, truncating only its
    final batch. Repeated parents count against the budget. Archive selection
    sees unrounded losses and retains first occurrence on exact ties, with
    training Brier as a secondary criterion for the validation champion.
    """
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    from fusionbench.eval.structural_key_audit import training_files

    y, val_y = np.asarray(y).ravel(), np.asarray(val_y).ravel()
    fit, val = _positive(fit_probs, y), _positive(val_probs, val_y)
    n = fit.shape[1]
    if val.shape[1] != n:
        raise ValueError("Training and validation scores must use the same providers")
    validation = {f"p{i}.npy": val[:, i : i + 1] for i in range(n)}
    archive, raw = {}, []
    digest = hashlib.sha256()
    calls = generation = 0

    def objective(pred, truth):
        loss = float(np.mean(np.square(pred.ravel() - truth.ravel())))
        raw.append(loss)
        return 1.0 - loss

    class LastCall(Exception):
        pass

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls
            cohort = self.population if trees is None else trees
            batch = cohort[: 600 - calls]
            raw.clear()
            scores = super()._calculate_fitnesses(batch)
            for tree, loss in zip(batch, raw, strict=True):
                calls += 1
                key = tree.structural_key()
                digest.update((repr(key) + "\n").encode())
                digest.update(struct.pack("<d", loss))
                if key not in archive:
                    clone = tree.copy()
                    clone.mutation_chance = tree.mutation_chance
                    p = clone.do_pred_on_another_tensors(
                        current_tensors=validation
                    ).ravel()
                    archive[key] = loss, float(np.mean(np.square(p - val_y))), clone
            if calls == 600:
                raise LastCall
            return scores

    with training_files(fit, y) as (paths, gt):
        start = perf_counter()
        gp = Observed(
            paths,
            gt,
            n,
            2,
            2,
            backend="numpy",
            seed=gp_seed,
            allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
            postprocessing_function=np.asarray,
            objective_functions=(objective,),
            reuse_survivor_fitness=False,
        )
        try:
            while True:
                generation += 1
                gp.run_iteration()
        except LastCall:
            pass
        elapsed = perf_counter() - start
    a = min(archive.values(), key=lambda e: e[0])
    b = min(archive.values(), key=lambda e: (e[1], e[0]))
    return StockResult(
        a[2],
        b[2],
        calls,
        len(archive),
        digest.hexdigest(),
        a[0],
        b[1],
        elapsed,
        generation,
    )
