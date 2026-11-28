"""Exact capacity certificate for the two binary laws on the uniform eight cells.

Write input probabilities as L + D*X_j, where the X_j are independent fair
bits and D=H-L=3/4. A native program outputs p for the positive class; full
Brier is the sum over both classes, so its excess risk equals 2 E[(p-q)^2].

For S=C(A or B) on bits in role order, E[S]=3/8 and Cov(X_j,S) is
(1/16, 1/16, 3/16). For a normalized weighted mean with total weight one,
E[(w·X-S)^2]=(1/2-3/8)^2+15/64+sum(w_j^2)/4-2 sum(w_j Cov(X_j,S)).
The quadratic on the plane sum(w)=1 equals 1/12 + sum((w_j-w*_j)^2)/4,
where w*=(1/6,1/6,2/3). In particular w* is feasible for native WEIGHTED_MEAN.
Omitted or repeated providers only change their net weights. MEAN is covered.

A single MIN/MAX reduces to the min/max on its nonempty support. For CLOSE
and FAR on two endpoints, which endpoint is closer/further is decided by which
side of the midpoint the scalar threshold is on. At the midpoint every operand
ties, so the first operand is returned. On rows where all inputs are identical,
both reductions and the first operand already agree. Thus it suffices to
enumerate seven nonempty subsets and three singletons, regardless of arity,
duplicates or threshold. Exact rational enumeration below verifies the minima.
The binary native postprocessor is identity on the positive-probability column.
"""

import json
from fractions import Fraction as F
from itertools import combinations, permutations, product
from statistics import mean

import numpy as np
from loguru import logger
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import (
    CloseThresholdNode,
    FarThresholdNode,
    MaxNode,
    MeanNode,
    MinNode,
    ValueNode,
    WeightedMeanNode,
)
from okapi.tree import Tree

L, H = F(1, 8), F(7, 8)
SPAN = H - L
CUBE = tuple(product((0, 1), repeat=3))


def excess(p, q):
    return 2 * mean((u - v) ** 2 for u, v in zip(p, q, strict=True))


def optimum_on_bits(s):
    ybar = mean(F(y) for y in s)
    cov = [
        mean(F(x[i] * y) for x, y in zip(CUBE, s, strict=True)) - ybar / 2
        for i in range(3)
    ]
    assert ybar == F(3, 8) and cov == [F(1, 16), F(1, 16), F(3, 16)]
    # Linear terms of the squared norm at this optimum have a common gradient,
    # hence their dot product with any displacement on sum(w)=1 vanishes.
    w = [4 * c + (1 - 4 * sum(cov)) / 3 for c in cov]
    assert w == [F(1, 6), F(1, 6), F(2, 3)] and sum(w) == 1
    assert len({weight / 2 - 2 * c for weight, c in zip(w, cov, strict=True)}) == 1
    predicted = [
        sum(value * weight for value, weight in zip(x, w, strict=True)) for x in CUBE
    ]
    return w, excess(predicted, s) / 2


def exact_certificate():
    bits = tuple(c and (a or b) for a, b, c in CUBE)
    w, bit_error = optimum_on_bits(bits)
    assert bit_error == F(1, 12)
    X = tuple(tuple(L + SPAN * b for b in row) for row in CUBE)
    q = tuple(L + SPAN * b for b in bits)
    qnull = tuple(mean(row) for row in X)
    risk_floor = mean(2 * s * (1 - s) for s in q)
    null_floor = mean(2 * s * (1 - s) for s in qnull)
    flat = 2 * SPAN**2 * bit_error
    mean_excess = excess(qnull, q)
    assert (risk_floor, null_floor, flat, mean_excess) == (
        F(7, 32),
        F(13, 32),
        F(3, 32),
        F(9, 64),
    )
    by_reduction = {}
    for name, reduce in (("min", min), ("max", max)):
        errs = {}
        for size in (1, 2, 3):
            for subset in combinations(range(3), size):
                pred = [reduce(row[i] for i in subset) for row in X]
                errs["".join("ABC"[i] for i in subset)] = excess(pred, q)
        assert min(errs.values()) == F(9, 64)
        by_reduction[name] = {k: str(v) for k, v in errs.items()}
    return {
        "population": "uniform on eight cells; Bernoulli outcomes conditional on the cell",
        "score": "full binary Brier (sum over two classes)",
        "low": str(L),
        "high": str(H),
        "null": {
            "law": "mean(A,B,C)",
            "bayes_brier": str(null_floor),
            "mean_excess": "0",
        },
        "positive": {
            "law": "min(C,max(A,B))",
            "bit_truth_table_lexicographic_ABC": list(map(int, bits)),
            "bayes_brier": str(risk_floor),
            "two_update_excess": "0",
            "equal_mean_brier": str(risk_floor + mean_excess),
            "equal_mean_excess": str(mean_excess),
            "best_one_update_brier": str(risk_floor + flat),
            "best_one_update_excess": str(flat),
            "convex_minimizer_weights_ABC": list(map(str, w)),
            "best_min_or_max_or_threshold_excess": "9/64",
            "one_update_reduction_excess_by_support": by_reduction,
        },
    }


def native_checks():
    x = np.asarray(CUBE, dtype=float) * float(SPAN) + float(L)

    def val(i, ops=()):
        return ValueNode(list(ops), x[:, i : i + 1], f"anonymous_{i}.npy")

    def forward(tree, nops):
        assert len(tree.op_nodes) == nops
        return np.asarray(tree.predict()).ravel()

    checked = 0
    natural = np.array([float(mean(L + SPAN * b for b in row)) for row in CUBE])
    for a, b, c in permutations(range(3)):
        truth = np.minimum(x[:, c], np.maximum(x[:, a], x[:, b]))
        witness = Tree(val(a, [MaxNode([val(b)]), MinNode([val(c)])]))
        np.testing.assert_array_equal(forward(witness, 2), truth)
        mean_tree = Tree(val(a, [MeanNode([val(b), val(c)])]))
        np.testing.assert_array_equal(forward(mean_tree, 1), natural)
        best = Tree(val(a, [WeightedMeanNode([val(b), val(c)], [1 / 6, 1 / 6, 2 / 3])]))
        np.testing.assert_allclose(
            2 * np.mean((forward(best, 1) - truth) ** 2), 3 / 32, rtol=0, atol=1e-15
        )
        for size in (1, 2, 3):
            for support in permutations(range(3), size):
                anchor = support[0]
                partners = support[1:] or (anchor,)
                group = x[:, support]
                low, high, first = group.min(1), group.max(1), group[:, 0]
                expected = [
                    (MinNode, (), low),
                    (MaxNode, (), high),
                    (CloseThresholdNode, (0.0,), low),
                    (CloseThresholdNode, (0.5,), first),
                    (CloseThresholdNode, (1.0,), high),
                    (FarThresholdNode, (0.0,), high),
                    (FarThresholdNode, (0.5,), first),
                    (FarThresholdNode, (1.0,), low),
                ]
                for node_type, params, ref in expected:
                    op = node_type([val(j) for j in partners], *params)
                    scores = forward(Tree(val(anchor, [op])), 1)
                    np.testing.assert_array_equal(scores, ref)
                    assert 2 * np.mean((scores - truth) ** 2) >= 9 / 64
                    checked += 1
    return {
        "role_permutations": 6,
        "two_update_witnesses": 6,
        "attaining_weighted_means": 6,
        "null_mean_trees": 6,
        "native_threshold_min_max_checks": checked,
    }


def verify():
    logger.disable("okapi")
    set_backend("numpy")
    summary = exact_certificate()
    with use_postprocessing_function(lambda x: x):
        summary["native_replay"] = native_checks()
    return summary


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
