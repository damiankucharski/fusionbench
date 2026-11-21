"""Exact next-crossover probabilities for the four-input scalar MIN/MAX grammar.

Each integer is a 16-corner truth table. At one corner an arbitrary single-input
monotone context is determined by its outputs for input zero and one. Applying
these 16 contexts in parallel is one bitwise operation per mask.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import combinations
from numbers import Integral

import numpy as np

from fusionbench.eval.lattice_signature import provider_signatures

ALL = (1 << 16) - 1
DEFAULT_NAMES = tuple(f"p{i}.npy" for i in range(4))
IDENTITY = (0, ALL)


def _apply(context, x):
    lo, hi = context
    return lo | (hi & x)


def _compose(outer, inner):
    return _apply(outer, inner[0]), _apply(outer, inner[1])


@dataclass(frozen=True)
class Site:
    path: tuple[int, ...]
    outer: tuple[int, int]
    incoming: int
    donor: tuple[int, int]

    def with_donor(self, other):
        """Resulting root signature after replacing this site by the same-type donor."""
        return _apply(self.outer, _apply(other.donor, self.incoming))


@dataclass(frozen=True, eq=False)
class PreparedTree:
    provider_names: tuple
    signature: int
    value_sites: tuple[Site, ...]
    operator_sites: tuple[Site, ...]
    subtree_signatures: frozenset[int]


def _freeze(node):
    if isinstance(node, dict):
        kind = node["type"]
        name = node.get("id")
        children = node["children"]
    else:
        kind = type(node).__name__
        name = getattr(node, "id", None)
        children = node.children
    if kind not in {"ValueNode", "MinNode", "MaxNode"}:
        raise ValueError(f"Unsupported four-input lattice node {kind!r}")
    return kind, name, tuple(_freeze(x) for x in children)


@lru_cache(maxsize=4096)
def _prepare(root, names):
    signatures = provider_signatures(names)

    @cache
    def inside(node):
        kind, name, children = node
        if kind == "ValueNode":
            if name not in signatures:
                raise ValueError(f"Unknown lattice provider {name!r}")
            acc = signatures[name]
            for op in children:
                if op[0] == "ValueNode":
                    raise ValueError("A ValueNode requires operator children")
                acc = _apply(inside(op), acc)
            return acc, acc
        is_min = kind == "MinNode"
        fold = ALL if is_min else 0
        for child in children:
            if child[0] != "ValueNode":
                raise ValueError("An operator requires ValueNode children")
            sig = inside(child)[0]
            fold = (fold & sig) if is_min else (fold | sig)
        return (0, fold) if is_min else (fold, ALL)

    values, operators = [], []

    def visit(node, path, outer):
        _, name, ops = node
        values.append(Site(path, outer, 0, inside(node)))
        # The input of each operator is the result left by its earlier siblings.
        prefixes = [signatures[name]]
        for op in ops:
            prefixes.append(_apply(inside(op), prefixes[-1]))
        suffix = outer
        for k in reversed(range(len(ops))):
            op = ops[k]
            op_path = path + (k,)
            operators.append(Site(op_path, suffix, prefixes[k], inside(op)))
            is_min = op[0] == "MinNode"
            children = op[2]
            prefix = [prefixes[k]]
            for child in children:
                sig = inside(child)[0]
                prefix.append((prefix[-1] & sig) if is_min else (prefix[-1] | sig))
            tail = ALL if is_min else 0
            for j in reversed(range(len(children))):
                held = (prefix[j] & tail) if is_min else (prefix[j] | tail)
                ingredient = (0, held) if is_min else (held, ALL)
                visit(children[j], op_path + (j,), _compose(suffix, ingredient))
                sig = inside(children[j])[0]
                tail = (tail & sig) if is_min else (tail | sig)
            suffix = _compose(suffix, inside(op))

    if root[0] != "ValueNode":
        raise ValueError("The native tree root must be a ValueNode")
    visit(root, (), IDENTITY)
    return PreparedTree(
        names,
        inside(root)[0],
        tuple(values),
        tuple(operators),
        frozenset(site.donor[0] for site in values),
    )


def prepare_tree(tree, *, provider_names=DEFAULT_NAMES):
    """Accept a native tree or a serialized native root, without running any GP."""
    names = tuple(provider_names)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Exactly four distinct provider names are required")
    if isinstance(tree, PreparedTree):
        if tree.provider_names != names:
            raise ValueError("Prepared trees must use a common provider order")
        return tree
    return _prepare(_freeze(getattr(tree, "root", tree)), names)


def tournament_weights(rounded_fitness):
    """Two distinct contestants in random order; the first wins an exact distance tie."""
    values = np.asarray(rounded_fitness, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Expected at least two finite scalar fitness values in [0,1]")
    # Use the same float64 distance calculation as native tournaments.
    distance = np.sqrt((1.0 - values) ** 2)
    worse = (distance[:, None] < distance).sum(axis=1)
    tied = (distance[:, None] == distance).sum(axis=1) - 1
    return (2 * worse + tied) / (len(values) * (len(values) - 1))


@lru_cache(maxsize=8192)
def _offspring_pairs(left, right):
    """Counts within each admissible node type; native chooses a type uniformly first."""
    groups = [(left.value_sites, right.value_sites)]
    if left.operator_sites and right.operator_sites:
        groups.append((left.operator_sites, right.operator_sites))
    result = []
    for a, b in groups:
        counts = Counter((x.with_donor(y), y.with_donor(x)) for x in a for y in b)
        result.append(
            (len(a) * len(b), tuple((x, y, count) for (x, y), count in counts.items()))
        )
    return tuple(result)


@dataclass(frozen=True)
class Joins:
    """Counts of unordered pairs of distinct individuals satisfying each criterion."""

    union_pairs: int
    intersection_pairs: int
    strict_union_pairs: int
    strict_intersection_pairs: int
    parents_free_union_pairs: int
    parents_free_intersection_pairs: int
    parents_free_strict_union_pairs: int
    parents_free_strict_intersection_pairs: int


def _join_counts(ingredients, parents_free, targets):
    counts = np.zeros(8, dtype=int)
    for i, j in combinations(range(len(ingredients)), 2):
        plus = minus = strict_plus = strict_minus = False
        for a in ingredients[i]:
            for b in ingredients[j]:
                union = (a | b) in targets
                intersection = (a & b) in targets
                plus |= union
                minus |= intersection
                if a not in targets and b not in targets:
                    strict_plus |= union
                    strict_minus |= intersection
        flags = np.array([plus, minus, strict_plus, strict_minus], dtype=int)
        counts[:4] += flags
        if parents_free[i] and parents_free[j]:
            counts[4:] += flags
    return Joins(*map(int, counts))


@dataclass(frozen=True)
class CrossoverOpportunity:
    target_present: bool
    available: bool
    new_available: bool
    p_uniform: float
    p_tournament: float
    p_uniform_new: float
    p_tournament_new: float
    p_uniform_parents_free: float
    p_tournament_parents_free: float
    p_uniform_given_parents_free: float | None
    p_tournament_given_parents_free: float | None
    parent_signatures: tuple[int, ...]
    parent_weights: tuple[float, ...]
    per_pair: np.ndarray
    root_joins: Joins
    subtree_joins: Joins


def _targets(value):
    raw = (value,) if isinstance(value, Integral) else tuple(value)
    if not raw or any(
        isinstance(x, (bool, np.bool_))
        or not isinstance(x, Integral)
        or not 0 <= x <= ALL
        for x in raw
    ):
        raise ValueError("Expected one or more integer 16-corner target signatures")
    return frozenset(map(int, raw))


def analyze_population(
    trees, rounded_fitness, target_bits, *, provider_names=DEFAULT_NAMES
):
    """Exact finite-site enumeration for one uncapped crossover, reported as probabilities.

    The two parent tournaments are independent, including self-mating. A site
    swap succeeds when either of its two children belongs to the target set.
    The two *_new probabilities include the probability of selecting two
    target-free parent roots. The *_given_parents_free values condition on
    doing so and are undefined (None) if that condition has zero probability.
    A join is strict when neither ingredient is itself any target.
    """
    prepared = [prepare_tree(tree, provider_names=provider_names) for tree in trees]
    n = len(prepared)
    weights = tournament_weights(rounded_fitness)
    if len(weights) != n:
        raise ValueError("Expected one tournament fitness per parent individual")
    targets = _targets(target_bits)
    roots = tuple(tree.signature for tree in prepared)
    free = np.array([sig not in targets for sig in roots])
    chances = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            groups = _offspring_pairs(prepared[i], prepared[j])
            numerator, denominator = 0, 1
            for total, histogram in groups:
                hits = sum(
                    count for a, b, count in histogram if a in targets or b in targets
                )
                numerator = numerator * total + hits * denominator
                denominator *= total
            chances[i, j] = chances[j, i] = numerator / (denominator * len(groups))
    productive = chances * (free[:, None] & free)
    p_uniform_new = float(productive.mean())
    p_tournament_new = float(weights @ productive @ weights)
    uf, tf = float(free.mean() ** 2), float(weights[free].sum() ** 2)
    return CrossoverOpportunity(
        target_present=not free.all(),
        available=bool(chances.any()),
        new_available=bool(productive.any()),
        p_uniform=float(chances.mean()),
        p_tournament=float(weights @ chances @ weights),
        p_uniform_new=p_uniform_new,
        p_tournament_new=p_tournament_new,
        p_uniform_parents_free=uf,
        p_tournament_parents_free=tf,
        p_uniform_given_parents_free=p_uniform_new / uf if uf else None,
        p_tournament_given_parents_free=p_tournament_new / tf if tf else None,
        parent_signatures=roots,
        parent_weights=tuple(map(float, weights)),
        per_pair=chances,
        root_joins=_join_counts([(sig,) for sig in roots], free, targets),
        subtree_joins=_join_counts(
            [tree.subtree_signatures for tree in prepared], free, targets
        ),
    )
