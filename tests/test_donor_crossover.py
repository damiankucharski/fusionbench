from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement, permutations

import numpy as np
import pytest

from fusionbench.eval.donor_crossover import (
    analyze_population,
    prepare_tree,
    tournament_weights,
)
from fusionbench.eval.lattice_signature import provider_signatures
from fusionbench.eval.search_graded import SCORES

NAMES = tuple(f"p{i}.npy" for i in range(4))
MASK = provider_signatures(NAMES)
CORNERS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
INPUT = np.concatenate((CORNERS, SCORES))


def leaf(index, *ops):
    return {"type": "ValueNode", "id": NAMES[index], "children": list(ops)}


def op(name, *values):
    return {"type": name, "children": list(values)}


def native_tree(record):
    pytest.importorskip("okapi")
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    def build(entry):
        children = [build(item) for item in entry["children"]]
        if entry["type"] == "ValueNode":
            return ValueNode(
                children,
                INPUT[:, NAMES.index(entry["id"]) : NAMES.index(entry["id"]) + 1],
                entry["id"],
            )
        return {"MinNode": MinNode, "MaxNode": MaxNode}[entry["type"]](children)

    return Tree(build(record))


def at(node, path):
    for step in path:
        node = node.children[step]
    return node


def forced(tree, path, donor, donor_path):
    changed = tree.copy()
    changed.replace_at(
        at(changed.root, path), at(donor.root, donor_path).copy_subtree()
    )
    changed.update_nodes()
    out = changed.predict(clear_cache=True).ravel()
    sig = sum(1 << i for i, x in enumerate(out[:16]) if x == 1)
    binary = np.array([(sig >> i) & 1 for i in range(16)])
    levels = np.array([0.1, 0.35, 0.65, 0.9])
    grades = np.zeros(len(SCORES), dtype=int)
    for threshold in levels[1:]:
        grades += binary[((SCORES >= threshold) * (1 << np.arange(4))).sum(axis=1)]
    np.testing.assert_array_equal(out[16:], levels[grades])
    return sig


def test_tournaments_exhaust_distinct_contestant_orders_and_allow_self_mating():
    f = np.array([0.92, 0.92, 0.88, 0.8])
    w = tournament_weights(f)
    counts = Counter(i if f[i] >= f[j] else j for i, j in permutations(range(4), 2))
    np.testing.assert_array_equal(w, [counts[k] / 12 for k in range(4)])
    roots = [leaf(i) for i in range(4)]
    report = analyze_population(roots, f, MASK[NAMES[0]])
    expected = 1.0 - (1 - w[0]) ** 2
    assert report.p_tournament == pytest.approx(expected)
    assert report.target_present and report.available and not report.new_available
    assert report.p_tournament_new == 0 and report.p_uniform_new == 0
    assert (
        analyze_population(
            roots, f, list(MASK.values())
        ).p_tournament_given_parents_free
        is None
    )


def test_hypothetical_joins_do_not_imply_a_one_step_native_swap():
    trees = [leaf(0, op("MinNode", leaf(1))), leaf(2, op("MinNode", leaf(3)))]
    target = (MASK[NAMES[0]] & MASK[NAMES[1]]) | (MASK[NAMES[2]] & MASK[NAMES[3]])
    outcome = analyze_population(trees, [0.72, 0.72], target)
    assert outcome.root_joins.parents_free_strict_union_pairs == 1
    assert outcome.subtree_joins.parents_free_strict_union_pairs == 1
    assert not outcome.available and outcome.p_tournament_new == 0


def test_every_contextual_site_matches_native_forced_swaps_with_both_children():
    pytest.importorskip("okapi")
    from okapi.globals import use_postprocessing_function

    a = leaf(
        0,
        op("MinNode", leaf(1), leaf(0)),
        op("MaxNode", leaf(2, op("MinNode", leaf(3)))),
        op("MinNode", leaf(1)),
    )
    b = leaf(
        3,
        op("MaxNode", leaf(1, op("MinNode", leaf(2))), leaf(0)),
        op("MinNode", leaf(2)),
    )
    records = [a, b, leaf(1)]
    mat = [prepare_tree(r) for r in records]
    actual = [native_tree(r) for r in records]
    target = frozenset({MASK[NAMES[1]] & MASK[NAMES[2]], mat[0].signature})
    per = np.zeros((3, 3))
    found_joint = False
    with use_postprocessing_function(np.asarray):
        for i, j in combinations_with_replacement(range(3), 2):
            groups = [(mat[i].value_sites, mat[j].value_sites)]
            if mat[i].operator_sites and mat[j].operator_sites:
                groups.append((mat[i].operator_sites, mat[j].operator_sites))
            p = Fraction()
            for left, right in groups:
                assert len(left) and len(right)
                hit = 0
                for x in left:
                    for y in right:
                        first = forced(actual[i], x.path, actual[j], y.path)
                        second = forced(actual[j], y.path, actual[i], x.path)
                        assert x.with_donor(y) == first and y.with_donor(x) == second
                        if first in target and second in target:
                            found_joint = True
                        hit += first in target or second in target
                p += Fraction(hit, len(left) * len(right) * len(groups))
            per[i, j] = per[j, i] = p
    assert found_joint
    result = analyze_population(actual, [0.8, 0.9, 0.9], target)
    np.testing.assert_array_equal(result.per_pair, per)
    expected = np.array([0.0, 0.5, 0.5])
    np.testing.assert_allclose(result.parent_weights, expected)
    assert result.p_tournament == pytest.approx(expected @ per @ expected)


def test_named_encodings_and_invalid_grammars():
    p = prepare_tree(leaf(1))
    assert prepare_tree(p) is p and p.subtree_signatures == frozenset({MASK[NAMES[1]]})
    with pytest.raises(ValueError, match="common provider"):
        prepare_tree(p, provider_names=NAMES[::-1])
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_tree(leaf(0, op("MeanNode", leaf(1))))
    with pytest.raises(ValueError):
        analyze_population([leaf(0), leaf(1)], [1, 1], 1 << 16)


def test_uniform_productive_quarter_can_be_suppressed_entirely_by_tournaments():
    # Swapping b in a OR b with singleton c produces a OR c. The root swap does not.
    trees = [leaf(0, op("MaxNode", leaf(1))), leaf(2)]
    target = MASK[NAMES[0]] | MASK[NAMES[2]]
    tied = analyze_population(trees, [0.8, 0.8], target)
    np.testing.assert_array_equal(tied.per_pair, [[0.0, 0.5], [0.5, 0.0]])
    assert tied.p_uniform_new == tied.p_tournament_new == Fraction(1, 4)
    assert not tied.target_present and tied.new_available
    favored = analyze_population(trees, [0.9, 0.8], target)
    assert favored.p_tournament_new == 0
    assert favored.p_uniform_given_parents_free == 0.25
    assert favored.new_available
