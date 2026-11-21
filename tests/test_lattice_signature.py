import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures


def test_corner_order_and_size_guard():
    assert provider_signatures(["p0", "p1", "p2"]) == {
        "p0": 0b10101010,
        "p1": 0b11001100,
        "p2": 0b11110000,
    }
    assert provider_signatures(["p0"]) == {"p0": 0b10}
    masks = provider_signatures(range(12))
    assert masks[11] == ((1 << 2048) - 1) << 2048
    for invalid in ([], [1, 1], range(13)):
        with pytest.raises(ValueError):
            provider_signatures(invalid)


@pytest.fixture
def native(monkeypatch):
    pytest.importorskip("okapi")
    from okapi.backend.backend import Backend
    from okapi.backend.numpy_backend import NumpyBackend
    from okapi.globals import postprocessing_function

    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", np.asarray)


def cube(names):
    corners = np.arange(1 << len(names))
    return {
        name: ((corners >> i) & 1).astype(float)[:, None]
        for i, name in enumerate(names)
    }


def native_signature(tree):
    # Independent of symbolic traversal and provider masks.
    bits = tree.predict().ravel()
    assert np.all((bits == 0) | (bits == 1))
    return sum(1 << int(i) for i in np.flatnonzero(bits))


@pytest.mark.parametrize("seed", range(50))
def test_agrees_with_native_on_all_corners_of_random_tree(native, seed):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    rng = np.random.default_rng(seed)
    width = 8 if seed < 38 else seed - 37
    names = [f"p{i}.npy" for i in range(width)]
    corners = cube(names)
    masks = provider_signatures(names)

    def branch(depth):
        name = names[rng.integers(width)]
        value = ValueNode(None, corners[name], name)
        if depth == 0:
            return value
        for _ in range(rng.integers(1 if depth == 3 else 0, 3)):
            op = (MinNode, MaxNode)[rng.integers(2)]
            value.add_child(
                op([branch(depth - 1) for _ in range(rng.integers(1, 3))])
            )
        return value

    tree = Tree(branch(3))
    assert lattice_signature(tree, masks) == native_signature(tree)


def test_logged_collisions_and_positive_counterexample(native):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    names = [f"p{i}.npy" for i in range(8)]
    corners = cube(names)
    masks = provider_signatures(names)
    operators = {"MIN": MinNode, "MAX": MaxNode}

    def rebuild(record):
        code, children = record
        nodes = [rebuild(child) for child in children]
        if code in operators:
            return operators[code](nodes)
        name = code.removeprefix("VN[").removesuffix("]")
        return ValueNode(nodes, corners[name], name)

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/ensemble-study/search-regret"
        / "structural-key/collisions.csv"
    )
    with path.open() as stream:
        distinct = {
            (row["retained_tree"], row["suppressed_tree"])
            for row in csv.DictReader(stream)
        }
    assert len(distinct) == 14
    for left, right in distinct:
        a, b = [Tree(rebuild(ast.literal_eval(text))) for text in (left, right)]
        assert repr(a) == repr(b)
        assert lattice_signature(a, masks) == lattice_signature(b, masks)
        assert native_signature(a) == native_signature(b) == lattice_signature(a, masks)

    def witness(split):
        a, *leaves = [ValueNode(None, corners[name], name) for name in names[:4]]
        a.add_child(MinNode(leaves[:split]))
        a.add_child(MaxNode(leaves[split:]))
        return Tree(a)

    left, right = witness(2), witness(1)
    assert repr(left) == repr(right)
    assert lattice_signature(left, masks) == native_signature(left)
    assert lattice_signature(right, masks) == native_signature(right)
    assert lattice_signature(left, masks) != lattice_signature(right, masks)


def test_rejects_non_lattice_operator(native):
    from okapi.node import MeanNode, ValueNode
    from okapi.tree import Tree

    value = np.ones((1, 1))
    root = ValueNode([MeanNode([ValueNode(None, value, "b")])], value, "a")
    with pytest.raises(TypeError, match="MeanNode"):
        lattice_signature(Tree(root), provider_signatures(["a", "b"]))
