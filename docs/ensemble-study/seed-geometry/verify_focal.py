"""Independently verify the controlled topology search and replay 18 bare native trajectories."""

import argparse
import gzip
import hashlib
import json
import struct
from itertools import pairwise, product
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LEVS = np.array([0.1, 0.35, 0.65, 0.9])
TICKS = np.array([2, 7, 13, 18])
SH = np.array([0.4, 0.3, 0.2, 0.1])


def draw(law, k, seed):
    ids = np.random.default_rng([seed, k, 401]).choice(k, size=4, replace=False)

    def q(x):
        a, b, c, d = (x[:, i] for i in ids)
        if law == "branch":
            return np.maximum(np.minimum(a, b), np.minimum(c, d))
        if law == "chain":
            return np.minimum(np.maximum(np.minimum(a, b), c), d)
        return (a + b + c + d) / 4

    parts = {}
    for j, (name, n) in enumerate(
        (("fit", 256), ("validation", 256), ("iid", 2048), ("shift", 2048))
    ):
        g = np.random.default_rng([seed, k, j, 8093])
        x = LEVS[g.choice(4, (n, k), p=SH if j == 3 else None)]
        parts[name] = (x, g.random(n) < q(x))
    return ids, q, parts


def code(mask):
    return sum(1 << i for i in np.flatnonzero(mask).tolist())


def decoded(program, x):
    pred = x[:, int(program["id"][1:-4])]
    for op in program["children"]:
        func = {"MinNode": np.minimum, "MaxNode": np.maximum}[op["type"]]
        for child in op["children"]:
            pred = func(pred, decoded(child, x))
    return pred


def lift(sig, x, order=None):
    """Descending threshold sets, stopping at the first accepted prefix."""
    rank = np.argsort(-x, axis=1, kind="stable") if order is None else order
    active = np.bitwise_or.accumulate(1 << rank, axis=1)
    truth = np.frombuffer(
        int(sig).to_bytes((1 << x.shape[1]) // 8, "little"), dtype=np.uint8
    )
    first = np.argmax(np.unpackbits(truth, bitorder="little")[active], axis=1)
    return x[np.arange(len(x)), rank[np.arange(len(x)), first]]


def mincut_units(x, y):
    k = x.shape[1]
    d = 1 << k
    endpoints = (x * 20).round().astype(int)
    target = y.astype(int) * 20
    unary = np.zeros(d, dtype=int)
    base = np.sum((2 - target) ** 2)
    for lo, hi in pairwise(TICKS):
        active = ((endpoints >= hi) * (1 << np.arange(k))).sum(axis=1)
        inc = (hi - target) ** 2 - (lo - target) ** 2
        np.add.at(unary, active, inc)
    inf = int(np.sum(np.abs(unary))) + 1
    graph = nx.DiGraph()

    def edge(i, j, cost):
        graph.add_edge(
            i, j, capacity=cost + graph.get_edge_data(i, j, {}).get("capacity", 0)
        )

    for i in range(d):
        if unary[i] < 0:
            edge(d, i, int(-unary[i]))
        elif unary[i] > 0:
            edge(i, d + 1, int(unary[i]))
        for j in range(k):
            if not i >> j & 1:
                edge(i, i | 1 << j, inf)
    edge(d, d - 1, inf)
    edge(0, d + 1, inf)
    cut, _ = nx.minimum_cut(
        graph, d, d + 1, flow_func=nx.algorithms.flow.shortest_augmenting_path
    )
    return int(base + np.minimum(0, unary).sum() + cut)


def units(p, y):
    return int(
        np.sum((np.rint(np.asarray(p) * 20).astype(int) - 20 * y.astype(int)) ** 2)
    )


def normalized(x):
    x = np.clip(x, 1e-12, None)
    return x / x.sum(axis=1, keepdims=True)


def features(x):
    p = np.stack((1 - x, x), axis=-1).reshape(len(x), -1)
    return np.log(p / (1 - p))


def refs(a, y):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    )
    model.fit(features(a), y)
    truth = np.eye(2)[y.astype(int)]

    def scores(z):
        return np.mean(np.sum((normalized(z) - truth) ** 2, axis=1))

    packed = np.stack([1 - a, a], axis=-1)
    ranking = np.argsort(
        [scores(packed[:, i]) for i in range(a.shape[1])], kind="stable"
    )
    total = np.zeros((len(y), 2))
    loss = []
    for n, i in enumerate(ranking, 1):
        total += packed[:, i]
        loss.append(scores(total / n))
    chosen = 1 + np.argmin(loss)

    def top(x):
        prob = np.stack([1 - x, x], axis=-1)
        total = np.zeros((len(x), 2))
        for i in ranking[:chosen]:
            total += prob[:, i]
        return normalized(total / chosen)[:, 1]

    return (
        {
            "mean": lambda x: x.mean(axis=1),
            "top_n": top,
            "stack_logit": lambda x: normalized(model.predict_proba(features(x)))[:, 1],
        },
        ranking,
        loss,
    )


def recreate_seeds(engine, shape, seed):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    rng = np.random.default_rng(seed)
    for _ in range(30):
        n = rng.integers(2, 5) if shape == "stock" else 4
        pos = rng.choice(len(engine.ids), size=n, replace=False)
        v = [ValueNode(None, engine.models[i], engine.ids[i]) for i in pos]
        op = (MinNode, MaxNode)[int(rng.integers(2))]
        other = MaxNode if op is MinNode else MinNode
        a = v[0]
        if shape == "stock":
            a.add_child(op(v[1:]))
        elif shape == "branch":
            a.add_child(op([v[1]]))
            v[2].add_child(op([v[3]]))
            a.add_child(other([v[2]]))
        else:
            a.add_child(op([v[1]]))
            a.add_child(other([v[2]]))
            a.add_child(op([v[3]]))
        engine.population.append(Tree(a))
    engine.population_size = min(40, engine.population_size + 30)
    engine.fitnesses = None


def native_replay(k, parts, q, expected, seeds):
    from okapi import Okapi
    from okapi.globals import use_postprocessing_function
    from okapi.operators import MAX, MIN

    xf, y = parts["fit"]
    xv, yv = parts["validation"]
    corners = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(float)
    bind = {f"p{i}.npy": corners[:, i : i + 1] for i in range(k)}
    efforts = selected = 0
    with TemporaryDirectory() as d, use_postprocessing_function(np.asarray):
        root = Path(d)
        survey = [root / f"p{i}.npy" for i in range(k)]
        for i, path in enumerate(survey):
            np.save(path, xf[:, i : i + 1])
        label = root / "labels.npy"
        np.save(label, y[:, None])

        def arm(shape, rows):
            digest = hashlib.sha256()
            archive = {}
            expected_at = {}
            prev = None
            offset = calls = 0

            def instrumented_score(p, _):
                nonlocal calls, prev, offset
                if prev is not native.population:
                    prev = native.population
                    offset = 0
                tree = prev[offset]
                offset += 1
                calls += 1
                loss = units(p.ravel(), y)
                key = tree.structural_key()
                digest.update((repr(key) + "\n").encode())
                digest.update(struct.pack("<I", loss))
                if calls <= k + 30:
                    assert (
                        repr(key) in seeds[shape]["seed_structures"]
                        if calls > k
                        else tree.nodes_count == 1
                    )
                # For the bare driver the semantic signature is read from genuine native predictions.
                sig = code(
                    tree.do_pred_on_another_tensors(current_tensors=bind).ravel()
                )
                if sig not in archive:
                    archive[sig] = (loss, units(lift(sig, xv), yv), calls)
                if calls in (k + 30, 600, 1200):
                    best_fit = min(
                        archive, key=lambda i: (archive[i][0], archive[i][2])
                    )
                    best_val = min(
                        archive,
                        key=lambda i: (archive[i][1], archive[i][0], archive[i][2]),
                    )
                    expected_at[calls] = (
                        digest.hexdigest(),
                        best_fit,
                        best_val,
                        len(archive),
                    )
                return 1.0 - loss / (400 * len(y))

            native = Okapi(
                survey,
                label,
                k,
                3,
                5,
                seed=907,
                backend="numpy",
                allowed_ops=(MIN, MAX),
                objective_functions=(instrumented_score,),
                postprocessing_function=None,
                minimize_node_count=True,
                reuse_survivor_fitness=False,
                operator_point_mutation=False,
            )
            recreate_seeds(native, shape, 907)
            while calls < 1200:
                native.run_iteration()
            for row in rows.itertuples():
                h, bf, bv, archive_size = expected_at[row.calls]
                assert row.trajectory_sha256 == h and int(row.selected_hex, 16) == (
                    bf if row.selector == "fit" else bv
                )
                assert row.unique_semantics == archive_size
            return len(expected_at), len(rows)

        for shape, rows in expected.groupby("method"):
            a, b = arm(shape, rows)
            efforts += a
            selected += b
    return efforts, selected


def audit(folder, historical):
    from loguru import logger

    logger.disable("okapi")
    raw = pd.read_csv(folder / "focal-runs.csv.gz", float_precision="round_trip")
    initial = {}
    with gzip.open(folder / "focal-seeds.jsonl.gz", "rt") as file:
        for line in file:
            z = json.loads(line)
            key = (z["law"], z["width"], z["data_seed"], z["method"], z["gp_seed"])
            initial[key] = z
    source = historical / "okapi" / "lattice_search.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "0ace77b3a6d0db4eecf673d6d867d95f950fb43b51b12e76d907feafcd5e3573"
    )
    historic = run_path(str(source))["fit_small_lattice"]
    native_cp = native_selected = enumerated = certified = 0
    largest = 0.0
    metric_count = 0
    assert len(initial) == 432
    for (k, seed, law), rows in raw.groupby(["width", "data_seed", "law"]):
        roles, q, parts = draw(law, k, seed)
        xf, yf = parts["fit"]
        xv, yv = parts["validation"]
        population = np.asarray(list(product(LEVS, repeat=k)))
        order = np.argsort(-population, axis=1, kind="stable")
        corner = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(float)
        qq = q(population)
        target = None if law == "additive" else code(q(corner))
        probs = np.prod(np.take(SH, np.searchsorted(LEVS, population)), axis=1)
        floor = mincut_units(xf, yf)
        certified += 1
        if k == 4:
            model = historic(xf, yf)
            assert model.n_candidates == 166 and units(model.predict(xf), yf) == floor
            enumerated += 1
        assert rows.cut_optimal_units.eq(floor).all()
        baselines, rank, prefix = refs(xf, yf)
        known = {}
        for row in rows.itertuples():
            assert json.loads(row.roles) == roles.tolist()
            if row.method == "top_n":
                assert np.array_equal(rank, json.loads(row.ranking))
                np.testing.assert_allclose(
                    prefix, json.loads(row.prefix_scores), atol=2e-13, rtol=0
                )
            if pd.notna(row.program):
                program = json.loads(row.program)
                assert code(decoded(program, corner)) == int(row.selected_hex, 16)
                assert units(decoded(program, xf), yf) == row.selected_fit_units
                assert units(decoded(program, xv), yv) == row.selected_val_units
                z = initial[(law, k, seed, row.method, int(row.gp_seed))]
                assert len(z["seed_signatures"]) == len(z["seed_structures"]) == 30
                assert row.proposed_semantic_count == len(set(z["seed_signatures"]))
                assert row.initial_semantic_count == len(
                    set(z["singleton_signatures"] + z["seed_signatures"])
                )
                if target is not None:
                    assert bool(row.true_in_initial) == (
                        hex(target) in z["seed_signatures"] + z["singleton_signatures"]
                    )
                    assert bool(row.selected_true) == (
                        target == int(row.selected_hex, 16)
                    )
            key = row.selected_hex if pd.notna(row.selected_hex) else row.method
            if key not in known:
                if key in baselines:
                    p = baselines[key](population)
                    sample_pred = {
                        name: baselines[key](x) for name, (x, y) in parts.items()
                    }
                else:
                    sig = int(key, 16)
                    p = lift(sig, population, order)
                    sample_pred = {name: lift(sig, x) for name, (x, y) in parts.items()}
                residual = (p - qq) ** 2
                bayes = qq * (1 - qq)
                metrics = {
                    "iid_excess": np.mean(residual),
                    "shift_excess": np.sum(probs * residual),
                    "iid_risk": np.mean(residual + bayes),
                    "shift_risk": np.sum(probs * (residual + bayes)),
                }
                metrics.update(
                    {
                        part + "_brier": np.mean((arr - parts[part][1]) ** 2)
                        for part, arr in sample_pred.items()
                    }
                )
                metrics["fit_minus_cut"] = metrics["fit_brier"] - floor / (
                    400 * len(yf)
                )
                known[key] = (
                    hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest(),
                    metrics,
                )
            h, metrics = known[key]
            assert row.prediction_sha256 == h, (law, k, seed, key)
            for attr, v in metrics.items():
                err = abs(v - getattr(row, attr))
                assert err < 5e-13, (attr, err)
                metric_count += 1
                largest = max(largest, err)
        if seed == 7200:
            gp_rows = rows[rows.gp_seed.eq(907)]
            where = {
                m: initial[(law, k, seed, m, 907)] for m in ("stock", "branch", "chain")
            }
            c, s = native_replay(k, parts, q, gp_rows, where)
            native_cp += c
            native_selected += s
    contrasts = pd.read_csv(folder / "focal-paired.csv", float_precision="round_trip")
    gp = raw[raw.gp_seed.notna()]
    total = 0
    for (law, k, cp, sel), group in contrasts.groupby(
        ["law", "width", "checkpoint", "selector"]
    ):
        here = gp[
            gp.law.eq(law) & gp.width.eq(k) & gp.checkpoint.eq(cp) & gp.selector.eq(sel)
        ]
        for r in group.itertuples():
            bank = here.groupby(["data_seed", "method"])[r.metric].mean().unstack()
            diff = (bank[r.left] - bank[r.right]).dropna().values
            mean = diff.mean()
            half = (
                t.ppf(0.975, len(diff) - 1) * np.std(diff, ddof=1) / np.sqrt(len(diff))
            )
            np.testing.assert_allclose(
                [mean, mean - half, mean + half],
                [r.difference, r.ci_low, r.ci_high],
                atol=1e-11,
                rtol=1e-12,
            )
            total += 1
    return {
        "runs_sha256": hashlib.sha256(
            (folder / "focal-runs.csv.gz").read_bytes()
        ).hexdigest(),
        "banks_cells": certified,
        "historical_enumerations": enumerated,
        "native_trajectories": native_cp // 3,
        "native_prefixes": native_cp,
        "native_archive_winners": native_selected,
        "prediction_vectors": len(raw),
        "scalar_metrics": metric_count,
        "max_metric_error": largest,
        "paired_intervals": total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--historical-okapi", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        evidence = audit(args.folder, args.historical_okapi)
    (args.folder / "focal-verification.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
