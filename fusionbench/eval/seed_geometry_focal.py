"""Compare shallow, chain and branched native GP seeds on controlled graded inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from fusionbench.eval.lattice_brier_oracle import fit_lattice_brier
from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from fusionbench.eval.native_switch import serialize
from fusionbench.eval.seed_geometry import append_geometry_seeds, make_seed_plans
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.stacking import StackingLogit

LEVELS = np.array([0.1, 0.35, 0.65, 0.9])
TICKS = np.array([2, 7, 13, 18])
SHIFT = np.array([0.4, 0.3, 0.2, 0.1])
HISTORICAL_SHA = "0ace77b3a6d0db4eecf673d6d867d95f950fb43b51b12e76d907feafcd5e3573"
LAWS = ("branch", "chain", "additive")
SHAPES = ("stock", "chain", "branch")
STARTS = (907, 919)


def law_probability(x, roles, law):
    a, b, c, d = x[:, roles].T
    if law == "branch":
        return np.maximum(np.minimum(a, b), np.minimum(c, d))
    if law == "chain":
        return np.minimum(np.maximum(np.minimum(a, b), c), d)
    if law == "additive":
        return (a + b + c + d) / 4
    raise ValueError(f"Unknown focal law {law}")


def target_signature(k, roles, law):
    if law == "additive":
        return None
    bits = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(float)
    return sum(
        1 << j for j in np.flatnonzero(law_probability(bits, roles, law)).tolist()
    )


def threshold_masks(x):
    bits = 1 << np.arange(x.shape[1])
    return [(x >= threshold) @ bits for threshold in LEVELS[1:]]


def signature_values(signature, width, masks):
    table = np.unpackbits(
        np.frombuffer(
            int(signature).to_bytes((1 << width) // 8, "little"), dtype=np.uint8
        ),
        bitorder="little",
    )
    return LEVELS[table[masks[0]] + table[masks[1]] + table[masks[2]]]


@lru_cache(maxsize=2)
def support(width):
    digits = np.array(list(product(range(4), repeat=width)), dtype=np.uint8)
    x = LEVELS[digits]
    return x, threshold_masks(x), SHIFT[digits].prod(axis=1)


def integer_brier(pred, y):
    delta = np.rint(20 * np.asarray(pred)).astype(np.int64) - 20 * y.astype(np.int64)
    return int(delta @ delta)


def data_for(width, seed, law):
    roles = np.random.default_rng([seed, width, 401]).choice(width, 4, replace=False)
    splits = {}
    for j, (part, n) in enumerate(
        (("fit", 256), ("validation", 256), ("iid", 2048), ("shift", 2048))
    ):
        rng = np.random.default_rng([seed, width, j, 8093])
        digits = rng.choice(4, size=(n, width), p=SHIFT if part == "shift" else None)
        x = LEVELS[digits]
        y = rng.random(n) < law_probability(x, roles, law)
        splits[part] = x, y
    return roles, splits


def public_seeds(engine, shape, seed):
    if shape == "stock":
        OKAPIMethod(operator_family="minmax", n_seeds=30)._seed_population(
            engine, np.random.default_rng(seed)
        )
    else:
        plans = make_seed_plans(engine.ids, seed, count=30)
        append_geometry_seeds(engine, plans, shape)


@dataclass(frozen=True)
class Candidate:
    signature: int
    fit: int
    val: int
    first_call: int
    nodes: int
    program: str


def search(paths, label_file, xfit, yfit, xval, yval, seed, shape, budgets=(600, 1200)):
    """The trajectory depends only on fitting rows; validation scores never enter selection."""
    from loguru import logger
    from okapi import Okapi
    from okapi.globals import use_postprocessing_function
    from okapi.operators import MAX, MIN

    logger.disable("okapi")
    k = xfit.shape[1]
    initial = k + 30
    checkpoints = (initial, *budgets)
    if min(budgets) <= initial or tuple(sorted(set(budgets))) != tuple(budgets):
        raise ValueError(
            "Checkpoints must be increasing and strictly after the initial population"
        )
    names = tuple(f"p{i}.npy" for i in range(k))
    masks = provider_signatures(names)
    valmasks = threshold_masks(xval)
    archive = {}
    snapshots = {}
    signatures = {}
    trace = hashlib.sha256()
    calls = 0
    first_batch = []
    iterator = iter(())
    start = time.perf_counter()

    class End(Exception):
        pass

    class Observed(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal iterator
            trees = self.population if trees is None else trees
            iterator = iter(trees)
            return super()._calculate_fitnesses(trees)

    def objective(p, _):
        nonlocal calls
        tree = next(iterator)
        key = tree.structural_key()
        if key not in signatures:
            signatures[key] = lattice_signature(tree, masks)
        signature = signatures[key]
        fit = integer_brier(p.ravel(), yfit)
        calls += 1
        trace.update((repr(key) + "\n").encode())
        trace.update(struct.pack("<I", fit))
        if signature not in archive:
            fitted = Candidate(
                signature,
                fit,
                integer_brier(signature_values(signature, k, valmasks), yval),
                calls,
                tree.nodes_count,
                json.dumps(serialize(tree.root), separators=(",", ":")),
            )
            archive[signature] = fitted
        elif archive[signature].fit != fit:
            raise AssertionError(
                "The same semantic function acquired two fitting losses"
            )
        if calls <= initial:
            first_batch.append(signature)
        if calls in checkpoints:
            # All known functions are eligible. Visit order resolves remaining exact ties.
            fit_best = min(archive.values(), key=lambda row: row.fit)
            val_best = min(archive.values(), key=lambda row: (row.val, row.fit))
            snapshots[calls] = {
                "fit": fit_best,
                "validation": val_best,
                "unique_semantics": len(archive),
                "unique_structures": len(signatures),
                "trajectory_sha256": trace.hexdigest(),
                "seconds": time.perf_counter() - start,
            }
        if calls == checkpoints[-1]:
            raise End
        return 1 - fit / (400 * len(yfit))

    with use_postprocessing_function(np.asarray):
        gp = Observed(
            paths,
            label_file,
            k,
            3,
            5,
            backend="numpy",
            seed=seed,
            allowed_ops=(MIN, MAX),
            minimize_node_count=True,
            objective_functions=(objective,),
            operator_point_mutation=False,
            reuse_survivor_fitness=False,
            selection_fitness_decimals=3,
        )
        public_seeds(gp, shape, seed)
        if len(gp.population) != initial or gp.population_size != initial:
            raise AssertionError(
                "Expected all singletons followed by exactly 30 native seeds"
            )
        initial_structures = [repr(tree.structural_key()) for tree in gp.population]
        try:
            while calls < checkpoints[-1]:
                gp.run_iteration()
        except End:
            pass
    if set(snapshots) != set(checkpoints):
        raise AssertionError("Missing a paid native checkpoint")
    return snapshots, archive, first_batch, initial_structures


@lru_cache(maxsize=2)
def historical_enumerator(checkout):
    path = Path(checkout) / "okapi" / "lattice_search.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != HISTORICAL_SHA:
        raise ValueError("Expected the literal OKAPI lattice_search.py from c614670")
    return run_path(str(path))["fit_small_lattice"]


def as_proba(x):
    return np.stack([1 - x, x], axis=-1)


def panel(job):
    law, width, seed, old_checkout = job
    roles, splits = data_for(width, seed, law)
    xfit, yfit = splits["fit"]
    xval, yval = splits["validation"]
    atoms, grid_masks, shift = support(width)
    q = law_probability(atoms, roles, law)
    target = target_signature(width, roles, law)
    cut = fit_lattice_brier(xfit, yfit)
    cut_sig = sum(1 << i for i in np.flatnonzero(cut.lower).tolist())
    cut_units = integer_brier(cut.predict(xfit), yfit)
    if abs(cut_units / (400 * len(yfit)) - float(cut.minimum_loss)) > 1e-12:
        raise AssertionError(
            "The empirical cut certificate did not attain its integer score"
        )
    enum_units = None
    enum_sig = None
    if width == 4:
        exact = historical_enumerator(old_checkout)(xfit, yfit)
        enum_units = integer_brier(exact.predict(xfit), yfit)
        corners = ((np.arange(1 << width)[:, None] >> np.arange(width)) & 1).astype(
            float
        )
        enum_sig = sum(1 << i for i in np.flatnonzero(exact.predict(corners)).tolist())
        if enum_units != cut_units or exact.n_candidates != 166:
            raise AssertionError(
                "Historical OKAPI enumeration and weighted min-cut disagree"
            )
    rows = []
    seeds = []
    base = {
        "law": law,
        "width": width,
        "data_seed": seed,
        "roles": json.dumps(roles.tolist()),
        "target_hex": hex(target) if target is not None else None,
        "cut_optimal_units": cut_units,
        "cut_rule_hex": hex(cut_sig),
        "historical_optimal_units": enum_units,
        "historical_rule_hex": hex(enum_sig) if enum_sig is not None else None,
    }
    bayes_noise = q * (1 - q)
    split_masks = {name: threshold_masks(x) for name, (x, _) in splits.items()}
    cached_metrics = {}

    def evaluate(prob, selected_signature=None):
        if selected_signature is not None and selected_signature in cached_metrics:
            return cached_metrics[selected_signature]
        if selected_signature is not None:
            atlas = signature_values(selected_signature, width, grid_masks)
            scored = {
                name: signature_values(selected_signature, width, m)
                for name, m in split_masks.items()
            }
        else:
            atlas = prob(atoms)
            scored = {name: prob(x) for name, (x, _) in splits.items()}
        err = (atlas - q) ** 2
        metrics = {
            "iid_excess": float(err.mean()),
            "shift_excess": float(shift @ err),
            "iid_risk": float(np.mean(err + bayes_noise)),
            "shift_risk": float(shift @ (err + bayes_noise)),
            "prediction_sha256": hashlib.sha256(
                np.asarray(atlas, dtype="<f8").tobytes()
            ).hexdigest(),
        }
        for name, (_, y) in splits.items():
            metrics[name + "_brier"] = float(np.mean((scored[name] - y) ** 2))
        metrics["fit_minus_cut"] = metrics["fit_brier"] - cut_units / (400 * len(yfit))
        if selected_signature is not None:
            cached_metrics[selected_signature] = metrics
        return metrics

    def reference(method, metrics, **details):
        rows.append(
            {
                **base,
                "method": method,
                "checkpoint": "fixed",
                "selector": "fit",
                **metrics,
                **details,
            }
        )

    reference("exact_cut", evaluate(None, cut_sig), selected_hex=hex(cut_sig))
    if enum_sig is not None:
        reference(
            "historical_erm", evaluate(None, enum_sig), selected_hex=hex(enum_sig)
        )
    reference("mean", evaluate(lambda x: x.mean(axis=1)))
    packed = as_proba(xfit)
    top = TopN("brier").fit(packed, yfit, task="binary", n_classes=2)
    stack = StackingLogit().fit(packed, yfit, task="binary", n_classes=2)
    reference(
        "top_n",
        evaluate(lambda x: top.predict(as_proba(x))[:, 1]),
        n_selected=top.n_selected_,
        ranking=json.dumps(top.ranking_.tolist()),
        prefix_scores=json.dumps(top.prefix_scores_.tolist()),
    )
    reference("stack_logit", evaluate(lambda x: stack.predict(as_proba(x))[:, 1]))

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = [tmp / f"p{i}.npy" for i in range(width)]
        for i, path in enumerate(paths):
            np.save(path, xfit[:, i : i + 1])
        label_file = tmp / "labels.npy"
        np.save(label_file, yfit[:, None].astype(int))
        for gp_seed, shape in product(STARTS, SHAPES):
            snaps, archive, initial, structures = search(
                paths, label_file, xfit, yfit, xval, yval, gp_seed, shape
            )
            first = archive[target].first_call if target in archive else None
            paid = len(initial)
            seeds.append(
                {
                    **base,
                    "method": shape,
                    "gp_seed": gp_seed,
                    "initial_calls": paid,
                    "seed_signatures": [hex(i) for i in initial[width:]],
                    "seed_structures": structures[width:],
                    "singleton_signatures": [hex(i) for i in initial[:width]],
                    "initial_semantic_count": len(set(initial)),
                    "proposed_semantic_count": len(set(initial[width:])),
                }
            )
            for calls, snap in snaps.items():
                stamp = "initial" if calls == paid else str(calls)
                for selector in ("fit", "validation"):
                    winner = snap[selector]
                    rows.append(
                        {
                            **base,
                            "method": shape,
                            "checkpoint": stamp,
                            "calls": calls,
                            "gp_seed": gp_seed,
                            "selector": selector,
                            "selected_hex": hex(winner.signature),
                            "program": winner.program,
                            "selected_fit_units": winner.fit,
                            "selected_val_units": winner.val,
                            "first_selected_visit": winner.first_call,
                            "selected_nodes": winner.nodes,
                            "selected_true": winner.signature == target
                            if target is not None
                            else None,
                            "true_available": first is not None and first <= calls
                            if target is not None
                            else None,
                            "first_true_call": first
                            if first is not None and first <= calls
                            else None,
                            "true_in_initial": target in initial
                            if target is not None
                            else None,
                            "first_true_after_initial": paid < first <= calls
                            if first is not None
                            else (False if target is not None else None),
                            "initial_semantic_count": len(set(initial)),
                            "proposed_semantic_count": len(set(initial[width:])),
                            "unique_semantics": snap["unique_semantics"],
                            "validation_functions": snap["unique_semantics"],
                            "unique_structures": snap["unique_structures"],
                            "trajectory_sha256": snap["trajectory_sha256"],
                            "seconds": snap["seconds"],
                            **evaluate(None, winner.signature),
                        }
                    )
    return rows, seeds


def summarize(raw):
    keys = ["law", "width", "method", "checkpoint", "selector"]
    metrics = [
        "fit_brier",
        "validation_brier",
        "iid_brier",
        "shift_brier",
        "iid_risk",
        "shift_risk",
        "iid_excess",
        "shift_excess",
        "fit_minus_cut",
        "true_available",
        "true_in_initial",
        "selected_true",
        "first_true_after_initial",
        "unique_semantics",
        "unique_structures",
        "proposed_semantic_count",
        "seconds",
    ]
    for flag in (
        "true_available",
        "true_in_initial",
        "selected_true",
        "first_true_after_initial",
    ):
        raw[flag] = pd.to_numeric(raw[flag], errors="coerce")
    by_bank = raw.groupby(keys + ["data_seed"], dropna=False)[metrics].mean()
    means = (
        by_bank.groupby(level=list(range(len(keys))), dropna=False).mean().reset_index()
    )
    pairs = []
    for grouping, block in raw[raw.method.isin(SHAPES)].groupby(
        ["law", "width", "checkpoint", "selector"]
    ):
        for key in metrics:
            pivot = block.pivot(
                index=["data_seed", "gp_seed"], columns="method", values=key
            )
            for left, right in [
                ("branch", "chain"),
                ("branch", "stock"),
                ("chain", "stock"),
            ]:
                d = (pivot[left] - pivot[right]).groupby("data_seed").mean().dropna()
                if len(d) < 2:
                    continue
                estimate = float(d.mean())
                half = float(d.sem() * t.ppf(0.975, len(d) - 1))
                pairs.append(
                    {
                        "law": grouping[0],
                        "width": grouping[1],
                        "checkpoint": grouping[2],
                        "selector": grouping[3],
                        "metric": key,
                        "left": left,
                        "right": right,
                        "banks": len(d),
                        "difference": estimate,
                        "ci_low": estimate - half,
                        "ci_high": estimate + half,
                    }
                )
    return means, pd.DataFrame(pairs)


def _job(spec):
    with threadpool_limits(1):
        return panel(spec)


def run(output, old_checkout, workers):
    import gzip

    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    jobs = [
        (law, k, s, str(Path(old_checkout).resolve()))
        for s, k, law in product(range(7200, 7212), (4, 8), LAWS)
    ]
    rows = []
    seeds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for block, initial in pool.map(_job, jobs):
            rows.extend(block)
            seeds.extend(initial)
    raw = pd.DataFrame(rows)
    means, paired = summarize(raw)
    raw.to_csv(path / "focal-runs.csv.gz", index=False)
    means.to_csv(path / "focal-means.csv", index=False)
    paired.to_csv(path / "focal-paired.csv", index=False)
    with gzip.open(path / "focal-seeds.jsonl.gz", "wt") as file:
        for entry in seeds:
            file.write(json.dumps(entry, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical-okapi", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.out, args.historical_okapi, args.workers)
