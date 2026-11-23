"""Construct and select nested, native one-update prunings of frozen OKAPI trees."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from okapi.interventions import operator_paths, tree_with_bypass
from scipy.stats import t

from fusionbench.eval.full_suite_point import SHIFTS, fingerprint, scenario_block
from fusionbench.eval.gate_necessity import native_tree
from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.registry import SCENARIOS

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "mixed-deep-init"


def node_cost(tree):
    return len({n.id for n in tree.nodes["value_nodes"]}), tree.nodes_count


def brier(pred, labels):
    return np.square(pred - np.eye(pred.shape[1])[labels]).sum(axis=1).mean().item()


def loss(tree, data):
    x, y, _ = data
    return brier(predict_tree(tree, x), y)


def pruning_path(tree, fit, prune):
    programs = [tree]
    cuts = [None]
    fit_losses = [loss(tree, fit)]
    fit_evaluations = 1
    while tree.nodes["op_nodes"]:
        best = None
        for path, _ in operator_paths(tree):
            candidate = tree_with_bypass(tree, [path])
            value = loss(candidate, fit)
            fit_evaluations += 1
            key = (value, *node_cost(candidate))
            if best is None or key < best[0]:
                best = key, candidate, path
        key, tree, path = best
        programs.append(tree)
        cuts.append(path)
        fit_losses.append(key[0])
    steps = []
    for i, candidate in enumerate(programs):
        providers, nodes = node_cost(candidate)
        steps.append(
            {
                "step": i,
                "cut": cuts[i],
                "providers": providers,
                "nodes": nodes,
                "fit_brier": fit_losses[i],
                "prune_brier": loss(candidate, prune),
            }
        )
    allowed = [s for s in steps if s["prune_brier"] <= steps[0]["prune_brier"] + 0.005]
    selected = min(
        allowed, key=lambda s: (s["providers"], s["nodes"], s["prune_brier"])
    )["step"]
    return programs, steps, selected, fit_evaluations


def record(key, test, predictor, *, providers, nodes=None):
    rows = []
    for split, (x, y, _) in test.items():
        p = predictor(x)
        if not np.isfinite(p).all() or not np.allclose(
            p.sum(axis=1), 1, atol=2e-12, rtol=0
        ):
            raise ValueError("Invalid probabilities")
        rows.append(
            dict(
                **key,
                split=split,
                providers=providers,
                nodes=nodes,
                brier=brier(p, y),
                prediction_sha256=fingerprint(p),
            )
        )
    return rows


def controls(name, bank, training, test):
    c = training[0][0].shape[-1]
    m = training[0][0].shape[1]
    key = {"scenario": name, "data_seed": bank, "gp_seed": None, "arm": "baseline"}
    rows = record(
        {**key, "method": "mean"}, test, lambda x: x.mean(axis=1), providers=m
    )
    prefixes = []
    for size, blocks in [(640, training[:2]), (896, training)]:
        x = np.concatenate([d[0] for d in blocks])
        y = np.concatenate([d[1] for d in blocks])
        assert len(x) == size
        task = "binary" if c == 2 else "multiclass"
        for label, model in [("top_n", TopN("brier")), ("stack", StackingLogit())]:
            model.fit(x, y, task=task, n_classes=c)
            rows.extend(
                record(
                    {**key, "method": f"{label}_{size}"},
                    test,
                    model.predict,
                    providers=model.n_selected_ if label == "top_n" else m,
                )
            )
            if label == "top_n":
                assert model.n_selected_ == 1 + np.argmin(model.prefix_scores_)
                prefixes.append(
                    {
                        "scenario": name,
                        "data_seed": bank,
                        "labels": size,
                        "selected": int(model.n_selected_),
                        "ranking": model.ranking_.tolist(),
                        "prefix_brier": model.prefix_scores_.tolist(),
                    }
                )
    return rows, prefixes


def panel(job):
    logger.disable("okapi")
    (name, bank), sources = job
    factory = SCENARIOS[name]
    scenario = factory()
    fit = scenario_block(scenario, bank, 0, 384)
    validation = scenario_block(scenario, bank, 1, 256)
    prune = scenario_block(scenario, bank, 2301, 256)
    manifests, frozen = [], []
    for source, reference in sources:
        key = {
            "scenario": name,
            "data_seed": bank,
            "gp_seed": source["gp_seed"],
            "arm": source["method"],
        }
        tree = native_tree(source["program"], _bind(fit[0]))
        if node_cost(tree) != (source["providers"], source["nodes"]):
            raise AssertionError("Restored program costs changed")
        replay_gap = abs(loss(tree, validation) - reference)
        if replay_gap > 1e-9:
            raise AssertionError(
                f"Original validation loss mismatch {key}: {replay_gap}"
            )
        programs, steps, selected, evaluations = pruning_path(tree, fit, prune)
        manifests.append(
            dict(
                **key,
                selected=selected,
                endpoint=len(steps) - 1,
                replay_gap=replay_gap,
                fit_evaluations=evaluations,
                steps=steps,
            )
        )
        for label, i in [
            ("intact", 0),
            ("pruned", selected),
            ("endpoint", len(steps) - 1),
        ]:
            frozen.append(({**key, "method": label}, programs[i]))

    test = {"iid": scenario_block(scenario, bank, 2302, 4096)}
    if name in SHIFTS:
        shifted = factory(**SHIFTS[name])
        assert shifted.model_ids() == scenario.model_ids()
        test["shift"] = scenario_block(shifted, bank, 2303, 4096)
    rows, prefixes = controls(name, bank, [fit, validation, prune], test)
    for key, tree in frozen:
        providers, nodes = node_cost(tree)
        rows.extend(
            record(
                key,
                test,
                lambda x, model=tree: predict_tree(model, x),
                providers=providers,
                nodes=nodes,
            )
        )
    return rows, manifests, prefixes


def jobs():
    reference = {}
    with gzip.open(SOURCE / "scores.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["gp_seed"] and r["split"] == "validation":
                reference[
                    (
                        r["scenario"],
                        int(r["data_seed"]),
                        int(float(r["gp_seed"])),
                        r["method"],
                    )
                ] = float(r["brier"])
    grouped = defaultdict(list)
    with gzip.open(SOURCE / "trees.jsonl.gz", "rt") as fh:
        for line in fh:
            s = json.loads(line)
            key = s["scenario"], s["data_seed"], s["gp_seed"], s["method"]
            grouped[key[:2]].append((s, reference[key]))
    assert (
        len(reference) == 480
        and len(grouped) == 80
        and all(len(g) == 6 for g in grouped.values())
    )
    return sorted(grouped.items())


def summarize(scores):
    group = ["scenario", "data_seed", "arm", "method", "split"]
    bank = scores.groupby(group)[["brier", "providers", "nodes"]].mean()
    means = bank.groupby(["scenario", "arm", "method", "split"]).mean()
    means.to_csv(OUT / "means.csv")
    contrast = []
    for metric, sign in [("brier", 1), ("providers", -1), ("nodes", -1)]:
        wide = bank[metric].unstack("method")
        diff = sign * (wide.pruned - wide.intact).dropna()
        for (name, arm, split), v in diff.groupby(["scenario", "arm", "split"]):
            avg = v.mean()
            half = t.ppf(0.975, len(v) - 1) * v.sem()
            contrast.append(
                {
                    "scenario": name,
                    "arm": arm,
                    "split": split,
                    "metric": metric,
                    "banks": len(v),
                    "mean": avg,
                    "low": avg - half,
                    "high": avg + half,
                }
            )
    pd.DataFrame(contrast).to_csv(OUT / "paired.csv", index=False)


def main(workers):
    start = perf_counter()
    rows, paths, prefixes = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, p, n in pool.map(panel, jobs()):
            rows.extend(r)
            paths.extend(p)
            prefixes.extend(n)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv.gz", index=False)
    for name, records in [("paths", paths), ("topn-prefixes", prefixes)]:
        with gzip.open(OUT / f"{name}.jsonl.gz", "wt") as fh:
            for rec in records:
                print(json.dumps(rec, separators=(",", ":")), file=fh)
    summarize(df)
    print(
        f"{len(paths)} frozen trajectories, {len(rows)} scores, {perf_counter() - start:.1f} wall seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args().workers)
