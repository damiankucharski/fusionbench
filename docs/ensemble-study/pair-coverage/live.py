"""Pair probes as scored offspring in the fixed-budget arms of series 1 and 3."""

import hashlib
import json
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from okapi import Okapi
from okapi.node import ValueNode
from okapi.pair_coverage import pair_schedule, pair_tree
from okapi.population import choose_pareto_then_proximity

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.native_pattern import _full_probability
from fusionbench.methods.okapi import OKAPIMethod, _task_postprocessing
from fusionbench.scenarios.native_switch import draw_native_switch
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
make_founder = run_path(str(HERE.parent / "seeded-checkpoint/native_seed.py"))[
    "make_founder"
]


def fingerprint(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def schedule(m, seed, panel):
    if not panel:
        return pair_schedule(
            m, 8, seed=seed + 4_000_000, policy="unique", operators=("min",)
        )
    per_op = []
    for op_id, op in enumerate(("min", "max")):
        stream = int(np.random.SeedSequence([seed, op_id, 6103]).generate_state(1)[0])
        per_op.append(
            pair_schedule(m, 4, seed=stream, policy="unique", operators=(op,))
        )
    return tuple(item for duo in zip(*per_op, strict=True) for item in duo)


def fit(x, y, seed, probes):
    """Perform real scoring for every slot, including unchanged proposals. No audit inputs."""
    _, m, c = x.shape
    adapter = OKAPIMethod(seed=seed)
    task = "binary" if c == 2 else "multiclass"
    target = np.eye(c)[np.asarray(y).reshape(-1)]
    n_calls = 0
    risks = {}

    def score(tree):
        nonlocal n_calls
        p = _full_probability(tree.predict(clear_cache=True))
        risk = float(np.mean(np.sum((p - target) ** 2, axis=1)))
        n_calls += 1
        risks[tree.structural_key()] = risk

    def selection_scores(trees):
        raw = np.array([risks[t.structural_key()] for t in trees])
        return np.round(1 - 0.5 * raw, 3).reshape(-1, 1)

    try:
        directory = adapter._write_split(x, "fit")
        gt = Path(adapter._tmp.name) / "gt.npy"
        np.save(gt, y)
        with _task_postprocessing(task):
            gp = Okapi(
                directory,
                gt,
                min(8, m),
                2,
                2,
                seed=seed,
                backend="numpy",
                max_depth=8,
                max_nodes=max(31, m + 1),
            )
            gp.population.append(make_founder(gp.train_tensors, "seeded"))
            for tree in gp.population:
                score(tree)
            n_founders = n_calls
            founders = fingerprint(tuple(t.structural_key() for t in gp.population))
            gp.fitnesses = selection_scores(gp.population)
            roots = [
                ValueNode(None, gp.train_tensors[f"m{i}.npy"], f"m{i}.npy")
                for i in range(m)
            ]
            trace = []
            for generation in range(1, 17):
                offered = [prop.tree for prop in gp.propose_offspring(8)]
                assert len(offered) == 8
                inserted = None
                if probes and generation % 2:
                    inserted = probes[(generation - 1) // 2]
                    offered[0] = pair_tree(roots, inserted)
                for tree in offered:
                    score(tree)
                unique = {}
                for tree in [*gp.population, *offered]:
                    unique.setdefault(tree.structural_key(), tree)
                pool = list(unique.values())
                gp.population, gp.fitnesses = choose_pareto_then_proximity(
                    pool,
                    selection_scores(pool),
                    gp.objectives,
                    gp.population_size,
                    True,
                )
                keys = tuple(t.structural_key() for t in gp.population)
                trace.append(
                    {
                        "generation": generation,
                        "objective_calls": n_calls,
                        "inserted": repr(inserted) if inserted else "",
                        "probe_active": offered[0].structural_key() in keys
                        if inserted
                        else None,
                        "survivor_digest": fingerprint(keys),
                    }
                )
            winner = min(
                gp.population, key=lambda t: (risks[t.structural_key()], t.nodes_count)
            )
            assert n_calls == n_founders + 128
            return (
                winner,
                {
                    "n_founders": n_founders,
                    "calls": n_calls,
                    "unique_scored": len(risks),
                    "founder_digest": founders,
                    "winner_digest": fingerprint(winner.structural_key()),
                    "fit_brier": risks[winner.structural_key()],
                    "nodes": winner.nodes_count,
                },
                trace,
            )
    finally:
        adapter._tmp.cleanup()


def banks():
    for seed in range(241100, 241132):
        train, audit = (
            draw_native_switch(seed, split, n) for split, n in ((1, 192), (2, 4096))
        )
        for law in ("min", "singleton"):
            qs = []
            for draw in (train, audit):
                p = draw.preds[:, draw.parent, 1]
                qs.append(
                    np.minimum(p, draw.preds[:, draw.child, 1]) if law == "min" else p
                )
            yield (
                law,
                seed,
                train.preds,
                (train.uniform < qs[0]).astype(int),
                audit.preds,
                (audit.uniform < qs[1]).astype(int),
                qs[1],
            )
    for family in load_preset("standard-v4")["scenarios"]:
        for seed in range(243100, 243104):
            bank = generate_dataset(family, seed=seed, n_per_split=512)
            yield (
                family,
                seed,
                bank.fit_preds,
                bank.fit_y,
                bank.test_preds,
                bank.test_y,
                None,
            )


def summarize(scores):
    summary = []
    for cohort, subset in scores.groupby("cohort", sort=False):
        metric = "test_brier" if cohort == "panel" else "test_excess"
        cells = subset.pivot(index=["family", "seed"], columns="arm", values=metric)
        delta = (cells["probes"] - cells["ordinary"]).unstack("seed").to_numpy()
        nf, ns = delta.shape
        assert (nf, ns) == ((25, 4) if cohort == "panel" else (1, 32))
        draws = np.random.default_rng(320141).integers(ns, size=(10_000, nf, ns))
        lo, hi = np.quantile(
            delta[np.arange(nf)[None, :, None], draws].mean((1, 2)), [0.025, 0.975]
        )
        summary.append(
            {
                "cohort": cohort,
                "metric": metric,
                "ordinary": cells.ordinary.mean(),
                "probes": cells.probes.mean(),
                "difference": delta.mean(),
                "low": lo,
                "high": hi,
                "better": int(np.sum(delta < -1e-12)),
                "worse": int(np.sum(delta > 1e-12)),
                "tied": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        )
    return pd.DataFrame(summary)


def main():
    started = perf_counter()
    scores, traces = [], []
    for family, seed, x, y, test, test_y, q in banks():
        panel = q is None
        search_seed = 0 if panel else seed + 1_000_000
        menu = schedule(x.shape[1], seed, panel)
        for arm, probes in (("ordinary", ()), ("probes", menu)):
            tree, info, trace = fit(x, y, search_seed, probes)
            p = predict_tree(tree, test)
            context = {
                "cohort": "panel" if panel else family,
                "family": family,
                "seed": seed,
                "arm": arm,
            }
            fit_risk = brier(predict_tree(tree, x), y)
            np.testing.assert_allclose(fit_risk, info["fit_brier"], rtol=0, atol=1e-12)
            scores.append(
                dict(
                    **context,
                    **info,
                    test_brier=brier(p, test_y),
                    test_excess=None if panel else 2 * np.mean((p[:, 1] - q) ** 2),
                    exact=None
                    if panel
                    else np.allclose(p[:, 1], q, rtol=0, atol=1e-12),
                )
            )
            traces.extend({**context, **row} for row in trace)
        assert scores[-2]["calls"] == scores[-1]["calls"]
        assert scores[-2]["founder_digest"] == scores[-1]["founder_digest"]
    scores = pd.DataFrame(scores)
    assert len(scores) == 328
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS / "live-scores.csv", index=False)
    pd.DataFrame(traces).to_csv(
        RESULTS / "live-generations.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    summary = summarize(scores)
    summary.to_csv(RESULTS / "live-contrasts.csv", index=False)
    (RESULTS / "live-run.json").write_text(
        json.dumps(
            {
                "seconds": perf_counter() - started,
                "search_fits": len(scores),
                "workers": 1,
                "generations": len(traces),
                "bootstrap_seed": 320141,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
