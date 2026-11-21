"""Find hidden pairs, retain uncertainty, and compile a mixture of native trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.matching_panel import make_panel, stock_winners
from fusionbench.eval.matching_posterior import MatchingFamily
from fusionbench.eval.native_switch import serialize, tensors
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit

LOW, HIGH = 1 / 64, 63 / 64
LAWS = ("matching", "logistic")
SPLITS = {"two_balanced": 2, "two_uniform": 2, "three_uniform": 3, "four_uniform": 4}


@lru_cache
def family_for(k):
    return MatchingFamily(k)


@lru_cache
def test_rules(k, dimensions, encoded):
    family = family_for(k)
    z = np.frombuffer(encoded, dtype=np.bool_).reshape(dimensions)
    return np.asarray(family.contains(z), dtype=float)


def digest(p):
    return hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()


def bank(seed):
    from experiments.matching_oracle.fit import fit_matching, matching_tree
    from loguru import logger
    from okapi.backend.backend import Backend
    from okapi.globals import set_postprocessing_function

    logger.disable("okapi")
    Backend.set_backend("numpy")
    set_postprocessing_function(np.asarray)
    rows = []
    runs = []
    errors = []
    for k in (4, 6):
        family = family_for(k)
        edges = [tuple(map(int, x)) for x in family.edges]
        for design in ("balanced", "uniform"):
            data = make_panel(seed, k, design)
            true_edges = sorted(edges.index(tuple(x)) for x in data.matching)
            true_index = int(
                np.flatnonzero(np.all(family.matchings == true_edges, axis=1))[0]
            )
            for n in (16, 64):
                z = data.fit.bits[:n]
                fit = data.fit.preds[:n]
                bits = {
                    "fit": z,
                    "validation": data.validation.bits,
                    **{i: b.bits for i, b in data.test.items()},
                }
                preds = {
                    "fit": fit,
                    "validation": data.validation.preds,
                    **{i: b.preds for i, b in data.test.items()},
                }
                membership = {}
                for split, matrix in bits.items():
                    if isinstance(split, int):
                        a = np.ascontiguousarray(matrix, dtype=np.bool_)
                        membership[split] = test_rules(k, a.shape, a.tobytes())
                    else:
                        membership[split] = family.contains(matrix)
                for law in LAWS:
                    y = data.fit.y[law][:n]
                    experiment = {
                        "data_seed": seed,
                        "k": k,
                        "train_design": design,
                        "labels": n,
                        "fit_law": law,
                    }

                    def score(
                        method,
                        vectors,
                        *,
                        decision="fit",
                        gp_seed=None,
                        data=data,
                        law=law,
                        n=n,
                        y=y,
                        experiment=experiment,
                        **extra,
                    ):
                        for name, vec in vectors.items():
                            p = np.asarray(vec).ravel()
                            fingerprints = digest(p)
                            blocks = (
                                [name]
                                if isinstance(name, str)
                                else [label for label, c in SPLITS.items() if c == name]
                            )
                            for target in blocks:
                                if target == "fit":
                                    q = data.fit.q[law][:n]
                                    labels = y
                                    w = None
                                elif target == "validation":
                                    q = data.validation.q[law]
                                    labels = data.validation.y[law]
                                    w = None
                                else:
                                    block = data.test[name]
                                    q = block.q[law]
                                    labels = block.y[law]
                                    w = data.weights[target]
                                row_loss = (
                                    (p - labels) ** 2
                                    if labels.ndim == 1
                                    else ((p[:, None] - labels) ** 2).mean(1)
                                )
                                rows.append(
                                    {
                                        **experiment,
                                        "method": method,
                                        "decision": decision,
                                        "gp_seed": gp_seed,
                                        "split": target,
                                        "conditional_excess": np.average(
                                            (p - q) ** 2, weights=w
                                        ),
                                        "observed_brier": np.average(
                                            row_loss, weights=w
                                        ),
                                        "prediction_sha256": fingerprints,
                                        **extra,
                                    }
                                )

                    top = TopN("brier").fit(fit, y, task="binary", n_classes=2)
                    score("average", {s: p[:, :, 1].mean(1) for s, p in preds.items()})
                    score(
                        "top_n",
                        {s: top.predict(p)[:, 1] for s, p in preds.items()},
                        n_selected=top.n_selected_,
                        ranking=json.dumps(top.ranking_.tolist()),
                        prefix_scores=json.dumps(top.prefix_scores_.tolist()),
                    )
                    classes = np.unique(y)
                    if len(classes) == 1:
                        p = {s: np.full(len(pp), classes[0]) for s, pp in preds.items()}
                    else:
                        stack = StackingLogit().fit(fit, y, task="binary", n_classes=2)
                        p = {s: stack.predict(pp)[:, 1] for s, pp in preds.items()}
                    score("stack_logit", p, one_class=len(classes) == 1)
                    fit_report = fit_matching(fit[:, :, 1], y)
                    choice = family.fit_erm(z, y, low=LOW, high=HIGH)
                    chosen_edges = family.matchings[choice]
                    chosen_pairs = [edges[i] for i in chosen_edges]
                    assert tuple(chosen_pairs) == fit_report.pairs
                    score(
                        "matching_map",
                        {
                            s: LOW + (HIGH - LOW) * m[:, choice]
                            for s, m in membership.items()
                        },
                        ties=fit_report.n_optimal,
                        correct_edges=len(set(chosen_edges) & set(true_edges)),
                        model=json.dumps(chosen_pairs),
                        dp_states=fit_report.n_states,
                    )
                    # Compare against an independently compiled and evaluated native MIN/MAX tree.
                    for split in bits:
                        tree = matching_tree(fit_report.pairs, preds[split][:, :, 1].T)
                        np.testing.assert_array_equal(
                            tree.predict().ravel(),
                            LOW + (HIGH - LOW) * membership[split][:, choice],
                        )
                    for mode in ("prior", "conditional", "x_only", "joint"):
                        if mode == "prior":
                            # Empty observed rows are a uniform prior over matchings.
                            post = family.fit(
                                z[:0],
                                y[:0],
                                z[:0],
                                pi=data.pi,
                                low=LOW,
                                high=HIGH,
                                mode="conditional",
                            )
                        else:
                            post = family.fit(
                                z,
                                y,
                                data.unlabeled.bits,
                                pi=data.pi,
                                low=LOW,
                                high=HIGH,
                                mode=mode,
                            )
                        if mode in ("conditional", "joint"):
                            assert (
                                post.fit_tie_count == fit_report.n_optimal
                                and post.erm_index == choice
                            )
                        explicit = {
                            s: LOW + (HIGH - LOW) * (m @ post.weights)
                            for s, m in membership.items()
                        }
                        score(
                            mode,
                            explicit,
                            posterior_true=post.weights[true_index],
                            expected_correct_edges=post.edge_weights[true_edges].sum(),
                            effective_models=1 / (post.weights @ post.weights),
                        )
                        for order in (1, 2):
                            compiled = {
                                s: post.predict(zz, order=order)
                                for s, zz in bits.items()
                            }
                            for s, p in compiled.items():
                                d = np.max(abs(p - explicit[s]))
                                dc = np.max(abs(np.clip(p, 0, 1) - explicit[s]))
                                if s != 4 or order == 2:
                                    assert d < 1e-12, (experiment, mode, s, order, d)
                                errors.append(
                                    {
                                        **experiment,
                                        "mode": mode,
                                        "order": order,
                                        "support": str(s),
                                        "states": len(p),
                                        "max_abs": d,
                                        "clipped_max_abs": dc,
                                    }
                                )
                            if mode == "joint" and order == 1:
                                score("joint_pair_network", compiled)
                            if mode == "joint" and order == 2:
                                score("joint_pair_quad_network", compiled)
                    score(
                        "known_law",
                        {
                            "fit": data.fit.q[law][:n],
                            "validation": data.validation.q[law],
                            **{s: sample.q[law] for s, sample in data.test.items()},
                        },
                    )
                    for gp_seed in (521, 523):
                        out = stock_winners(
                            fit,
                            y,
                            data.validation.preds,
                            data.validation.y[law],
                            gp_seed,
                        )
                        runs.append(
                            {
                                **experiment,
                                "gp_seed": gp_seed,
                                "calls": out.calls,
                                "archive_trees": out.archive_trees,
                                "elapsed_seconds": out.elapsed_seconds,
                                "path_sha256": out.path_digest,
                                "generation": out.max_generation,
                                "fit_brier": out.fit_brier,
                                "validation_brier": out.validation_brier,
                            }
                        )
                        for decision, tree in (
                            ("fit", out.fit_tree),
                            ("validation", out.validation_tree),
                        ):
                            predictions = {
                                s: tree.do_pred_on_another_tensors(
                                    current_tensors=tensors(pp[:, :, 1])
                                ).ravel()
                                for s, pp in preds.items()
                            }
                            score(
                                "stock_gp",
                                predictions,
                                decision=decision,
                                gp_seed=gp_seed,
                                nodes=tree.nodes_count,
                                model=json.dumps(serialize(tree.root)),
                            )
    return rows, runs, errors


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(bank, range(4400, 4416)))
    results, gp, checks = [
        pd.DataFrame([r for batch in batches for r in batch[i]]) for i in range(3)
    ]
    results.to_csv(out / "runs.csv.gz", index=False)
    gp.to_csv(out / "gp.csv.gz", index=False)
    checks.to_csv(out / "compilation.csv.gz", index=False)
    keys = ["k", "train_design", "labels", "fit_law", "split", "decision"]
    means = (
        results.groupby([*keys, "method", "data_seed"])[
            ["conditional_excess", "observed_brier"]
        ]
        .mean()
        .groupby([*keys, "method"])
        .mean()
        .reset_index()
    )
    means.to_csv(out / "means.csv", index=False)
    paired = []
    for key, df in results[(results.decision == "fit")].groupby(keys):
        for metric in ["conditional_excess", "observed_brier"]:
            wide = df.pivot_table(index="data_seed", columns="method", values=metric)
            for a, b in [
                ("joint", "conditional"),
                ("x_only", "prior"),
                ("joint", "matching_map"),
                ("joint", "stock_gp"),
                ("joint", "stack_logit"),
                ("joint_pair_network", "joint"),
            ]:
                delta = wide[a] - wide[b]
                center = delta.mean()
                half = t.ppf(0.975, len(delta) - 1) * delta.sem()
                paired.append(
                    dict(zip(keys, key, strict=True))
                    | {
                        "metric": metric,
                        "method": a,
                        "comparator": b,
                        "banks": len(delta),
                        "difference": center,
                        "low_95": center - half,
                        "high_95": center + half,
                    }
                )
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    args = cli.parse_args()
    run(args.out, args.workers)
