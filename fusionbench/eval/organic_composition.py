"""Separate operator-family and input-identity point mutations during native GP.

Langdon, Barrett and Buxton (2002, §5.5) used arity-preserving point mutation
inside a larger mix of moves for classifier fusion. The factorial control and
the mixed-cube certificate are specific to this benchmark. See the dated
entry and langdon2002combining in docs/ensemble-study/literature/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.gate_necessity import native_tree
from fusionbench.eval.native_switch import serialize, tensors
from fusionbench.eval.search_continuous import brier
from fusionbench.eval.structural_key_audit import training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.compositional_switch import LAWS, CompositionalSwitch

ARMS = ("stock", "type_only", "provider_only", "both")
ONE_UPDATE_BOUND = 961 / 16384


def propose(tree, path, child_index, rng, providers, arm):
    from okapi.interventions import operator_paths, tree_with_operand_provider
    from okapi.mutation import tree_with_operator_type
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )
    from okapi.random_state import use_rng

    from fusionbench.eval.organic_gp import Edit

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    op = dict(operator_paths(tree))[path]
    old_id = op.children[child_index].id
    types = [
        t
        for t in (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
        if t is not type(op)
    ]
    other = [k for k in sorted(providers) if k != old_id]
    family = types[rng.randint(len(types))]
    partner = other[rng.randint(len(other))]
    # Draw parameters even when the counterfactual family is inactive.
    # use_rng restores OKAPI's ordinary stream before the next native action.
    with use_rng(rng):
        typed = tree_with_operator_type(tree, path, family)
    result = typed if arm in ("type_only", "both") else tree
    if arm in ("provider_only", "both"):
        result = tree_with_operand_provider(
            result, path, child_index, partner, providers[partner]
        )
    return Edit(result, family.__name__, partner)


def composition_program(roles):
    def val(i, *ops):
        return {"type": "ValueNode", "id": f"p{i}.npy", "children": list(ops)}

    def op(kind, child):
        return {"type": kind, "children": [child]}

    a, b, c, d = roles
    return val(a, op("MinNode", val(b)), op("MaxNode", val(c, op("MinNode", val(d)))))


def summarize(runs):
    index = ["fit_law", "split", "test_law", "decision", "checkpoint"]
    selected = runs[runs.gp_seed.notna()]
    contrasts = []
    for keys, g in selected.groupby(index, sort=False):
        for metric in ("conditional_excess", "observed_brier"):
            group = g.pivot(
                index=["data_seed", "gp_seed"], columns="method", values=metric
            )
            comparisons = {
                arm + "_minus_stock": group[arm] - group.stock for arm in ARMS[1:]
            }
            comparisons.update(
                type_minus_provider=group.type_only - group.provider_only,
                interaction=group.both
                - group.type_only
                - group.provider_only
                + group.stock,
            )
            if (
                keys[0] == "composition"
                and keys[1] == "iid"
                and keys[2] == "composition"
                and metric == "conditional_excess"
            ):
                comparisons.update(
                    {
                        arm + "_minus_single_update_bound": group[arm]
                        - ONE_UPDATE_BOUND
                        for arm in ARMS
                    }
                )
            for name, diffs in comparisons.items():
                bank = diffs.groupby("data_seed").mean()
                center = bank.mean()
                half = t.ppf(0.975, len(bank) - 1) * bank.sem()
                contrasts.append(
                    dict(zip(index, keys, strict=True))
                    | {
                        "contrast": name,
                        "metric": metric,
                        "banks": len(bank),
                        "starts": len(diffs),
                        "difference": center,
                        "low_95": center - half,
                        "high_95": center + half,
                    }
                )
    means = (
        runs.groupby([*index, "method", "data_seed"], dropna=False)[
            ["observed_brier", "conditional_excess", "nodes", "calls"]
        ]
        .mean()
        .groupby([*index, "method"], dropna=False)
        .mean()
        .reset_index()
    )
    return means, pd.DataFrame(contrasts)


def panel(seed):
    from loguru import logger
    from okapi.backend.backend import Backend
    from okapi.globals import set_postprocessing_function

    from fusionbench.eval.organic_gp import organic_gp

    logger.disable("okapi")
    Backend.set_backend("numpy")
    set_postprocessing_function(np.asarray)
    case = CompositionalSwitch(role_seed=seed)
    splits = {
        name: case.draw(n, np.random.default_rng([seed, i]), p)
        for i, (name, n, p) in enumerate(
            (
                ("fit", 512, 0.5),
                ("validation", 256, 0.5),
                ("iid", 8192, 0.5),
                ("interior", 8192, 0),
            )
        )
    }
    bind = {name: tensors(draw.scores) for name, draw in splits.items()}
    y = {law: {k: d.labels(law) for k, d in splits.items()} for law in LAWS}
    q = {law: {k: d.probability(law) for k, d in splits.items()} for law in LAWS}
    witness = composition_program(case.roles)
    tree = native_tree(witness, bind["fit"])
    reference = {
        k: tree.do_pred_on_another_tensors(current_tensors=store).ravel()
        for k, store in bind.items()
    }
    for k in splits:
        np.testing.assert_array_equal(reference[k], q["composition"][k])
    rows = []
    events = []
    ledger = []

    def score(fit_law, method, vectors, **meta):
        for split, p in vectors.items():
            digest = hashlib.sha256(
                np.asarray(p, dtype="<f8").ravel().tobytes()
            ).hexdigest()
            for test_law in (fit_law,) if split in ("fit", "validation") else LAWS:
                rows.append(
                    dict(
                        data_seed=seed,
                        fit_law=fit_law,
                        split=split,
                        test_law=test_law,
                        method=method,
                        prediction_sha256=digest,
                        observed_brier=brier(p, y[test_law][split]),
                        conditional_excess=brier(p, q[test_law][split]),
                        **meta,
                    )
                )

    for law in LAWS:
        fit, labels = splits["fit"], y[law]["fit"]
        top = TopN("brier").fit(fit.preds, labels, task="binary", n_classes=2)
        lr = StackingLogit().fit(fit.preds, labels, task="binary", n_classes=2)
        fixed = {"decision": "fit", "checkpoint": "fixed"}
        score(law, "average", {k: d.scores.mean(1) for k, d in splits.items()}, **fixed)
        score(
            law,
            "top_n",
            {k: top.predict(d.preds)[:, 1] for k, d in splits.items()},
            **fixed,
            n_selected=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(top.prefix_scores_.tolist()),
        )
        score(
            law,
            "stack_logit",
            {k: lr.predict(d.preds)[:, 1] for k, d in splits.items()},
            **fixed,
        )
        score(
            law,
            "achievable_composition",
            reference,
            **fixed,
            nodes=tree.nodes_count,
            program=json.dumps(witness),
        )
        score(law, "known_law", q[law], **fixed)
        with training_files(fit.scores, labels) as (paths, gt):
            for gp_seed in (331, 337):
                event_seed = int(
                    np.random.SeedSequence([seed, gp_seed, 419]).generate_state(1)[0]
                )
                for arm in ARMS:
                    result = organic_gp(
                        paths,
                        gt,
                        bind["validation"],
                        y[law]["validation"],
                        gp_seed,
                        arm,
                        event_seed=event_seed,
                        transform=propose,
                    )
                    common = {
                        "data_seed": seed,
                        "fit_law": law,
                        "gp_seed": gp_seed,
                        "arm": arm,
                    }
                    events.extend({**common, **event} for event in result.events)
                    for name, cp in result.checkpoints.items():
                        entry = (
                            asdict(cp.ledger) if is_dataclass(cp.ledger) else cp.ledger
                        )
                        ledger.append(
                            {
                                **common,
                                "checkpoint": name,
                                "calls": cp.calls,
                                "generation": cp.generation,
                                **entry,
                            }
                        )
                        for decision, tmodel in (
                            ("fit", cp.fit_tree),
                            ("validation", cp.validation_tree),
                        ):
                            p = {
                                k: tmodel.do_pred_on_another_tensors(
                                    current_tensors=m
                                ).ravel()
                                for k, m in bind.items()
                            }
                            score(
                                law,
                                arm,
                                p,
                                decision=decision,
                                checkpoint=name,
                                gp_seed=gp_seed,
                                calls=cp.calls,
                                generation=cp.generation,
                                archive_trees=cp.n_trees,
                                search_seconds=cp.elapsed_seconds,
                                path_digest=cp.path_digest,
                                nodes=tmodel.nodes_count,
                                operators=len(tmodel.op_nodes),
                                program=json.dumps(serialize(tmodel.root)),
                            )
    return rows, events, ledger


def run(out, workers):
    out.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(panel, range(4200, 4212)))
    runs, events, ledger = [
        pd.DataFrame([row for batch in batches for row in batch[i]]) for i in range(3)
    ]
    runs.to_csv(out / "runs.csv.gz", index=False)
    events.to_csv(out / "events.csv.gz", index=False)
    ledger.to_csv(out / "costs.csv.gz", index=False)
    means, paired = summarize(runs)
    means.to_csv(out / "means.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=1)
    a = cli.parse_args()
    run(a.out, a.workers)
