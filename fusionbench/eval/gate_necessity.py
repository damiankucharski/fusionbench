"""Evaluate frozen, actually inherited threshold updates on fresh scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from fusionbench.eval.native_switch import tensors
from fusionbench.eval.search_continuous import brier
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch


def native_tree(program, provider_tensors):
    """Restore an exact ordered native program without drawing any new parameters."""
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

    ordinary = {"MeanNode": MeanNode, "MaxNode": MaxNode, "MinNode": MinNode}
    gates = {
        "CloseThresholdNode": CloseThresholdNode,
        "FarThresholdNode": FarThresholdNode,
    }

    def restore(record):
        children = [restore(child) for child in record["children"]]
        kind = record["type"]
        if kind == "ValueNode":
            return ValueNode(children, provider_tensors[record["id"]], record["id"])
        if kind in gates:
            return gates[kind](children, record["threshold"])
        if kind == "WeightedMeanNode":
            return WeightedMeanNode(children, record["weights"])
        return ordinary[kind](children)

    return Tree(restore(program))


def laws(draw):
    return {
        "far": (draw, 0),
        "close": (replace(draw, close=True, threshold=37 / 64), 0),
        "linear": (draw, 1),
    }


def model_outputs(tree, paths):
    from okapi.gate_lineage import tree_with_gate_cutoff
    from okapi.interventions import tree_with_bypass

    return {
        "intact": tree,
        "bypassed": tree_with_bypass(tree, paths),
        "cutoff_half": tree_with_gate_cutoff(tree, paths, threshold=0.5),
    }


def panel(source, seed):
    from okapi.backend.backend import Backend
    from okapi.globals import set_postprocessing_function

    Backend.set_backend("numpy")
    set_postprocessing_function(np.asarray)
    fit = draw_native_switch(seed, 0, 512)
    test = draw_native_switch(seed, 9701, 32768)
    fit_laws = laws(fit)
    targets = {
        law: (d.labels(m), d.probability(m)) for law, (d, m) in laws(test).items()
    }
    binding = tensors(test.preds[:, :, 1])
    rows = []

    def score(prediction, *, method, relevant, source_it=None, **metadata):
        p = np.asarray(prediction).ravel()
        checksum = hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()
        delta = np.abs(p - source_it) if source_it is not None else None
        for law, (y, q) in targets.items():
            rows.append(
                dict(
                    data_seed=seed,
                    fit_regime=relevant,
                    test_regime=law,
                    method=method,
                    observed_brier=brier(p, y),
                    conditional_excess=brier(p, q),
                    prediction_sha256=checksum,
                    rows_changed=None
                    if delta is None
                    else int(np.count_nonzero(delta)),
                    max_change=None if delta is None else float(delta.max()),
                    **metadata,
                )
            )

    for law, (d, m) in fit_laws.items():
        y = d.labels(m)
        top = TopN("brier").fit(d.preds, y, task="binary", n_classes=2)
        stack = StackingLogit().fit(d.preds, y, task="binary", n_classes=2)
        score(test.preds[:, :, 1].mean(1), method="average", relevant=law)
        score(
            top.predict(test.preds)[:, 1],
            method="top_n",
            relevant=law,
            n_selected=top.n_selected_,
            ranking=json.dumps(top.ranking_.tolist()),
            prefix_scores=json.dumps(top.prefix_scores_.tolist()),
        )
        score(stack.predict(test.preds)[:, 1], method="stack_logit", relevant=law)
        for row in source[source.fit_regime == law].itertuples():
            tree = native_tree(json.loads(row.program), binding)
            intact = tree.predict().ravel()
            records = json.loads(row.winner_occurrences)
            options = {"archive_first": [tuple(o["path"]) for o in records]}
            if row.decision == "fit":
                options["first_marked_copy"] = [
                    tuple(path) for path in json.loads(row.lineage_winner_ever_paths)
                ]
            for provenance, paths in options.items():
                for edit, variant in model_outputs(tree, paths).items():
                    p = variant.predict().ravel()
                    if not paths:
                        np.testing.assert_array_equal(p, intact)
                    score(
                        p,
                        method=edit,
                        relevant=law,
                        source_it=intact,
                        gp_seed=int(row.gp_seed),
                        decision=row.decision,
                        provenance=provenance,
                        admitted=row.admitted,
                        occurrences=len(paths),
                        occurrence_paths=json.dumps(paths),
                        intact_nodes=tree.nodes_count,
                        edited_nodes=variant.nodes_count,
                    )
    return rows


def differences(frame):
    results = []
    subset = frame[frame.gp_seed.notna()]
    for keys, table in subset.groupby(
        ["fit_regime", "test_regime", "decision", "provenance"]
    ):
        pivot = table.set_index(["method", "data_seed", "gp_seed"])
        before = pivot.xs("intact")
        for selected, index in (
            ("all", before.index),
            ("marked", before[before.occurrences > 0].index),
        ):
            for edit in ("bypassed", "cutoff_half"):
                after = pivot.xs(edit).loc[index]
                for metric in ("conditional_excess", "observed_brier"):
                    delta = (
                        (after[metric] - before[metric].loc[index])
                        .groupby("data_seed")
                        .mean()
                    )
                    center = delta.mean()
                    half = (
                        t.ppf(0.975, len(delta) - 1) * delta.sem()
                        if len(delta) > 1
                        else np.nan
                    )
                    results.append(
                        dict(
                            zip(
                                ("fit_regime", "test_regime", "decision", "provenance"),
                                keys,
                                strict=True,
                            )
                        )
                        | {
                            "subset": selected,
                            "edit": edit,
                            "metric": metric,
                            "banks": len(delta),
                            "starts": len(index),
                            "difference": center,
                            "low_95": center - half,
                            "high_95": center + half,
                        }
                    )
    return pd.DataFrame(results)


def run(source, out):
    frame = pd.read_csv(source, float_precision="round_trip")
    winners = frame[(frame.split == "fit") & (frame.method == "fit_pair_fit_cut")]
    assert len(winners) == 192
    rows = pd.DataFrame(
        [
            r
            for seed, group in winners.groupby("data_seed")
            for r in panel(group, int(seed))
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "runs.csv.gz", index=False)
    differences(rows).to_csv(out / "paired.csv", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    run(args.source, args.out)
