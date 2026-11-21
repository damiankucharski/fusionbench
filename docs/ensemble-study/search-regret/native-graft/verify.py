"""Recompute recorded predictions, graft neighborhood optima and search ledgers independently."""

import argparse
import json
import runpy
from collections import Counter, deque
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch


def walk(tree):
    queue = deque([(tree, ())])
    while queue:
        node, path = queue.popleft()
        yield node, path
        queue.extend((child, (*path, i)) for i, child in enumerate(node["children"]))


def node_at(tree, path):
    for i in path:
        tree = tree["children"][i]
    return tree


def graft_checks(base, chosen, regime, seed, search_seed, x, y, eval_tree, stats):
    parent = json.loads(base.program)
    sites = [
        (path, n["id"])
        for n, path in walk(parent)
        if n["type"] == "ValueNode" and not n["children"]
    ]
    generator = np.random.default_rng(
        int(np.random.SeedSequence([seed, search_seed, 937]).generate_state(1)[0])
    )
    path, old = sites[generator.integers(len(sites))]
    other = [f"p{i}.npy" for i in range(x.shape[1]) if f"p{i}.npy" != old]
    order = tuple(other[i] for i in generator.permutation(len(other)))
    left_prediction = eval_tree(parent, x)
    loss_a = (left_prediction - y) ** 2
    base_loss = float(loss_a.mean())
    a = x[:, int(old[1:-4])]
    thresholds = np.arange(257)[:, None] / 256
    for r in chosen.itertuples():
        assert tuple(json.loads(r.site_path)) == path and r.anchor == old
        assert tuple(json.loads(r.operand_order)) == order
        trained = r.arm.endswith("_tuned")
        others = order if r.arm.startswith("all_") else order[:1]
        threshold_points = events = row_events = 0
        minimum = base_loss
        for name in others:
            b = x[:, int(name[1:-4])]
            hypothetical = deepcopy(parent)
            node_at(hypothetical, path)["id"] = name
            loss_b = (eval_tree(hypothetical, x) - y) ** 2
            grid = thresholds if trained else np.array([[0.5]])
            da, db = (a - grid) ** 2, (b - grid) ** 2
            for choose_a in (da <= db, da >= db):
                minimum = min(
                    minimum, np.where(choose_a, loss_a, loss_b).mean(axis=1).min()
                )
            if trained:
                mids = np.unique((a[a != b] + b[a != b]) / 2)
                cuts = np.r_[0.0, mids, 1.0]
                candidates = np.unique(np.r_[cuts, (cuts[1:] + cuts[:-1]) / 2, 0.5])
                events += 2 * len(mids)
                row_events += 2 * np.count_nonzero(a != b)
                threshold_points += 2 * len(candidates)
        assert abs(minimum - r.graft_fit_loss) < 3e-15, (
            seed,
            regime,
            r.arm,
            minimum,
            r.graft_fit_loss,
        )
        assert r.proposal_forwards == 1 + len(others) * (8 if trained else 2)
        assert (
            r.grafts == 2 * len(others) and r.threshold_candidates == threshold_points
        )
        assert r.events == events and r.row_events == row_events
        expected = deepcopy(parent)
        if r.graft_changed:
            assert r.selected_operand in others and r.direction in ("close", "far")
            node_at(expected, path)["children"].append(
                {
                    "type": "CloseThresholdNode"
                    if r.direction == "close"
                    else "FarThresholdNode",
                    "children": [
                        {"type": "ValueNode", "children": [], "id": r.selected_operand}
                    ],
                    "threshold": r.threshold,
                    "close": r.direction == "close",
                }
            )
        assert expected == json.loads(r.program)
        stats["graft_neighborhoods"] += 1


def predictions_for_group(domains, top, stack, evaluate):
    @cache
    def prediction(method, encoded, split):
        draw, mix = domains[split]
        x = draw.preds[:, :, 1]
        if encoded:
            prog = json.loads(encoded)
            if "type" in prog:
                return evaluate(prog, x)
            a, b = x[:, prog["parent"]], x[:, prog["child"]]
            da, db = (a - prog["threshold"]) ** 2, (b - prog["threshold"]) ** 2
            return np.where(
                da <= db if prog["direction"] == "close" else da >= db, a, b
            )
        if method == "known_q":
            return draw.probability(mix)
        if method == "average":
            return x.mean(1)
        return (top if method == "top_n" else stack).predict(draw.preds)[:, 1]

    return prediction


def verify(path):
    decoder = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "native-switch" / "verify.py")
    )
    evaluate = decoder["evaluate"]
    rows = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    stats = Counter()
    max_error = 0.0
    for (seed, regime), group in rows.groupby(["data_seed", "fit_regime"]):
        fit, val, test = [
            draw_native_switch(seed, i, n) for i, n in enumerate((512, 256, 8192))
        ]
        mix = float(regime == "linear")
        domains = {
            "fit": (fit, mix),
            "validation": (val, mix),
            "switch_test": (test, 0),
            "weak_test": (test, 0.5),
            "linear_test": (test, 1),
        }
        y = fit.labels(mix)
        top = TopN("brier").fit(fit.preds, y, task="binary", n_classes=2)
        stack = StackingLogit().fit(fit.preds, y, task="binary", n_classes=2)
        probe = group[group.method == "top_n"].iloc[0]
        assert (
            json.loads(probe.ranking) == top.ranking_.tolist()
            and probe.selected_n == top.n_selected_
        )
        np.testing.assert_array_equal(
            json.loads(probe.prefix_scores), top.prefix_scores_
        )
        exact = decoder["literal_optimum"](fit.preds[:, :, 1], y, False)
        gate = json.loads(group[group.method == "gate_free_threshold"].iloc[0].program)
        assert exact == (
            gate["loss_numerator"],
            gate["minimizers_on_grid"],
            (gate["parent"], gate["child"], gate["direction"], gate["threshold"]),
        )
        stats["exact_single_gate_fits"] += 1
        predict = predictions_for_group(domains, top, stack, evaluate)
        for r in group.itertuples():
            draw, k = domains[r.split]
            out = predict(
                r.method, r.program if isinstance(r.program, str) else "", r.split
            )
            err = max(
                abs(np.mean((out - draw.labels(k)) ** 2) - r.observed_brier),
                abs(np.mean((out - draw.probability(k)) ** 2) - r.conditional_excess),
            )
            assert err < 1e-13, (r.method, r.split, err)
            max_error = max(max_error, err)
            stats["brier_values"] += 2
        for gp_seed, tra in group[
            (group.split == "fit") & (group.decision == "fit")
        ].groupby("gp_seed"):
            base = tra[tra.method == "gp_initial"].iloc[0]
            chosen = tra[tra.method == "immediate"]
            graft_checks(
                base,
                chosen,
                regime,
                seed,
                int(gp_seed),
                fit.preds[:, :, 1],
                y,
                evaluate,
                stats,
            )
            full = chosen[chosen.arm == "all_tuned"].iloc[0]
            get = lambda name, frame=tra: frame[frame.method == name].iloc[0]
            active, sham = get("active_tail"), get("archive_only_tail")
            assert active.ordinary_calls == sham.ordinary_calls == base.base_calls + 300
            assert sham.path_sha256 == get("plain_tail").path_sha256
            assert active.graft_status in (
                "unchanged",
                "duplicate_parent",
                "rejected",
                "admitted",
            )
            assert active.full_graft_inserted == (active.graft_status == "admitted")
            if not active.full_graft_inserted:
                assert (
                    active.path_sha256 == sham.path_sha256
                    and active.program == sham.program
                )
            for r in chosen.itertuples():
                for match, extra in (
                    ("matched_forwards", r.proposal_forwards),
                    (
                        "matched_candidates",
                        r.proposal_forwards + r.threshold_candidates,
                    ),
                ):
                    line = tra[(tra.method == match) & (tra.arm == r.arm)].iloc[0]
                    assert line.ordinary_calls == base.base_calls + extra
                    stats["immediate_cost_matches"] += 1
            for name, extra in (
                ("matched_forwards_tail", full.proposal_forwards),
                (
                    "matched_candidates_tail",
                    full.proposal_forwards + full.threshold_candidates,
                ),
            ):
                assert get(name).ordinary_calls == base.base_calls + extra + 300
                stats["tail_cost_matches"] += 1
    # Post hoc structural description of archived winners after the frozen comparisons.
    take = rows[
        (rows.fit_regime == "switch")
        & (rows.split == "switch_test")
        & (rows.decision == "fit")
    ]
    originals = take[take.method == "gp_initial"].set_index(["data_seed", "gp_seed"])
    for r in take[
        take.method.isin(
            ["active_tail", "archive_only_tail", "matched_candidates_tail"]
        )
    ].itertuples():
        nodes = list(walk(json.loads(r.program)))
        if len(nodes) == 3 and nodes[1][0]["type"] in (
            "CloseThresholdNode",
            "FarThresholdNode",
        ):
            stats[f"simple_gate_{r.method}"] += 1
        if r.method != "active_tail" or not r.full_graft_inserted:
            continue
        original = originals.loc[r.data_seed, r.gp_seed]
        expected = (r.threshold, r.direction == "close")

        def has_mark(entries, marker):
            return any(
                (n.get("threshold"), n.get("close")) == marker for n, _ in entries
            )

        stats["admitted_offers"] += 1
        stats["new_parameter_already_in_original_winner"] += int(
            has_mark(walk(json.loads(original.program)), expected)
        )
        stats["new_parameter_in_final_winner"] += int(has_mark(nodes, expected))
    return {**stats, "max_abs_brier_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    print(json.dumps(verify(parser.parse_args().path), indent=2))
