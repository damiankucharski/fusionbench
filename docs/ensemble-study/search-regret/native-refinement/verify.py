"""Reconstruct losses and audit refinement choices against the actual native search."""

import argparse
import csv
import gzip
import json
import subprocess
from collections import Counter, defaultdict, deque
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from loguru import logger

from fusionbench.eval.native_switch import search, serialize, tensors
from fusionbench.eval.structural_key_audit import training_files
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from fusionbench.scenarios.native_switch import draw_native_switch

METHODS = ("gp_initial", "refined", "gp_matched_forwards", "gp_matched_candidates")
ARMS = ("operand", "threshold", "combined")
PIN = "6daff8c41234fffecf604734b71caba2b25f0ac3"


def integer(field):
    number = float(field)
    assert number.is_integer()
    return int(number)


def walk(program):
    pending = deque([program])
    while pending:
        node = pending.popleft()
        yield node
        pending.extend(node["children"])


def brier(p, truth):
    return float(np.mean((p - truth) ** 2))


def recompute_operands(program, x, truth, evaluate):
    nodes = [n for n in walk(program) if n["type"] == "ValueNode"]
    incumbent = best_loss = brier(evaluate(program, x), truth)
    winner = None
    for i, node in enumerate(nodes):
        old = node["id"]
        for j in range(x.shape[1]):
            new = f"p{j}.npy"
            if new == old:
                continue
            node["id"] = new
            loss = brier(evaluate(program, x), truth)
            if loss < best_loss:
                best_loss, winner = loss, (i, new)
        node["id"] = old
    if winner:
        nodes[winner[0]]["id"] = winner[1]
    return program, len(nodes) * (x.shape[1] - 1) + 1, incumbent, best_loss


def verify_edits(by_model, evaluate, stats, x, y):
    def row(method, arm, decision="fit", split="fit"):
        return by_model[method, arm, decision, split]

    def program(*args):
        return json.loads(row(*args)["program"])

    base = program("gp_initial", "none")
    op, op_cost, base_loss, op_loss = recompute_operands(
        program("gp_initial", "none"), x, y, evaluate
    )
    actual_op = program("refined", "operand")
    assert op == actual_op, "The best assignment, including tie order, must reproduce."
    stats["assignment_candidates"] += op_cost - 1
    stats["assignment_winners"] += 1
    for arm in ARMS:
        raw = row("refined", arm)
        actual = json.loads(raw["program"])
        selected = row("refined", arm, "validation_guard")
        source = actual_op if arm == "combined" else base
        gates = [node for node in walk(source) if "threshold" in node]
        stats[f"{arm}_with_gate" if gates else f"{arm}_no_gate"] += 1
        gate_cost = 3 + len(gates[0]["children"]) if gates else 0
        expected = (op_cost if arm != "threshold" else 0) + (
            gate_cost if arm != "operand" else 0
        )
        assert expected == integer(raw["forwards"])
        if arm != "operand":
            if gates:
                original_nodes = [n for n in walk(source) if n["type"] != "ValueNode"]
                assert integer(raw["index"]) == original_nodes.index(gates[0])
                assert gates[0]["threshold"] == float(raw["old_threshold"])
                edited_copy = json.loads(json.dumps(source))
                first = next(node for node in walk(edited_copy) if "threshold" in node)
                first["threshold"] = float(raw["new_threshold"])
                assert edited_copy == actual
                assert integer(raw["candidates"]) > 0
            else:
                assert actual == source and integer(raw["candidates"]) == 0
        else:
            assert integer(raw["candidates"]) == 0
        loss = brier(evaluate(actual, x), y)
        assert loss <= base_loss + 2e-15
        if arm == "combined":
            assert loss <= op_loss + 2e-15
        status = "improved" if loss < base_loss else "equal"
        stats[f"{arm}_{status}"] += 1
        stats[f"{arm}_identical"] += actual == base
        stats[f"{arm}_no_gate_changes"] += not gates and actual != base
        stats[f"{arm}_guard_accept"] += selected["validation_accepts"] == "True"
        for method, charge in (
            ("gp_matched_forwards", 0),
            ("gp_matched_candidates", integer(raw["candidates"])),
        ):
            target = 1200 + expected + charge
            matched = row(method, arm)
            assert target == integer(matched["calls"])
            stats["matched_checkpoints"] += 1


def replay_matched(paths, gt, seed, targets):
    """Read every true objective in the unmodified selection loop up to the last target."""
    from okapi import Okapi
    from okapi.operators import (
        CLOSE_THRESHOLD,
        FAR_THRESHOLD,
        MAX,
        MEAN,
        MIN,
        WEIGHTED_MEAN,
    )

    class Done(Exception):
        pass

    class Observe(Okapi):
        def _calculate_fitnesses(self, trees=None):
            nonlocal calls, best_loss, best
            queue = self.population if trees is None else trees
            batch = queue[: limit - calls]
            raw_losses.clear()
            scores = super()._calculate_fitnesses(batch)
            for tree, loss in zip(batch, raw_losses, strict=True):
                calls += 1
                archive.add(tree.structural_key())
                if loss < best_loss:
                    best_loss, best = loss, tree.copy()
                if calls in targets:
                    result[calls] = serialize(best.root), len(archive)
            if calls == limit:
                raise Done
            return scores

    def objective(pred, truth):
        loss = brier(pred.ravel(), truth.ravel())
        raw_losses.append(loss)
        return 1 - loss

    limit, calls, best_loss, best = max(targets), 0, np.inf, None
    raw_losses, result, archive = [], {}, set()
    runner = Observe(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        allowed_ops=(MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        objective_functions=(objective,),
        postprocessing_function=np.asarray,
        reuse_survivor_fitness=False,
    )
    try:
        while True:
            runner.run_iteration()
    except Done:
        pass
    assert set(result) == set(targets)
    return result


def native_winners(paths, labels, val_x, val_y, gp_seed, champ_rows, stats):
    records = search(paths, labels, tensors(val_x), val_y, gp_seed, True)
    _, archived, _, fit, _ = records[-1]
    initial = champ_rows["gp_initial", "none", "fit", "fit"]
    assert serialize(fit.root) == json.loads(initial["program"])
    assert archived == integer(initial["n_archive"])
    stats["native_initial_1200"] += 1
    matched = [
        r
        for (method, _, decision, split), r in champ_rows.items()
        if method.startswith("gp_matched") and decision == "fit" and split == "fit"
    ]
    assert len(matched) == 6
    target = {integer(r["calls"]) for r in matched}
    replay = replay_matched(paths, labels, gp_seed, target)
    for r in matched:
        result, archived = replay[integer(r["calls"])]
        assert result == json.loads(r["program"])
        assert archived == integer(r["n_archive"])
        stats["native_matched_winners"] += 1


def make_predict(domains, stack, top, evaluate):
    @lru_cache(None)
    def predict(encoded, domain, method):
        sample, mix = domains[domain]
        x = sample.preds[:, :, 1]
        if encoded:
            prog = json.loads(encoded)
            if "direction" not in prog:
                return evaluate(prog, x)
            a, b = x[:, prog["parent"]], x[:, prog["child"]]
            da, db = (a - prog["threshold"]) ** 2, (b - prog["threshold"]) ** 2
            return np.where(
                da <= db if prog["direction"] == "close" else da >= db, a, b
            )
        if method == "known_q":
            return sample.probability(mix)
        if method == "average":
            return x.mean(1)
        return (stack if method == "stack_logit" else top).predict(sample.preds)[:, 1]

    return predict


def verify(directory, native):
    spec = spec_from_file_location(
        "native_switch_verify",
        Path(__file__).resolve().parents[1] / "native-switch" / "verify.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = module.evaluate
    with gzip.open(directory / "runs.csv.gz", "rt") as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == 5400
    by_data = defaultdict(list)
    for row in runs:
        by_data[int(row["data_seed"]), row["fit_regime"]].append(row)
    assert len(by_data) == 24
    if native:
        import okapi

        folder = Path(okapi.__file__).resolve().parent
        rev = subprocess.check_output(
            ["git", "-C", folder, "rev-parse", "HEAD"], text=True
        ).strip()
        assert rev == PIN
        assert not subprocess.check_output(
            ["git", "-C", folder, "status", "--short", "--untracked-files=no"]
        )
    logger.disable("okapi")
    stats = Counter()
    largest = 0.0
    for (data_seed, regime), group in by_data.items():
        fit, val, test = [
            draw_native_switch(data_seed, i, n) for i, n in enumerate((512, 256, 8192))
        ]
        mix = int(regime == "linear")
        domains = {
            "fit": (fit, mix),
            "validation": (val, mix),
            "switch_test": (test, 0),
            "weak_test": (test, 0.5),
            "linear_test": (test, 1),
        }
        fit_x, y = fit.preds[:, :, 1], fit.labels(mix)
        top = TopN("brier").fit(fit.preds, y, task="binary", n_classes=2)
        stack = StackingLogit().fit(fit.preds, y, task="binary", n_classes=2)
        gate_rows = [r for r in group if r["method"] == "gate_free_threshold"]
        assert len(gate_rows) == 5 and len({r["program"] for r in gate_rows}) == 1
        gate = json.loads(gate_rows[0]["program"])
        loss, ties, winner = module.literal_optimum(fit_x, y, False)
        assert (
            gate["parent"],
            gate["child"],
            gate["direction"],
            gate["threshold"],
        ) == winner
        assert (gate["loss_numerator"], gate["minimizers_on_grid"]) == (loss, ties)
        stats["exhaustive_gate_optima"] += 1

        predict = make_predict(domains, stack, top, evaluate)
        by_search = defaultdict(dict)
        for r in group:
            p = predict(r["program"], r["split"], r["method"])
            sample, m = domains[r["split"]]
            losses = [brier(p, sample.labels(m)), brier(p, sample.probability(m))]
            actual = [float(r["observed_brier"]), float(r["conditional_excess"])]
            error = np.max(np.abs(np.subtract(actual, losses)))
            assert error < 1e-13, (data_seed, regime, r["split"], r["method"], error)
            largest = max(largest, float(error))
            stats["losses"] += 2
            if r["program"]:
                stats["program_losses"] += 2
            if r["method"] == "top_n":
                assert json.loads(r["ranking"]) == top.ranking_.tolist()
                assert json.loads(r["prefix_scores"]) == top.prefix_scores_.tolist()
                assert integer(r["selected_n"]) == top.n_selected_
            if r["method"] in METHODS:
                by_search[integer(r["gp_seed"])][
                    r["method"], r["arm"], r["decision"], r["split"]
                ] = r
        assert set(by_search) == {271, 277}
        with training_files(fit_x, y) as (paths, gt):
            for seed, rows in by_search.items():
                assert len(rows) == 100
                initial = rows["gp_initial", "none", "fit", "validation"]
                base = initial["program"]
                gate_present = any(
                    "threshold" in node for node in walk(json.loads(base))
                )
                base_val_loss = brier(
                    predict(base, "validation", "gp_initial"), val.labels(mix)
                )
                verify_edits(rows, evaluate, stats, fit_x, y)
                for method in METHODS:
                    for arm in ("none",) if method == "gp_initial" else ARMS:
                        candidate = rows[method, arm, "fit", "validation"]["program"]
                        cand_loss = brier(
                            predict(candidate, "validation", method), val.labels(mix)
                        )
                        accept = cand_loss < base_val_loss
                        for split in domains:
                            for decision, want in (
                                ("fit", candidate),
                                ("validation_guard", candidate if accept else base),
                            ):
                                r = rows[method, arm, decision, split]
                                assert r["program"] == want
                                assert (r["validation_accepts"] == "True") == accept
                                assert (r["changed"] == "True") == (want != base)
                                assert (
                                    r["initial_has_threshold"] == "True"
                                ) == gate_present
                                assert integer(r["nodes"]) == len(
                                    list(walk(json.loads(want)))
                                )
                        stats["guard_choices"] += 1
                if native:
                    native_winners(
                        paths,
                        gt,
                        val.preds[:, :, 1],
                        val.labels(mix),
                        seed,
                        rows,
                        stats,
                    )
        predict.cache_clear()
    return {"max_abs_brier_error": largest, **dict(sorted(stats.items()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--native",
        action="store_true",
        help="Replay the pinned six-operator GP at 1200 evaluations.",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.native), indent=2))
