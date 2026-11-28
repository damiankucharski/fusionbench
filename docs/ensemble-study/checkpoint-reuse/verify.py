"""Verify the paired archive, independently refit bank 284100 and invert its audit labels."""

import hashlib
import json
import tempfile
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from native import discover_and_score
from okapi import Okapi
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fusionbench.eval.native_switch import serialize
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.okapi import _task_postprocessing
from fusionbench.scenarios.checkpoint_bank import generate_checkpoint_bank

OUT = Path(__file__).parent / "results"
SEED = 284100
SEARCH = SEED + 1_000_000
KEY = ["seed", "k", "checkpoint_block", "fit_block", "method"]
METHODS = ("mean", "top_n", "prob_lr", "regularized", "gp_brier", "gp_f1")


def metrics(p, y):
    p = np.asarray(p).reshape(-1)
    y = np.asarray(y).reshape(-1)
    hit = p > 0.5
    positives = np.count_nonzero(y) + np.count_nonzero(hit)
    return (
        float(2 * np.mean((p - y) ** 2)),
        float(2 * (hit & (y == 1)).sum() / positives) if positives else 0.0,
    )


def tensor_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def program_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def checkpoints(block, k):
    # Separate provider-by-provider implementation, using all labelled rows.
    slots = []
    for j in range(8):
        scores = [
            metrics(block.P[:, j, candidate], block.y)[1] for candidate in range(k)
        ]
        slots.append(int(np.argmax(scores)))
    return slots


def apply_checkpoints(block, slots):
    p = np.column_stack([block.P[:, j, s] for j, s in enumerate(slots)])
    return np.stack((1 - p, p), axis=-1)


def four_methods(x, y, ax, ay):
    """Independent implementations of mean/ranked pool/sklearn stacking, direct production fit."""
    brier = [metrics(x[:, i, 1], y)[0] for i in range(8)]
    ranked = np.argsort(brier, kind="stable")
    prefixes = [metrics(x[:, ranked[:j], 1].mean(axis=1), y)[0] for j in range(1, 9)]
    n = int(np.argmin(prefixes)) + 1
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
    )
    lr.fit(x.reshape(len(y), -1), y)
    native = RegularizedNativePatternOKAPI(seed=SEARCH, split_seed=SEARCH).fit(
        x, y, task="binary", n_classes=2
    )
    outputs = {
        "mean": {},
        "top_n": {"n_selected": n},
        "prob_lr": {},
        "regularized": {"family": native.chosen_},
    }
    for name, a, labels in (("fit", x, y), ("audit", ax, ay)):
        for method, prob in (
            ("mean", a[:, :, 1].mean(axis=1)),
            ("top_n", a[:, ranked[:n], 1].mean(axis=1)),
            ("prob_lr", lr.predict_proba(a.reshape(len(labels), -1))[:, 1]),
            ("regularized", native.predict(a)[:, 1]),
        ):
            outputs[method][name + "_selected"], outputs[method][name + "_f1"] = (
                metrics(prob, labels)
            )
    return outputs


def record_native(x, y, name):
    # Independent of the study's _fit, Trace and audit replay. No audit inputs
    # are accepted while native random draws, survival, and fitness calls run.
    trees, index, visits, steps = [], {}, [], []
    offered = []

    def ids(batch):
        result = []
        for tree in batch:
            key = tree.structural_key()
            if key not in index:
                index[key] = len(trees)
                trees.append(tree)
            result.append(index[key])
        return result

    def skill(p, labels):
        loss, f1 = metrics(p, labels)
        return 1 - loss / 2 if name == "brier" else f1

    class Observed(Okapi):
        def _mutate_additional_population(self):
            count = super()._mutate_additional_population()
            offered[:] = ids(self.additional_population)
            return count

        def _calculate_fitnesses(self, batch=None):
            batch = self.population if batch is None else batch
            values = super()._calculate_fitnesses(batch)
            visits.extend(ids(batch))
            return values

    with tempfile.TemporaryDirectory() as tmp, _task_postprocessing("binary"):
        root = Path(tmp)
        inputs = root / "providers"
        inputs.mkdir()
        for i in range(8):
            np.save(inputs / f"m{i}.npy", x[:, i, 1:2])
        np.save(root / "y.npy", y)
        gp = Observed(
            inputs,
            root / "y.npy",
            8,
            2,
            2,
            seed=SEARCH,
            backend="numpy",
            objective_functions=(skill,),
            max_nodes=31,
            max_depth=8,
            minimize_node_count=True,
        )
        gp.fitnesses = gp._selection_fitnesses(gp.population)
        for generation in range(17):
            if generation:
                gp.run_iteration()
            steps.append(
                {
                    "active": ids(gp.population),
                    "offered": list(offered),
                    "calls": len(visits),
                }
            )
    return trees, visits, steps


def verify_native(x, y, ax, ay, objective):
    trees, visits, snapshots = record_native(x, y, objective)
    programs = [serialize(t.root) for t in trees]
    fit = np.array([metrics(predict_tree(t, x)[:, 1], y) for t in trees])
    audit = np.array([metrics(predict_tree(t, ax)[:, 1], ay) for t in trees])
    stages, seen, before = [], set(), 0
    for g, s in enumerate(snapshots):
        active = s["active"]
        seen.update(s["offered"])
        seen.update(active)
        winner = min(
            active,
            key=lambda i: (
                fit[i, 0] if objective == "brier" else -fit[i, 1],
                trees[i].nodes_count,
            ),
        )
        stages.append(
            {
                "generation": g,
                "proposals": len(s["offered"]),
                "score_calls": s["calls"] - before,
                "cumulative_score_calls": s["calls"],
                "n_seen": len(seen),
                "n_survivors": len(active),
                "selected_id": winner,
                "selected_nodes": trees[winner].nodes_count,
                "fit_selected": fit[winner, 0],
                "audit_selected": audit[winner, 0],
                "fit_f1": fit[winner, 1],
                "audit_f1": audit[winner, 1],
                "audit_best_final": audit[active, 0].min(),
                "audit_best_seen": audit[list(seen), 0].min(),
                "audit_best_final_f1": audit[active, 1].max(),
                "audit_best_seen_f1": audit[list(seen), 1].max(),
            }
        )
        before = s["calls"]
    t = trees[winner]
    row = dict(stages[-1])
    row.pop("selected_id")
    row.pop("generation")
    row.pop("cumulative_score_calls")
    row.update(
        {
            "score_calls": len(visits),
            "proposals": sum(len(s["offered"]) for s in snapshots),
            "objective": objective,
            "n_generations": 16,
            "selected_provider_count": len({leaf.id for leaf in t.value_nodes}),
            "fit_prediction_sha256": tensor_hash(predict_tree(t, x)[:, 1]),
            "audit_prediction_sha256": tensor_hash(predict_tree(t, ax)[:, 1]),
            "selected_program_sha256": program_hash(programs[winner]),
            "initial_population_sha256": program_hash(
                [programs[i] for i in snapshots[0]["active"]]
            ),
            "trajectory_sha256": program_hash([programs, visits, snapshots]),
        }
    )
    return row, stages


def check_archive(scores, choices, stages):
    banks = range(SEED, SEED + 32)
    cells = list(product(banks, (1, 8), ("A", "B"), ("A", "B")))
    expected_scores = {(*c, m) for c in cells for m in METHODS}
    expected_stages = {
        (*c, m, g) for c in cells for m in ("gp_brier", "gp_f1") for g in range(17)
    }
    expected_choices = set(product(banks, (1, 8), ("A", "B"), range(8)))
    for df, expected in (
        (scores, expected_scores),
        (stages, expected_stages),
        (choices, expected_choices),
    ):
        assert df.index.is_unique and set(df.index) == expected
    # At K=1 the checkpoint-block axis is redundant for a fixed fit block.
    for frame in (scores, stages):
        k1 = frame.xs(1, level="k")
        pd.testing.assert_frame_equal(
            k1.xs("A", level="checkpoint_block").sort_index(),
            k1.xs("B", level="checkpoint_block").sort_index(),
            check_exact=True,
        )
    assert (choices.xs(1, level="k").slot == 0).all()
    for metric in ("audit_selected", "audit_f1"):
        for _, group in scores.xs(1, level="k").groupby(level=["seed", "method"]):
            a = group[metric].droplevel(["seed", "method"])
            assert ((a["A", "A"] - a["A", "B"]) + (a["B", "B"] - a["B", "A"])) == 0
        mean = scores.xs("mean", level="method")[metric]
        assert (
            mean.xs("A", level="fit_block").sort_index()
            == mean.xs("B", level="fit_block").sort_index()
        ).all()


def verify_primary_contrast(scores):
    delta = []
    for seed in range(SEED, SEED + 32):

        def audit(cp, fit, seed=seed):
            return scores.loc[(seed, 8, cp, fit, "gp_brier"), "audit_selected"]

        delta.append(
            ((audit("A", "A") - audit("A", "B")) + (audit("B", "B") - audit("B", "A")))
            / 2
        )
    d = np.asarray(delta)
    index = np.random.default_rng(97821).integers(0, len(d), size=(10_000, len(d)))
    estimate = [d.mean(), *np.percentile(d[index].mean(axis=1), [2.5, 97.5])]
    archived = pd.read_csv(OUT / "summary.csv").query(
        "method == 'gp_brier' and k == 8 and metric == 'audit_brier' and comparison == 'reuse_minus_disjoint'"
    )
    np.testing.assert_allclose(
        estimate,
        archived[["estimate", "ci_low", "ci_high"]].iloc[0],
        rtol=0,
        atol=1e-12,
    )
    return dict(zip(("estimate", "ci_low", "ci_high"), estimate, strict=True))


def run():
    started = perf_counter()
    scores = pd.read_csv(OUT / "scores.csv", float_precision="round_trip").set_index(
        KEY
    )
    choices = pd.read_csv(
        OUT / "checkpoints.csv", float_precision="round_trip"
    ).set_index(KEY[:3] + ["provider"])
    stages = pd.read_csv(
        OUT / "trajectories.csv.gz", float_precision="round_trip"
    ).set_index(KEY + ["generation"])
    check_archive(scores, choices, stages)
    blocks = generate_checkpoint_bank(SEED)
    manifest = next(
        b["blocks"]
        for b in json.loads((OUT / "run.json").read_text())["banks"]
        if b["seed"] == SEED
    )
    for label, b in blocks.items():
        assert all(
            manifest[label][k] == tensor_hash(v)
            for k, v in (("probabilities", b.P), ("labels", b.y), ("scenes", b.scene))
        )
    assert set(blocks["A"].scene).isdisjoint(blocks["B"].scene)
    assert set(np.r_[blocks["A"].scene, blocks["B"].scene]).isdisjoint(
        blocks["I"].scene
    )
    comparisons = {"numeric": 0, "text": 0}
    max_diff = 0.0

    def check(produced, stored):
        nonlocal max_diff
        for name, value in produced.items():
            if isinstance(value, str):
                assert value == stored[name], (name, value, stored[name])
                comparisons["text"] += 1
            else:
                diff = abs(float(value) - float(stored[name]))
                assert diff <= 1e-12, (name, value, stored[name])
                max_diff = max(max_diff, diff)
                comparisons["numeric"] += 1

    cache, inverted = {}, {}
    for k, selector in product((1, 8), ("A", "B")):
        slots = checkpoints(blocks[selector], k)
        alt = "B" if selector == "A" else "A"
        for j, slot in enumerate(slots):
            vals = {"slot": slot}
            for field, bundle in (
                ("selection_f1", selector),
                ("opposite_f1", alt),
                ("audit_f1", "I"),
            ):
                b = blocks[bundle]
                vals[field] = metrics(b.P[:, j, slot], b.y)[1]
            check(vals, choices.loc[SEED, k, selector, j])
        ax, ay = apply_checkpoints(blocks["I"], slots), blocks["I"].y
        for fit_name in ("A", "B"):
            b = blocks[fit_name]
            x, y = apply_checkpoints(b, slots), b.y
            cell = (SEED, k, selector, fit_name)
            key = (tuple(slots), fit_name)
            if key not in cache:
                controls = four_methods(x, y, ax, ay)
                gps = {
                    objective: verify_native(x, y, ax, ay, objective)
                    for objective in ("brier", "f1")
                }
                cache[key] = controls, gps
            controls, gps = cache[key]
            for m, row in controls.items():
                check(row, scores.loc[(*cell, m)])
            for objective, (row, details) in gps.items():
                method = "gp_" + objective
                check(row, scores.loc[(*cell, method)])
                for g, detail in enumerate(details):
                    check(
                        {k: v for k, v in detail.items() if k != "generation"},
                        stages.loc[(*cell, method, g)],
                    )
                if k == 8 and selector == fit_name == "A":
                    # Exercise the actual production-of-results entry point: its
                    # audit labels are deliberately inverted after fixing inputs.
                    flipped, replay = discover_and_score(
                        x, y, ax, 1 - ay, seed=SEARCH, objective=objective
                    )
                    keys = [
                        "selected_program_sha256",
                        "trajectory_sha256",
                        "initial_population_sha256",
                    ]
                    assert all(flipped[f] == row[f] for f in keys)
                    assert (
                        tensor_hash(replay["audit_selected"])
                        == row["audit_prediction_sha256"]
                    )
                    assert (
                        tensor_hash(replay["fit_selected"])
                        == row["fit_prediction_sha256"]
                    )
                    assert flipped["audit_selected"] != row["audit_selected"]
                    inverted[objective] = {
                        "checkpoint_slots": slots,
                        **{f: row[f] for f in keys},
                        "fit_predictions": row["fit_prediction_sha256"],
                        "audit_predictions": row["audit_prediction_sha256"],
                        "audit_brier_original": row["audit_selected"],
                        "audit_brier_inverted": flipped["audit_selected"],
                    }
    return {
        "verified_seed": SEED,
        "search_seed": SEARCH,
        "numeric_threads": 1,
        "worker_processes": 1,
        "archive_rows": {
            "scores": len(scores),
            "checkpoints": len(choices),
            "trajectory": len(stages),
        },
        "recomputed_rows": {"scores": 48, "checkpoints": 32, "trajectory": 272},
        "unique_fit_inputs": len(cache),
        "regenerated_block_hashes": 9,
        "comparisons": comparisons,
        "primary_brier_paired_bootstrap": verify_primary_contrast(scores),
        "max_abs_numeric_error": max_diff,
        "k1_exact_swap_cancellation": True,
        "mean_exact_audit_swap_cancellation": True,
        "inverted_audit": inverted,
        "wall_seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        report = run()
    (OUT / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))
