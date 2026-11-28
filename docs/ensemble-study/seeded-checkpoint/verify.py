"""Independently regrow one bank and verify the archived selection experiment."""

import hashlib
import json
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 285100
SEARCH = SEED + 1_000_000
BANKS = range(SEED, SEED + 32)
LAWS = ("equal", "signal")
SPLITS = ("A", "B")
SEARCHES = ("stock", "sham", "seeded", "carry")
METHODS = (*SEARCHES, "protected", "mean", "top_n", "prob_lr", "regularized")
KEY = ["seed", "law", "checkpoint_block", "fit_block", "method"]
TOL = 1e-12


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def program_hash(item):
    packed = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def metrics(p, y):
    p = np.asarray(p).ravel()
    pred = p > 0.5
    denominator = np.count_nonzero(pred) + np.count_nonzero(y)
    true_pos = np.count_nonzero(pred & (y == 1))
    # Score the two class components explicitly, including the strict F1 tie.
    probs = np.column_stack([1 - p, p])
    loss = np.mean(np.sum((probs - np.eye(2)[y]) ** 2, axis=1))
    return float(loss), 2 * true_pos / denominator if denominator else 0.0


def risks(p, q):
    conditional = 2 * np.mean(q * (1 - p) ** 2 + (1 - q) * p**2)
    floor = np.mean(2 * q * (1 - q))
    excess = 2 * np.mean((p - q) ** 2)
    np.testing.assert_allclose(conditional - floor, excess, rtol=0, atol=TOL)
    return float(excess), float(conditional)


def compare(record, expected, errors):
    for key, want in expected.items():
        have = record[key]
        if want is None:
            assert pd.isna(have), (key, have)
        elif isinstance(want, (bool, np.bool_)):
            assert have in (True, False, "True", "False"), (key, have)
            assert (str(have) == "True") == want, (key, have, want)
        elif isinstance(want, (float, np.floating)):
            error = abs(have - want)
            assert error <= TOL, (key, have, want, error)
            errors[key] = max(errors.get(key, 0.0), float(error))
        else:
            assert have == want, (key, have, want)


def check_grid(data, keys, dimensions):
    assert data.index.names == keys
    assert data.index.is_unique
    assert set(data.index) == set(product(*dimensions))


def historical_hashes():
    historical = SEED - 1000
    path = ROOT.parent / "checkpoint-reuse/results/run.json"
    (entry,) = [
        r for r in json.loads(path.read_text())["banks"] if r["seed"] == historical
    ]
    bank = generate_checkpoint_bank(historical, law="equal")
    for name, block in bank.items():
        old = entry["blocks"][name]
        for values, key in [
            (block.P, "probabilities"),
            (block.y, "labels"),
            (block.scene, "scenes"),
        ]:
            assert array_hash(values) == old[key]
    return {"seed": historical, "matched_array_hashes": 9}


def paired_laws():
    banks = {law: generate_checkpoint_bank(SEED, law=law) for law in LAWS}
    signal = int(np.random.default_rng([SEED, 0, 1]).integers(8))
    for block_id, (name, size) in enumerate([("A", 24), ("B", 24), ("I", 256)], 1):
        scene_factor = np.random.default_rng([SEED, block_id, 0]).normal(size=(size, 1))
        row_factor = np.random.default_rng([SEED, block_id, 1]).normal(size=(size, 8))
        z = 0.9 * scene_factor + 0.8 * row_factor - 1.1
        q = 1 / (1 + np.exp(-1.8 * z))
        labels = np.random.default_rng([SEED, block_id, 2]).random(q.shape) < q
        old, planted = (banks[law][name] for law in LAWS)
        for block in (old, planted):
            np.testing.assert_array_equal(block.posterior, q.ravel())
            np.testing.assert_array_equal(block.y, labels.ravel())
        np.testing.assert_array_equal(old.scene, planted.scene)
        other = np.arange(8) != signal
        np.testing.assert_array_equal(old.P[:, other], planted.P[:, other])
        np.testing.assert_array_equal(
            planted.P[:, signal], np.repeat(q.reshape(-1, 1), 8, axis=1)
        )
    return banks, signal


def slots_and_inputs(bank, checkpoint):
    block = bank[checkpoint]
    slots = [
        int(np.argmax([metrics(block.P[:, j, k], block.y)[1] for k in range(8)]))
        for j in range(8)
    ]
    inputs = {}
    for name, block in bank.items():
        pos = np.column_stack([block.P[:, j, slot] for j, slot in enumerate(slots)])
        inputs[name] = np.stack((1 - pos, pos), axis=2)
    return slots, inputs


def metrics_row(predictions, fit_y, audit_y, audit_q, x, ax, signal):
    fit, audit = predictions
    fb, ff = metrics(fit, fit_y)
    ab, af = metrics(audit, audit_y)
    excess, _ = risks(audit, audit_q)

    def matches(pair):
        return all(
            np.allclose(a, b, rtol=0, atol=TOL)
            for a, b in zip(predictions, pair, strict=True)
        )

    return {
        "fit_brier": fb,
        "fit_f1": ff,
        "audit_brier": ab,
        "audit_f1": af,
        "audit_excess": excess,
        "selected_is_mean": matches([p[:, :, 1].mean(axis=1) for p in (x, ax)]),
        "selected_is_signal": None
        if signal is None
        else matches([p[:, signal, 1] for p in (x, ax)]),
    }


def reference_predictions(x, y, ax):
    order = sorted(range(8), key=lambda i: metrics(x[:, i, 1], y)[0])
    n = min(range(1, 9), key=lambda k: metrics(x[:, order[:k], 1].mean(axis=1), y)[0])
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
    )
    lr.fit(x.reshape(len(y), -1), y)
    model = RegularizedNativePatternOKAPI(seed=SEARCH, split_seed=SEARCH).fit(
        x, y, task="binary", n_classes=2
    )
    return {
        "mean": ([a[:, :, 1].mean(axis=1) for a in (x, ax)], {"n_providers": 8}),
        "top_n": (
            [a[:, order[:n], 1].mean(axis=1) for a in (x, ax)],
            {"n_providers": n, "n_selected": n},
        ),
        "prob_lr": (
            [lr.predict_proba(a.reshape(len(a), -1))[:, 1] for a in (x, ax)],
            {"n_providers": 8},
        ),
        "regularized": (
            [model.predict(a)[:, 1] for a in (x, ax)],
            {
                "family": model.chosen_,
                "n_providers": len(model.trees_[model.chosen_].unique_value_node_ids),
            },
        ),
    }


def verify_bootstrap(scores):
    comparisons = pd.read_csv(OUT / "comparisons.csv")
    rng = np.random.default_rng(97822)
    samples = rng.integers(0, len(BANKS), size=(10_000, len(BANKS)))
    results = {}
    for law in LAWS:
        contrasts = []
        for bank in BANKS:
            differences = []
            for block in SPLITS:
                key = bank, law, block, block
                differences.append(
                    scores.loc[(*key, "seeded"), "audit_excess"]
                    - scores.loc[(*key, "sham"), "audit_excess"]
                )
            contrasts.append(sum(differences) / 2)
        d = np.asarray(contrasts)
        estimate = [np.mean(d), *np.percentile(d[samples].mean(axis=1), [2.5, 97.5])]
        row = comparisons.query(
            "law == @law and allocation == 'reuse' and method == 'seeded' and reference == 'sham' and metric == 'audit_excess'"
        )
        assert len(row) == 1
        np.testing.assert_allclose(
            row[["estimate", "ci_low", "ci_high"]].to_numpy()[0],
            estimate,
            atol=TOL,
            rtol=0,
        )
        results[law] = {
            "estimate": float(estimate[0]),
            "ci_low": float(estimate[1]),
            "ci_high": float(estimate[2]),
            "n_banks": len(contrasts),
        }
    return results


def replay_founder(gp, provider_ids):
    from okapi.node import MeanNode, ValueNode
    from okapi.tree import Tree

    leaves = [ValueNode(None, gp.train_tensors[key], key) for key in provider_ids]
    leaves[0].add_child(MeanNode(leaves[1:]))
    return Tree(leaves[0])


def regrow(x, y, arm, sham):
    """Record native fit calls and survivors in a fresh fit-only process state."""
    trees, by_key, visits, stages = [], {}, [], []
    offered = []

    def tree_ids(batch):
        indices = []
        for tree in batch:
            key = tree.structural_key()
            if key not in by_key:
                by_key[key] = len(trees)
                trees.append(tree)
            indices.append(by_key[key])
        return indices

    class Observer(Okapi):
        def _calculate_fitnesses(self, trees=None):
            batch = self.population if trees is None else trees
            values = super()._calculate_fitnesses(batch)
            visits.extend(tree_ids(batch))
            return values

        def _mutate_additional_population(self):
            mutations = super()._mutate_additional_population()
            offered[:] = tree_ids(self.additional_population)
            return mutations

    def objective(preds, labels):
        return 1 - np.mean((np.ravel(preds) - np.ravel(labels)) ** 2)

    singleton_ids = [f"m{i}.npy" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmp, _task_postprocessing("binary"):
        root = Path(tmp)
        directory = root / "scores"
        directory.mkdir()
        for j, key in enumerate(singleton_ids):
            np.save(directory / key, x[:, j, 1:2])
        np.save(root / "y.npy", y)
        gp = Observer(
            directory,
            root / "y.npy",
            8,
            2,
            2,
            backend="numpy",
            seed=SEARCH,
            objective_functions=(objective,),
            minimize_node_count=True,
            max_nodes=31,
            max_depth=8,
            operator_point_mutation=False,
            reuse_survivor_fitness=False,
            selection_fitness_decimals=3,
        )
        assert len(gp.population) == 8 and {t.root.id for t in gp.population} == set(
            singleton_ids
        )
        assert all(t.nodes_count == 1 for t in gp.population)
        if arm != "stock":
            leaves = [singleton_ids[sham]] * 8 if arm == "sham" else singleton_ids
            founder = replay_founder(gp, leaves)
            assert (
                founder.nodes_count == 9
                and [node.id for node in founder.value_nodes] == leaves
            )
            gp.population.append(founder)
        gp.fitnesses = gp._selection_fitnesses(gp.population)
        reserve = gp.fitnesses[-1].copy()
        for gen in range(17):
            inserted = False
            if gen:
                gp.run_iteration()
                if arm == "carry" and founder.structural_key() not in {
                    t.structural_key() for t in gp.population
                }:
                    gp.population[-1] = founder.copy()
                    gp.population[-1].mutation_chance = founder.mutation_chance
                    gp.fitnesses[-1] = reserve
                    inserted = True
            stages.append(
                {
                    "active": tree_ids(gp.population),
                    "offered": list(offered),
                    "calls": len(visits),
                    "inserted": inserted,
                }
            )
    assert all(trees[i].structural_key() == key for key, i in by_key.items())
    return trees, visits, stages


def evaluate_gp(x, y, ax, ay, q, arm, sham, signal):
    trees, visits, stages = regrow(x, y, arm, sham)
    fit = [predict_tree(t, x)[:, 1] for t in trees]
    # Production deployment compares these unrounded positive-channel losses.
    native_losses = [2 * np.square(pred - y).mean() for pred in fit]
    winners = [
        min(s["active"], key=lambda i: (native_losses[i], trees[i].nodes_count))
        for s in stages
    ]
    audit = [predict_tree(t, ax)[:, 1] for t in trees]
    audit_summaries = [metrics(p, ay) for p in audit]
    excess = [risks(p, q)[0] for p in audit]
    init = stages[0]["active"]
    initial_order = [trees[i].root.id for i in init[:8]]
    programs = [serialize(t.root) for t in trees]
    init_digest = program_hash([programs[i] for i in init])
    plain_steps = [{k: s[k] for k in ("active", "offered", "calls")} for s in stages]
    trajectory_digest = program_hash([programs, visits, plain_steps])
    mean_ref = [a[:, :, 1].mean(axis=1) for a in (x, ax)]
    equivalent = np.array(
        [
            all(
                np.allclose(p, m, rtol=0, atol=TOL)
                for p, m in zip((f, a), mean_ref, strict=True)
            )
            for f, a in zip(fit, audit, strict=True)
        ]
    )

    def leaf(i):
        return {"id": f"m{i}.npy", "type": "ValueNode", "children": []}

    mean_program = {
        **leaf(0),
        "children": [{"type": "MeanNode", "children": [leaf(i) for i in range(1, 8)]}],
    }
    mean_exact = np.array([p == mean_program for p in programs])
    if signal is not None:
        signal_refs = [a[:, signal, 1] for a in (x, ax)]
        signal_equiv = np.array(
            [
                all(
                    np.allclose(p, m, rtol=0, atol=TOL)
                    for p, m in zip((f, a), signal_refs, strict=True)
                )
                for f, a in zip(fit, audit, strict=True)
            ]
        )
        signal_exact = np.array([p == leaf(signal) for p in programs])
    seen, records = set(), []
    count_prev = 0
    founder_error = None
    if arm != "stock":
        raw = [
            d[:, sham, 1] if arm == "sham" else d[:, :, 1].mean(axis=1) for d in (x, ax)
        ]
        founder_error = max(
            float(np.abs(obs - exp).max())
            for obs, exp in zip((fit[init[-1]], audit[init[-1]]), raw, strict=True)
        )
        assert founder_error <= TOL
    for g, (s, winner) in enumerate(zip(stages, winners, strict=True)):
        active = s["active"]
        seen.update(active)
        seen.update(s["offered"])
        assert seen == set(visits[: s["calls"]])
        if arm == "carry":
            assert init[-1] in active
        records.append(
            {
                "generation": g,
                "n_survivors": len(active),
                "n_seen": len(seen),
                "proposals": len(s["offered"]),
                "score_calls": s["calls"] - count_prev,
                "cumulative_score_calls": s["calls"],
                "selected_id": winner,
                "selected_nodes": trees[winner].nodes_count,
                "fit_selected": native_losses[winner],
                "audit_selected": audit_summaries[winner][0],
                "fit_f1": metrics(fit[winner], y)[1],
                "audit_f1": audit_summaries[winner][1],
                "audit_selected_excess": excess[winner],
                "selected_provider_count": len(
                    {leaf.id for leaf in trees[winner].value_nodes}
                ),
                "audit_best_seen_excess": min(excess[i] for i in seen),
                "audit_best_final_excess": min(excess[i] for i in active),
            }
        )
        flags = {"mean_exact": mean_exact, "mean_equiv": equivalent}
        flags.update(
            {"signal_exact": signal_exact, "signal_equiv": signal_equiv}
            if signal is not None
            else {}
        )
        for flag, mask in flags.items():
            for group, ids in (
                ("active", active),
                ("offered", s["offered"]),
                ("seen", sorted(seen)),
            ):
                records[-1]["n_" + group + "_" + flag] = int(mask[ids].sum())
            records[-1]["selected_" + flag] = bool(mask[winner])
        for field, ids in (("final", active), ("seen", seen)):
            records[-1]["audit_best_" + field] = min(audit_summaries[i][0] for i in ids)
            records[-1]["audit_best_" + field + "_f1"] = max(
                audit_summaries[i][1] for i in ids
            )
        records[-1]["carry_reinserted"] = bool(s["inserted"])
        count_prev = s["calls"]
    winner = winners[-1]

    def final(i):
        return {
            **metrics_row([fit[i], audit[i]], y, ay, q, x, ax, signal),
            "n_providers": len({leaf.id for leaf in trees[i].value_nodes}),
            "selected_program_sha256": program_hash(programs[i]),
            "trajectory_sha256": trajectory_digest,
            "initial_population_sha256": init_digest,
            "carry_reinsertions": sum(s["inserted"] for s in stages),
            "n_seen": len(seen),
            "n_survivors": len(stages[-1]["active"]),
            "fit_prediction_sha256": array_hash(fit[i]),
            "audit_prediction_sha256": array_hash(audit[i]),
            "score_calls": len(visits),
            "proposals": sum(len(s["offered"]) for s in stages),
            "audit_best_final_excess": records[-1]["audit_best_final_excess"],
            "audit_best_seen_excess": records[-1]["audit_best_seen_excess"],
        }

    raw_row = final(winner)
    result = {arm: raw_row}
    if arm == "seeded":
        choose_mean = native_losses[init[-1]] <= native_losses[winner]
        result["protected"] = {
            **final(init[-1] if choose_mean else winner),
            "used_mean_finalist": choose_mean,
        }
    metadata = {
        "singleton_order": initial_order,
        "initial_population_sha256": init_digest,
        "founder_count": len(init),
        "ninth_founder_sha256": None
        if arm == "stock"
        else program_hash(programs[init[-1]]),
        "ninth_founder_nodes": None if arm == "stock" else trees[init[-1]].nodes_count,
        "max_error_ninth_vs_numeric": founder_error,
        "carry_reinsertions": sum(s["inserted"] for s in stages),
        "carry_mean_in_all_generations": True if arm == "carry" else None,
    }
    return result, records, metadata


def verify_intervention(x, y, ax, ay, q, signal, sham, expected):
    from native_seed import run_seeded

    # Invert I labels and substitute a constant for the entire I posterior.
    # Every deployed vector/digest must still agree with the untouched refit.
    replacement = np.full_like(q, 0.5)
    poisoned, details = run_seeded(
        x,
        y,
        ax,
        1 - ay,
        seed=SEARCH,
        audit_q=replacement,
        arm="seeded",
        sham_provider=sham,
        signal_provider=signal,
    )
    output = {}
    for method, prefix in (("seeded", ""), ("protected", "protected_")):
        reference = expected[method]
        fit_hash = array_hash(details[prefix + "fit_selected"])
        audit_hash = array_hash(details[prefix + "audit_selected"])
        assert poisoned["trajectory_sha256"] == reference["trajectory_sha256"]
        assert (
            poisoned[prefix + "selected_program_sha256"]
            == reference["selected_program_sha256"]
        )
        assert (
            reference["fit_prediction_sha256"] == fit_hash
            and reference["audit_prediction_sha256"] == audit_hash
        )
        pred = details[prefix + "audit_selected"]
        invert_brier = metrics(pred, 1 - ay)[0]
        replaced_excess = risks(pred, replacement)[0]
        assert abs(poisoned[prefix + "audit_selected"] - invert_brier) <= TOL
        assert abs(poisoned[prefix + "audit_selected_excess"] - replaced_excess) <= TOL
        assert abs(invert_brier - reference["audit_brier"]) > TOL
        assert abs(replaced_excess - reference["audit_excess"]) > TOL
        output[method] = {
            "program_sha256": reference["selected_program_sha256"],
            "fit_prediction_sha256": fit_hash,
            "audit_prediction_sha256": audit_hash,
            "fit_only_trajectory_sha256": reference["trajectory_sha256"],
            "observed_brier": reference["audit_brier"],
            "inverted_brier": invert_brier,
            "original_excess": reference["audit_excess"],
            "replacement_excess": replaced_excess,
        }
    return output


def standard_panel():
    from fusionbench.transfer import load_preset

    suites = list(load_preset("standard-v3")["scenarios"])
    panel_seeds = (285201, 285202)
    ids = ["scenario_id", "data_seed", "method"]
    scores = pd.read_csv(OUT / "panel-scores.csv").set_index(ids)
    stages = pd.read_csv(OUT / "panel-trajectories.csv").set_index(
        [*ids[:2], "arm", "generation"]
    )
    check_grid(scores, ids, (suites, panel_seeds, METHODS))
    check_grid(
        stages,
        [*ids[:2], "arm", "generation"],
        (suites, panel_seeds, SEARCHES, range(17)),
    )
    assert np.isfinite(scores[["fit_brier", "test_brier"]]).all().all()
    assert stages.xs("carry", level="arm").genotype_mean_active.all()
    assert set(scores[scores.max_nodes > 31].index.get_level_values("scenario_id")) == {
        "needle"
    }
    rows = pd.read_csv(OUT / "panel-contrasts.csv")
    estimates = []
    for families in (suites, [s for s in suites if s != "needle"]):
        differences = np.array(
            [
                [
                    scores.loc[(s, seed, "seeded"), "test_brier"]
                    - scores.loc[(s, seed, "sham"), "test_brier"]
                    for seed in panel_seeds
                ]
                for s in families
            ]
        )
        # The two within-scenario draws give each scenario equal weight.
        choices = np.random.default_rng(97822).integers(
            2, size=(10_000, len(families), 2)
        )
        resampled = np.array(
            [
                differences[np.arange(len(families))[:, None], sample].mean()
                for sample in choices
            ]
        )
        values = [differences.mean(), *np.quantile(resampled, [0.025, 0.975])]
        row = rows[
            (rows.families == len(families))
            & (rows.method == "seeded")
            & (rows.comparator == "sham")
        ]
        assert len(row) == 1
        np.testing.assert_allclose(
            row[["mean_delta", "ci_lower", "ci_upper"]].to_numpy()[0],
            values,
            atol=TOL,
            rtol=0,
        )
        estimates.append(
            {
                "scenarios": len(families),
                "estimate": values[0],
                "ci_low": values[1],
                "ci_high": values[2],
            }
        )
    return {
        "complete_scores": len(scores),
        "complete_trajectory": len(stages),
        "seeded_minus_sham": estimates,
    }


def verify():
    scores = pd.read_csv(OUT / "scores.csv").set_index(KEY)
    stages = pd.read_csv(OUT / "trajectories.csv.gz").set_index([*KEY, "generation"])
    choice_keys = ["seed", "law", "checkpoint_block", "provider"]
    choices = pd.read_csv(OUT / "checkpoints.csv").set_index(choice_keys)
    check_grid(scores, KEY, (BANKS, LAWS, SPLITS, SPLITS, METHODS))
    check_grid(
        stages, [*KEY, "generation"], (BANKS, LAWS, SPLITS, SPLITS, SEARCHES, range(17))
    )
    check_grid(choices, choice_keys, (BANKS, LAWS, SPLITS, range(8)))
    assert (stages.xs("carry", level="method").n_active_mean_exact > 0).all()
    raw_terminal = stages.xs(16, level="generation")
    for field, target in [
        ("fit_selected", "fit_brier"),
        ("audit_selected", "audit_brier"),
        ("audit_f1", "audit_f1"),
        ("audit_selected_excess", "audit_excess"),
        ("audit_best_seen_excess", "audit_best_seen_excess"),
    ]:
        np.testing.assert_allclose(
            raw_terminal[field],
            scores.reindex(raw_terminal.index)[target],
            rtol=0,
            atol=TOL,
        )
    chosen_signal = scores[
        (scores.index.get_level_values("law") == "signal")
        & (scores.selected_is_signal == True)
    ]
    assert len(chosen_signal) and (chosen_signal.audit_excess < TOL**2 * 2).all()
    history = historical_hashes()
    banks, signal = paired_laws()
    manifest = json.loads((OUT / "run.json").read_text())
    (archived,) = (r for r in manifest["banks"] if r["seed"] == SEED)
    sham = int(np.random.default_rng([SEED, 0, 2]).integers(8))
    errors, recalc = {}, {}
    for law, bank in banks.items():
        specialist = signal if law == "signal" else None
        for name, block in bank.items():
            for field in ("P", "scene", "posterior", "y"):
                assert (
                    array_hash(getattr(block, field))
                    == archived["blocks"][law][name][field]
                )
        for pick in SPLITS:
            slots, _ = slots_and_inputs(bank, pick)
            for i, slot in enumerate(slots):
                compare(
                    choices.loc[(SEED, law, pick, i)],
                    {
                        "slot": slot,
                        "sham_provider": sham,
                        "signal_provider": specialist,
                    },
                    errors,
                )
        slots, prepared = slots_and_inputs(bank, "A")
        if specialist is not None:
            assert slots[specialist] == 0
        x, ax, y, ay, q = (
            prepared["A"],
            prepared["I"],
            bank["A"].y,
            bank["I"].y,
            bank["I"].posterior,
        )
        key = SEED, law, "A", "A"
        details = {}
        for method, (predictions, extra) in reference_predictions(x, y, ax).items():
            row = {**metrics_row(predictions, y, ay, q, x, ax, specialist), **extra}
            compare(scores.loc[(*key, method)], row, errors)
        seeded = None
        for arm in SEARCHES:
            deployments, trace, information = evaluate_gp(
                x, y, ax, ay, q, arm, sham, specialist
            )
            for method, expected in deployments.items():
                compare(scores.loc[(*key, method)], expected, errors)
            for g, stage in enumerate(trace):
                compare(
                    stages.loc[(*key, arm, g)],
                    {k: v for k, v in stage.items() if k != "generation"},
                    errors,
                )
            details[arm] = information
            if arm == "seeded":
                seeded = deployments
        assert all(
            info["singleton_order"] == details["stock"]["singleton_order"]
            for info in details.values()
        )
        assert (
            details["seeded"]["ninth_founder_sha256"]
            == details["carry"]["ninth_founder_sha256"]
        )
        assert (
            details["seeded"]["ninth_founder_sha256"]
            != details["sham"]["ninth_founder_sha256"]
        )
        floor = float((2 * q * (1 - q)).mean())
        mean_p = ax[:, :, 1].mean(axis=1)
        mean_excess, mean_risk = risks(mean_p, q)
        recalc[law] = {
            "slots_on_A": slots,
            "founders": details,
            "conditional_oracle_brier": floor,
            "conditional_mean_excess": mean_excess,
            "conditional_mean_brier": mean_risk,
            "audit_input_replacement": verify_intervention(
                x, y, ax, ay, q, specialist, sham, seeded
            ),
        }
    return {
        "seed": SEED,
        "search_seed": SEARCH,
        "checkpoint_count": 8,
        "checkpoint_block": "A",
        "fit_block": "A",
        "audit_block": "I",
        "signal_provider": signal,
        "sham_provider": sham,
        "historical_equal_law_replay": history,
        "laws": recalc,
        "checkpoint_grid": {
            "scores": len(scores),
            "stages": len(stages),
            "checkpoints": len(choices),
        },
        "independent_replay": {
            "deployed_scores": 2 * len(METHODS),
            "generation_rows": 2 * 17 * len(SEARCHES),
            "selected_slots": 2 * len(SPLITS) * 8,
            "max_abs_score_error": max(errors.values()),
            "max_abs_score_error_by_field": errors,
        },
        "seeded_minus_sham_reuse_excess_brier_bootstrap": verify_bootstrap(scores),
        "standard_panel": standard_panel(),
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = verify()
    dest = OUT / "verification.json"
    dest.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(f"Verified both laws, the panel and the paired bootstrap; wrote {dest}")
