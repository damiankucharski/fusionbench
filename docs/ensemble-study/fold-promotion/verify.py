"""Independent numerical, pairing and decision checks for the two frozen rounds."""

import hashlib
import json
from itertools import product
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
)
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "scene-dependence/results"
R2 = HERE / "results"
R1_SEEDS = range(287100, 287132)
R2_SEEDS = range(287200, 287232)
LAWS = ("equal", "signal")
DESIGNS1 = tuple(
    (s, g, r) for s, (g, r) in product((0.0, 0.5, 1.0), ((24, 8), (192, 1)))
)
DESIGNS2 = ((0.0, 24, 8), (1.0, 24, 8), (1.0, 192, 8))
REFS = ("mean", "top_n", "prob_lr", "regularized")
POLICIES = (
    "anchor",
    "pooled_final",
    "pooled_pooled",
    "pooled_fold",
    "fold_pooled",
    "fold_fold",
)
TOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b, key, tol=TOL):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        atol=tol,
        rtol=0,
        err_msg=str(key),
    )


def read(file):
    return pd.read_csv(file, float_precision="round_trip")


def grid(path, keys, expected):
    t = read(path).set_index(keys).sort_index()
    require(
        t.index.is_unique and set(t.index) == set(expected),
        f"Bad coverage: {path.name}",
    )
    return t


def sha(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def slots(block):
    selected = []
    for j in range(8):
        rates = []
        for p in block.P[:, j].T:
            hit = p > 0.5
            d = hit.sum() + block.y.sum()
            rates.append(2 * (hit & (block.y == 1)).sum() / d if d else 0.0)
        selected.append(int(np.argmax(rates)))
    return selected


def inputs(block, indices):
    p = np.column_stack([block.P[:, j, k] for j, k in enumerate(indices)])
    return np.stack((1 - p, p), axis=2)


def rows(block, groups, per_group):
    # Each row is identified by scene and position within that scene.
    ids = np.unique(block.scene)
    return np.flatnonzero(
        (block.scene <= ids[groups - 1]) & (np.arange(len(block.y)) % 8 < per_group)
    )


def row_brier(p, y):
    if p.ndim == 1:
        p = np.column_stack((1 - p, p))
    return np.square(p - np.eye(p.shape[1])[y]).sum(axis=1)


def conditional_excess(p, q):
    return float((2 * (q * (1 - p) ** 2 + (1 - q) * p**2) - 2 * q * (1 - q)).mean())


def control_replay(x, y, ax, ay, q, sub):
    ranked = sorted(range(x.shape[1]), key=lambda j: row_brier(x[:, j], y).mean())
    n = min(
        range(1, len(ranked) + 1),
        key=lambda k: row_brier(x[:, ranked[:k]].mean(axis=1), y).mean(),
    )
    for method, ids in (("mean", list(range(x.shape[1]))), ("top_n", ranked[:n])):
        predict = ax[:, ids].mean(axis=1)
        metrics = {
            "fit_brier": row_brier(x[:, ids].mean(axis=1), y).mean(),
            "audit_brier": row_brier(predict, ay).mean(),
            "audit_excess": conditional_excess(predict[:, 1], q),
        }
        for key, value in metrics.items():
            close(value, sub.loc[method, key], (method, key))


def bootstrap(values, seed):
    x = np.asarray(values, dtype=float)
    choices = np.random.default_rng(seed).integers(len(x), size=(10_000, len(x)))
    return [
        float(x.mean()),
        *map(float, np.percentile(x[choices].mean(axis=1), (2.5, 97.5))),
    ]


def named(values):
    return dict(zip(("estimate", "ci_low", "ci_high"), values, strict=True))


def full_marginal():
    # Row residual = independent Gaussian static noise plus a Gaussian AR(1)
    # path. Its entire 8-slot covariance fixes its joint law at one row.
    lag = np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
    ar = 0.65**lag
    reference = (0.5**2 + 0.5**2) + (0.85**2 + 0.5**2) * ar
    cross = []
    for s in (0, 0.5, 1):
        shared = np.sqrt([0.5 * s, 0.9725 * s])
        private = np.sqrt([0.5 * (1 - s), 0.9725 * (1 - s)])
        shared_kernel = shared[0] ** 2 + shared[1] ** 2 * ar
        close(
            shared_kernel + private[0] ** 2 + private[1] ** 2 * ar,
            reference,
            ("row covariance", s),
        )
        close(shared_kernel, s * reference, ("cross row covariance", s))
        cross.append(float(shared_kernel[0, 0]))
    return {
        "single_row_variance": float(reference[0, 0]),
        "single_row_covariance_lag_one": float(reference[0, 1]),
        "cross_row_variances": cross,
    }


def round_one():
    methods = ("seeded", *REFS)
    keys = ["seed", "law", "scene_error_share", "n_scenes", "rows_per_scene", "method"]
    expected = [
        (s, law, *state, m)
        for s, law, state, m in product(R1_SEEDS, LAWS, DESIGNS1, methods)
    ]
    scores = grid(R1 / "scores.csv", keys, expected)
    stages = grid(
        R1 / "trajectories.csv.gz",
        [*keys, "generation"],
        [(*key, g) for key in expected if key[-1] == "seeded" for g in range(17)],
    )
    components = grid(
        R1 / "variance-components.csv", keys[:3], product(R1_SEEDS, LAWS, (0, 0.5, 1))
    )
    manifest = json.loads((R1 / "run.json").read_text())
    require(
        sorted(r["seed"] for r in manifest["banks"]) == list(R1_SEEDS),
        "R1 bank coverage",
    )
    prior = json.loads((HERE.parent / "scene-count/results/run.json").read_text())[
        "banks"
    ][0]
    for law in LAWS:
        old = generate_scene_pool(prior["seed"], law=law)
        for dom in ("A", "F", "I"):
            for f, expected_hash in prior["laws"][law]["blocks"][dom].items():
                require(
                    sha(getattr(old[dom], f)) == expected_hash,
                    ("prior default changed", dom, f),
                )
    max_middle_error = 0.0
    for entry in manifest["banks"]:
        seed = entry["seed"]
        role = int(np.random.default_rng([seed, 0, 1]).integers(8))
        law_slots = {}
        paired_fields = {}
        for law in LAWS:
            info = entry["laws"][law]
            selection = generate_checkpoint_bank(seed, law=law)["A"]
            chosen = slots(selection)
            law_slots[law] = chosen
            require(info["slots"] == chosen, "R1 source checkpoint choice")
            for key, h in info["selection_A"].items():
                require(
                    sha(getattr(selection, key)) == h, ("R1 historical A hash", key)
                )
            require(
                sorted(v["scene_error_share"] for v in info["states"]) == [0, 0.5, 1],
                "R1 share states",
            )
            common = {}
            residuals = {}
            for state in info["states"]:
                s = state["scene_error_share"]
                bank = generate_scene_pool(seed, law=law, scene_error_share=s)
                F, I = bank["F"], bank["I"]
                for domain, b in (("F", F), ("I", I)):
                    for f in ("P", "scene", "posterior", "y"):
                        h = sha(getattr(b, f))
                        require(
                            state["blocks"][domain][f] == h,
                            ("R1 source hash", domain, f),
                        )
                        if f != "P":
                            require(
                                common.setdefault((domain, f), h) == h,
                                ("share altered immutable data", domain, f),
                            )
                    for field, arr in (
                        ("scene", b.scene),
                        ("y", b.y),
                        ("q", b.posterior),
                        ("unchanged_providers", b.P[:, np.arange(8) != role]),
                    ):
                        h = sha(arr)
                        require(
                            paired_fields.setdefault((s, domain, field), h) == h,
                            ("R1 laws not paired", field),
                        )
                    if law == "signal":
                        np.testing.assert_array_equal(
                            b.P[:, role],
                            np.broadcast_to(b.posterior[:, None], b.P[:, role].shape),
                        )
                # This coupling check tests the actual generator, independently
                # of its coefficient formula, across all slots of 64 F scenes.
                if law == "equal":
                    p = F.P[:512]
                    errors = (
                        logit(p) - (1.3 / 1.8) * logit(F.posterior[:512])[:, None, None]
                    )
                    residuals[s] = errors
                    if s == 1:
                        e = errors.reshape(64, 8, 8, 8)
                        close(
                            e,
                            np.broadcast_to(e[:, :1], e.shape),
                            "shared scene residual",
                            1e-10,
                        )
                x, ax = inputs(F, chosen), inputs(I, chosen)
                require(sha(ax) == state["audit_x_sha256"], "audit columns")
                for a in state["allocations"]:
                    g, r = a["n_scenes"], a["rows_per_scene"]
                    idx = rows(F, g, r)
                    for name, v in (
                        ("indices", idx),
                        ("fit_x", x[idx]),
                        ("fit_y", F.y[idx]),
                        ("scene_ids", F.scene[idx]),
                    ):
                        require(sha(v) == a[name + "_sha256"], ("R1 allocation", name))
                    control_replay(
                        x[idx],
                        F.y[idx],
                        ax,
                        I.y,
                        I.posterior,
                        scores.loc[(seed, law, s, g, r)],
                    )
                # Independent variance calculations for the designated fixed
                # singleton against the mean, conditioning on scene membership.
                d = row_brier(x[:, role], F.y) - row_brier(x.mean(axis=1), F.y)
                by = d.reshape(-1, 8)
                means = by.mean(axis=1)
                w = ((by - means[:, None]) ** 2).sum() / (len(means) * 7)
                b = ((means - means.mean()) ** 2).sum() / (len(means) - 1)
                close(
                    components.loc[
                        (seed, law, s), ["within", "between_means", "variance_ratio"]
                    ],
                    [w, b, 8 * b / (b + 7 * w / 8)],
                    "R1 fixed variance",
                )
            if law == "equal":
                error = float(
                    np.max(
                        np.abs(
                            residuals[0.5] - (residuals[0] + residuals[1]) / np.sqrt(2)
                        )
                    )
                )
                max_middle_error = max(max_middle_error, error)
                require(
                    error < 1e-10,
                    "actual intermediate residual differs from matched Gaussian mixture",
                )
        require(
            law_slots["signal"][role] == 0
            and np.array_equal(
                np.delete(law_slots["signal"], role),
                np.delete(law_slots["equal"], role),
            ),
            "planted slot",
        )
    values = []
    for seed in R1_SEEDS:

        def penalty(share, seed=seed):
            return (
                scores.loc[(seed, "signal", share, 24, 8, "seeded"), "audit_excess"]
                - scores.loc[(seed, "signal", share, 192, 1, "seeded"), "audit_excess"]
            )

        values.append(penalty(1) - penalty(0))
    estimate = bootstrap(values, 97824)
    paired = read(R1 / "paired-contrasts.csv")
    selected = paired.query(
        "law == 'signal' and method == 'seeded' and metric == 'audit_excess' and contrast == 'gap_at_1_minus_gap_at_0'"
    )
    require(len(selected) == 1, "round-one primary row")
    close(
        selected[["estimate", "ci_low", "ci_high"]].iloc[0],
        estimate,
        "round-one primary",
    )
    by_bank = read(R1 / "primary-by-bank.csv").set_index("seed")
    close(
        by_bank.loc[list(R1_SEEDS), "interaction"],
        values,
        "round-one raw matched contrasts",
    )
    return {
        "score_rows": len(scores),
        "trajectory_rows": len(stages),
        "verified_frozen_checkpoints": len(R1_SEEDS) * 2 * 8,
        "fixed_variance_checks": len(components),
        "numeric_control_scores": len(scores) * 2 // len(methods),
        "max_half_share_residual_error": max_middle_error,
        "matched_gaussian_marginal": full_marginal(),
        "primary_interaction": named(estimate),
    }, values


def assign_folds(groups, seed):
    shuffled = np.random.default_rng(seed + 3_000_000).permutation(np.unique(groups))
    assignments = {g: i % 4 for i, g in enumerate(shuffled)}
    return np.array([assignments[g] for g in groups], dtype=np.int64)


def key_hash(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def manual_choices(fitted, x, y, groups, seed):
    """Recompute each candidate's four anchor gains from native predictions."""
    from fusionbench.eval.suite_point_gp import predict_tree

    folds = assign_folds(groups, seed)
    require(sha(folds) == fitted["fold_digest"], "independent fold assignment")
    records = fitted["archive"]
    init = fitted["snapshots"][0]["keys"]
    final = fitted["snapshots"][-1]["keys"]
    costs, sizes, gains = {}, {}, {}
    for key, rec in records.items():
        tree = rec["tree"]
        prediction = predict_tree(tree, x)
        costs[key] = row_brier(prediction, y)
        sizes[key] = tree.nodes_count
        close(rec["pooled_brier"], costs[key].mean(), "rescored candidate pooled loss")
    anchor = min(init, key=lambda key: (costs[key].mean(), sizes[key]))
    require(anchor == fitted["anchor"], "independent initial anchor")
    for key, rec in records.items():
        net = costs[anchor] - costs[key]
        # Accumulate each fold in original row order and use its actual row
        # count. This is independent of the helper's grouped reduction.
        vals = [sum(map(float, net[folds == f])) / (folds == f).sum() for f in range(4)]
        gains[key] = sorted(vals)[1]
        close(vals, rec["fold_gains"], "four observed fold gains")
        close(gains[key], rec["robust_gain"], "second-smallest gain")
    candidates = [anchor, *final]
    manual = {
        fitted["path"] + "_pooled": min(
            candidates, key=lambda key: (costs[key].mean(), key != anchor, sizes[key])
        ),
        fitted["path"] + "_fold": min(
            candidates, key=lambda key: (-gains[key], key != anchor, sizes[key])
        ),
    }
    if fitted["path"] == "pooled":
        manual["anchor"] = anchor
        manual["pooled_final"] = min(
            final, key=lambda key: (costs[key].mean(), sizes[key])
        )
    require(manual == fitted["choices"], "independent anchored selections")
    return manual, len(records)


def replay(x, y, ax, ay, q, groups, seed, signal, expected):
    from fusionbench.eval.suite_point_gp import predict_tree

    engine = run_path(str(HERE / "search.py"))
    output = {}
    for path in ("pooled", "fold"):
        options = {
            "groups": groups,
            "data_seed": seed,
            "search_seed": seed + 1_000_000,
            "path": path,
        }
        original = engine["fit_search"](x, y, **options)
        winners, n_eval = manual_choices(original, x, y, groups, seed)
        clean_rows, clean_trace = engine["score_search"](
            original, ax, ay, audit_q=q, signal_provider=signal
        )
        # New fit object removes any possibility of reusing cached decisions.
        fresh = engine["fit_search"](x, y, **options)
        substitute = np.full_like(q, 0.5)
        poisoned_rows, poisoned_trace = engine["score_search"](
            fresh, ax, 1 - ay, audit_q=substitute, signal_provider=signal
        )
        require(
            original["choices"] == fresh["choices"],
            "held-out outcomes changed selection",
        )
        for a, b in zip(clean_trace, poisoned_trace, strict=True):
            for field in (
                "active_digest",
                "path_digest",
                "objective_evaluations",
                "founder_digest",
                "fold_digest",
            ):
                require(a[field] == b[field], ("poisoned path changed", field))
        checks = {}
        for method, key in winners.items():
            (clean,) = [r for r in clean_rows if r["method"] == method]
            (poisoned,) = [r for r in poisoned_rows if r["method"] == method]
            selected = expected.loc[method]
            require(
                clean["selected_digest"]
                == key_hash(key)
                == poisoned["selected_digest"]
                == selected.selected_digest,
                "archived winning structure",
            )
            require(clean["path_digest"] == selected.path_digest, "archived path")
            p = predict_tree(original["archive"][key]["tree"], ax)
            alt = predict_tree(fresh["archive"][key]["tree"], ax)
            np.testing.assert_array_equal(p, alt)
            clean_excess = conditional_excess(p[:, 1], q)
            close(
                [row_brier(p, ay).mean(), clean_excess],
                selected[["audit_brier", "audit_excess"]],
                "independent selected scores",
            )
            close(
                [
                    row_brier(alt, 1 - ay).mean(),
                    conditional_excess(alt[:, 1], substitute),
                ],
                [poisoned["audit_brier"], poisoned["audit_excess"]],
                "replaced audit scores",
            )
            require(
                abs(clean_excess - poisoned["audit_excess"]) > TOL
                and abs(clean["audit_brier"] - poisoned["audit_brier"]) > TOL,
                "expected changed audit outcomes",
            )
            checks[method] = {
                "structure_sha256": key_hash(key),
                "audit_prediction_sha256": sha(p),
                "excess": clean_excess,
                "constant_q_excess": poisoned["audit_excess"],
            }
        output[path] = {
            "unique_candidates_rechecked": n_eval,
            "verified_generations": len(clean_trace),
            "independent_choices": checks,
        }
    return output


def round_two():
    methods = (*POLICIES, *REFS)
    keys = ["seed", "law", "scene_error_share", "n_scenes", "rows_per_scene"]
    cells = [(seed, law, *d) for seed, law, d in product(R2_SEEDS, LAWS, DESIGNS2)]
    scores = grid(
        R2 / "scores.csv",
        [*keys, "method"],
        [(*c, m) for c, m in product(cells, methods)],
    )
    stages = grid(
        R2 / "trajectories.csv.gz",
        [*keys, "path", "generation"],
        [(*c, p, g) for c, p, g in product(cells, ("pooled", "fold"), range(17))],
    )
    info = json.loads((R2 / "run.json").read_text())
    require(
        sorted(e["seed"] for e in info["banks"]) == list(R2_SEEDS), "R2 bank coverage"
    )
    decisions = None
    for entry in info["banks"]:
        seed = entry["seed"]
        require(
            entry["group_seed"] == seed + 3_000_000
            and entry["search_seed"] == seed + 1_000_000,
            "fit seeds",
        )
        manifest_keys = {
            (m["law"], m["scene_error_share"], m["n_scenes"], m["rows_per_scene"]): m
            for m in entry["cells"]
        }
        require(
            set(manifest_keys) == {(law, *d) for law, d in product(LAWS, DESIGNS2)}
            and len(entry["cells"]) == 6,
            "manifest matched grid",
        )
        for law in LAWS:
            A = generate_checkpoint_bank(seed, law=law)["A"]
            chosen = slots(A)
            for s in (0.0, 1.0):
                pool = generate_scene_pool(
                    seed,
                    law=law,
                    n_fit_scenes=192,
                    n_audit_scenes=512,
                    scene_error_share=s,
                )
                F, I = pool["F"], pool["I"]
                ax = inputs(I, chosen)
                for share, g, r in DESIGNS2:
                    if share != s:
                        continue
                    idx = rows(F, g, r)
                    x, y, groups = inputs(F, chosen)[idx], F.y[idx], F.scene[idx]
                    cell = seed, law, s, g, r
                    sub = scores.loc[cell]
                    meta = manifest_keys[law, s, g, r]
                    require(meta["slot_ids"] == chosen, "R2 independent A slots")
                    for k, v in (
                        ("selection_A_x", A.P),
                        ("selection_A_y", A.y),
                        ("fit_x", x),
                        ("fit_y", y),
                        ("group", groups),
                        ("audit_x", ax),
                        ("audit_y", I.y),
                        ("audit_q", I.posterior),
                        ("audit_scene", I.scene),
                    ):
                        require(sha(v) == meta[k + "_sha256"], ("R2 input mismatch", k))
                    control_replay(x, y, ax, I.y, I.posterior, sub)
                    fold_ids = assign_folds(groups, seed)
                    for k in (
                        "founder_digest",
                        "anchor_digest",
                        "fold_digest",
                        "fold_row_counts",
                    ):
                        require(
                            meta["paths"]["pooled"][k] == meta["paths"]["fold"][k],
                            ("unmatched founders/folds", k),
                        )
                    for path in ("pooled", "fold"):
                        live = stages.loc[(*cell, path)]
                        record = meta["paths"][path]
                        require(record["fold_digest"] == sha(fold_ids), "fold hash")
                        np.testing.assert_array_equal(
                            record["fold_row_counts"],
                            np.bincount(fold_ids, minlength=4),
                        )
                        for k in (
                            "path_digest",
                            "founder_digest",
                            "fold_digest",
                            "anchor_digest",
                            "objective_evaluations",
                        ):
                            require(
                                live.loc[16, k] == record[k], ("matched final state", k)
                            )
                        for terminal in ("pooled", "fold"):
                            m = sub.loc[path + "_" + terminal]
                            require(
                                m.path_digest == record["path_digest"],
                                "terminal policy changed search",
                            )
                            close(
                                m.robust_gain,
                                sorted(m[[f"fold_gain_{j}" for j in range(4)]])[1],
                                "archived promotion utility",
                            )
                            require(
                                (
                                    m.pooled_gain
                                    if terminal == "pooled"
                                    else m.robust_gain
                                )
                                >= -TOL,
                                "selected below anchor on deployment objective",
                            )
                    if cell == (R2_SEEDS.start, "signal", 1.0, 24, 8):
                        decisions = replay(
                            x,
                            y,
                            ax,
                            I.y,
                            I.posterior,
                            groups,
                            seed,
                            meta["signal_provider"],
                            sub,
                        )
    delta = [
        scores.loc[(seed, "signal", 1.0, 24, 8, "fold_fold"), "audit_excess"]
        - scores.loc[(seed, "signal", 1.0, 24, 8, "pooled_pooled"), "audit_excess"]
        for seed in R2_SEEDS
    ]
    estimate = bootstrap(delta, 97825)
    contrasts = read(R2 / "contrasts.csv")
    selected = contrasts.query(
        "law=='signal' and scene_error_share==1 and n_scenes==24 and rows_per_scene==8 and metric=='audit_excess' and method=='fold_fold' and reference=='pooled_pooled'"
    )
    require(len(selected) == 1, "R2 primary estimand")
    close(
        selected[["estimate", "ci_low", "ci_high"]].iloc[0],
        estimate,
        "round-two primary",
    )
    return {
        "score_rows": len(scores),
        "trajectory_rows": len(stages),
        "matched_two_path_cells": len(cells),
        "numeric_control_scores": len(cells) * 2,
        "audit_replay": decisions,
        "primary_fold_fold_minus_pooled_pooled": named(estimate),
    }


def standard_panel():
    from fusionbench import generate_dataset

    scenarios = list(load_preset("standard-v4")["scenarios"])
    require(len(scenarios) == 25, "expected standard-v4 with 25 scenarios")
    seeds = (287300, 287301)
    keys = ["scenario_id", "data_seed"]
    scores = grid(
        R2 / "panel-scores.csv",
        [*keys, "method"],
        product(scenarios, seeds, (*POLICIES, *REFS)),
    )
    stages = grid(
        R2 / "panel-trajectories.csv",
        [*keys, "path", "generation"],
        product(scenarios, seeds, ("pooled", "fold"), range(17)),
    )
    for scenario, seed in product(scenarios, seeds):
        bank = generate_dataset(scenario, n_per_split=512, seed=seed)
        x, y, ax, ay = bank.fit_preds, bank.fit_y, bank.test_preds, bank.test_y
        sub = scores.loc[(scenario, seed)]
        for key, data in (
            ("fit_inputs", x),
            ("fit_labels", y),
            ("test_inputs", ax),
            ("test_labels", ay),
        ):
            require(
                (sub[key + "_sha256"] == sha(data)).all(),
                ("panel input mismatch", scenario, seed, key),
            )
        for col in ("founder_digest", "anchor_digest", "fold_digest"):
            require(
                sub.loc[list(POLICIES), col].nunique() == 1,
                ("panel path setup", scenario, seed, col),
            )
        require(
            sub.loc["fold_fold", "fold_digest"]
            == sha(assign_folds(np.arange(len(y)), seed)),
            "row groups for suite",
        )
        for path in ("pooled", "fold"):
            stage = stages.loc[(scenario, seed, path, 16)]
            for terminal in ("pooled", "fold"):
                m = sub.loc[path + "_" + terminal]
                require(m.path_digest == stage.path_digest, "terminal path")
        rank = sorted(range(x.shape[1]), key=lambda j: row_brier(x[:, j], y).mean())
        n = min(
            range(1, len(rank) + 1),
            key=lambda k: row_brier(x[:, rank[:k]].mean(axis=1), y).mean(),
        )
        for method, ids in (("mean", list(range(x.shape[1]))), ("top_n", rank[:n])):
            close(
                row_brier(x[:, ids].mean(axis=1), y).mean(),
                sub.loc[method, "fit_brier"],
                ("panel fit", scenario, method),
            )
            close(
                row_brier(ax[:, ids].mean(axis=1), ay).mean(),
                sub.loc[method, "test_brier"],
                ("panel test", scenario, method),
            )
    table = read(R2 / "panel-contrasts.csv")
    selected = table.query("method=='fold_fold' and reference=='pooled_pooled'")
    require(len(selected) == 1, "standard panel counterpart")
    delta = np.array(
        [
            [
                scores.loc[(s, i, "fold_fold"), "test_brier"]
                - scores.loc[(s, i, "pooled_pooled"), "test_brier"]
                for i in seeds
            ]
            for s in scenarios
        ]
    )
    sel = np.random.default_rng(97825).integers(0, 2, size=(10000, len(scenarios), 2))
    draws = delta[np.arange(len(scenarios))[None, :, None], sel].mean(axis=(1, 2))
    estimate = [delta.mean(), *np.quantile(draws, [0.025, 0.975])]
    close(
        selected[["estimate", "ci_low", "ci_high"]].iloc[0],
        estimate,
        "equal-weight suite contrast",
    )
    return {
        "verified_scenarios": len(scenarios),
        "score_rows": len(scores),
        "trajectory_rows": len(stages),
        "verified_numeric_mean_and_top_n": len(scenarios) * len(seeds) * 2,
        "fold_fold_minus_pooled_pooled": named(estimate),
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        one, _ = round_one()
        two = round_two()
        panel = standard_panel()
    target = R2 / "verification.json"
    target.write_text(
        json.dumps(
            {"round_one": one, "round_two": two, "standard_v4": panel},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Verified both locked studies and the standard-v4 panel. Wrote {target}")
