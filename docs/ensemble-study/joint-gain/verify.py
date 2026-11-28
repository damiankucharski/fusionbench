"""Recompute the three locked primaries and check fit-only native deployment."""

import json
from itertools import product
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
)
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PREVIOUS = HERE.parent / "fold-promotion"
# Pure checks from the preceding independent audit retain its verified full-class
# Brier, Top-N tie rules, F1 checkpoint selection and four-fold permutation.
old = run_path(str(PREVIOUS / "verify.py"))
require, close, grid, read, sha, row_brier, key_hash, assign_folds = (
    old[k]
    for k in (
        "require",
        "close",
        "grid",
        "read",
        "sha",
        "row_brier",
        "key_hash",
        "assign_folds",
    )
)
engine = run_path(str(PREVIOUS / "search.py"))
REFS = ("mean", "top_n", "prob_lr", "regularized")
RULES = ("pooled", "fold", "veto", "joint")
PATHS = ("pooled", "fold", "joint")
CI = ["estimate", "ci_low", "ci_high"]


def methods(paths):
    extra = ("anchor", "pooled_final") if "pooled" in paths else ("anchor",)
    return (*extra, *(f"{p}_{rule}" for p in paths for rule in RULES), *REFS)


def checkpoint(seed, law, n):
    a = generate_checkpoint_bank(seed, law=law)["A"]
    selected = old["slots"](a)
    pool = generate_scene_pool(
        seed, law=law, n_fit_scenes=192, n_audit_scenes=512, scene_error_share=1.0
    )
    f, i = pool["F"], pool["I"]
    x, ax = (old["inputs"](block, selected) for block in (f, i))
    return (
        (x[:n], f.y[:n], f.scene[:n], f.posterior[:n], ax, i.y, i.posterior),
        a,
        selected,
    )


def switch(seed, law, n):
    # Build the six anonymous columns from the public law's primitive streams.
    parent, child = np.random.default_rng([seed, 15]).permutation(6)[:2]

    def split(stream, count):
        gen = np.random.default_rng([seed, stream])
        p = gen.integers(1, 64, size=(count, 6)) / 64
        q = (
            p[:, parent]
            if law == "singleton"
            else np.minimum(p[:, parent], p[:, child])
        )
        return np.stack((1 - p, p), axis=-1), (gen.random(count) < q).astype(int), q

    x, y, fq = split(1, 1536)
    ax, ay, aq = split(2, 4096)
    return (x[:n], y[:n], np.arange(n), fq[:n], ax, ay, aq), int(parent), int(child)


def paired(values, frame, seed, **filters):
    row = frame
    for k, v in filters.items():
        row = row.loc[row[k] == v]
    require(len(row) == 1, ("paired row", filters))
    estimate = old["bootstrap"](values, seed)
    close(estimate, row.iloc[0][CI], filters)
    return dict(zip(CI, estimate, strict=True))


def matched(cell, paths, trace=None):
    native = cell.loc[[k for k in cell.index if k not in REFS]]
    for k in ("anchor_digest", "founder_digest", "fold_digest"):
        require(
            native[k].notna().all() and native[k].nunique() == 1,
            ("matched native inputs", k),
        )
    anchor = cell.loc["anchor", "selected_digest"]
    for path in paths:
        p = cell.loc[[f"{path}_{r}" for r in RULES]]
        fold, veto, joint = (p.loc[f"{path}_{r}"] for r in ("fold", "veto", "joint"))
        require(p.path_digest.nunique() == 1, ("shared final menu", path))
        for rule in (veto, joint):
            if rule.selected_digest != anchor:
                require(
                    rule.pooled_gain > 0 and rule.robust_gain > 0,
                    "ineligible nonanchor",
                )
        fold_ok = fold.pooled_gain > 0 and fold.robust_gain > 0
        require(
            veto.selected_digest == (fold.selected_digest if fold_ok else anchor),
            "veto choice",
        )
        if fold_ok:
            require(
                joint.selected_digest == fold.selected_digest,
                "ordinary winner eligible",
            )
        if trace is not None:
            s = trace.loc[(path, 16)]
            require(
                s.path_digest == fold.path_digest
                and s.unique_evaluated == fold.unique_evaluated,
                "final terminal trace",
            )


def gains(delta, fold_ids):
    pooled = float(np.mean(delta))
    per_fold = [
        sum(map(float, delta[fold_ids == i])) / int((fold_ids == i).sum())
        for i in range(4)
    ]
    return pooled, sorted(per_fold)[1], per_fold


def corners():
    from okapi.fold_gain import FoldGain

    # The unit-scale anchor permits exact binary fractions on both sides of zero.
    ids = np.arange(8)
    gate = FoldGain(np.ones(8), ids, group_seed=3_000_000)
    folds = assign_folds(ids, 0)

    def candidate(vector, size):
        delta = np.asarray(vector)[folds]
        score = gate.score(1.0 - delta)
        p, f, four = gains(delta, folds)
        close(score.fold_gains, four, "four known fold gains")
        close(
            [
                score.pooled_gain,
                score.robust_gain,
                score.joint_gain,
                gate.joint_fitness(score),
            ],
            [p, f, min(p, f), 1.0 - 0.5 * (1.0 - min(p, f))],
            "three continuous training criteria",
        )
        require(
            score.supports_replacement == (p > 0 and f > 0), "two strict positive gains"
        )
        return {
            "tree": SimpleNamespace(nodes_count=size),
            "robust_gain": f,
            "supports_replacement": score.supports_replacement,
        }

    archive = {
        "anchor": candidate([0.0] * 4, 1),
        "pooled_zero": candidate([-0.75, 0.25, 0.25, 0.25], 2),
        "fold_zero": candidate([0.0, 0.0, 0.25, 0.25], 2),
        "pooled_negative": candidate([-1.0, 0.25, 0.25, 0.25], 2),
        "negative": candidate([-0.25] * 4, 2),
        "long": candidate([0.125] * 4, 5),
        "early": candidate([0.125] * 4, 2),
        "late": candidate([0.125] * 4, 2),
    }
    for k in ("pooled_zero", "fold_zero", "negative", "pooled_negative"):
        require(
            engine["joint_finalists"](archive, [k], "anchor", k)
            == ("anchor", "anchor"),
            ("veto boundary", k),
        )
    final = ["pooled_negative", "long", "early", "late"]
    require(
        engine["joint_finalists"](archive, final, "anchor", final[0])
        == ("anchor", "early"),
        "rerank eligible runner-up, size and order",
    )
    require(
        engine["joint_finalists"](archive, ["long", "late", "early"], "anchor", "late")
        == ("late", "late"),
        "eligible winner retained",
    )
    return {
        "strict_zero_gates": 2,
        "negative_cases": 2,
        "size_then_survivor_order": True,
        "veto_and_rerank_diverge": True,
    }


def audit_replay(data, seed, path, archived, *, check_poison=False, witness=False):
    x, y, groups, fq, ax, ay, aq = data
    fit = engine["fit_search"](
        x,
        y,
        groups=groups,
        data_seed=seed,
        search_seed=seed + 1_000_000,
        path=path,
        joint_study=True,
    )
    folds = assign_folds(groups, seed)
    require(sha(folds) == fit["fold_digest"], "independent fold grouping")
    records, snapshots = fit["archive"], fit["snapshots"]
    init, final = snapshots[0]["keys"], snapshots[-1]["keys"]
    require(len(snapshots) == 17, "16 native generations")
    losses, sizes, audit_loss, audit_risk, exact, predictions = {}, {}, {}, {}, {}, {}
    for k, r in records.items():
        t = r["tree"]
        p, a = predict_tree(t, x), predict_tree(t, ax)
        losses[k] = row_brier(p, y)
        sizes[k] = t.nodes_count
        audit_loss[k] = row_brier(a, ay).mean()
        audit_risk[k] = 2 * np.square(a[:, 1] - aq).mean()
        exact[k] = (
            np.max(np.abs(a[:, 1] - aq)) <= 1e-12
            and np.max(np.abs(p[:, 1] - fq)) <= 1e-12
        )
        predictions[k] = sha(a)
        close(r["pooled_brier"], losses[k].mean(), "rebound fit score")
    anchor = min(init, key=lambda k: (losses[k].mean(), sizes[k]))
    require(anchor == fit["anchor"], "empirical founder selection")
    require(
        set(init) == {k for k, r in records.items() if r["generation"] == 0},
        "only founders at generation zero",
    )
    if witness:
        require(
            len(init) == x.shape[1] + 1
            and {
                records[k]["tree"].unique_value_node_ids[0]
                for k in init
                if sizes[k] == 1
            }
            == {f"m{j}.npy" for j in range(x.shape[1])},
            "all singleton founders",
        )
        compound = [k for k in init if sizes[k] > 1]
        require(
            len(compound) == 1
            and compound[0] == fit["mean_key"]
            and all(not exact[k] for k in init),
            "MIN witness absent from live founders",
        )
        close(
            predict_tree(records[compound[0]]["tree"], x),
            x.mean(1),
            "only compound founder is arithmetic mean",
        )
    p_gain, f_gain = {}, {}
    for k, r in records.items():
        p, f, four = gains(losses[anchor] - losses[k], folds)
        p_gain[k], f_gain[k] = p, f
        close(
            [p, f, min(p, f), *four],
            [r["pooled_gain"], r["robust_gain"], r["joint_gain"], *r["fold_gains"]],
            "archive paired gains",
        )
        require(
            r["supports_replacement"] == (p > 0 and f > 0), "candidate strict support"
        )
    menu = [anchor, *final]
    fold = min(menu, key=lambda k: (-f_gain[k], k != anchor, sizes[k]))
    can = lambda k: p_gain[k] > 0 and f_gain[k] > 0
    choices = {
        path + "_fold": fold,
        path + "_pooled": min(
            menu, key=lambda k: (losses[k].mean(), k != anchor, sizes[k])
        ),
        path + "_veto": fold if can(fold) else anchor,
        path + "_joint": min(
            (k for k in final if can(k)),
            key=lambda k: (-f_gain[k], sizes[k]),
            default=anchor,
        ),
    }
    if path == "pooled":
        choices.update(
            anchor=anchor,
            pooled_final=min(final, key=lambda k: (losses[k].mean(), sizes[k])),
        )
    require(choices == fit["choices"], "all deployment rules reranked independently")
    for method, k in choices.items():
        row = archived.loc[method]
        require(
            key_hash(k) == row.selected_digest
            and snapshots[-1]["path_digest"] == row.path_digest,
            "regrown path & chosen program",
        )
        close(
            [
                losses[k].mean(),
                audit_loss[k],
                audit_risk[k],
                min(audit_risk.values()),
                min(audit_risk[k] for k in final),
            ],
            [
                row.fit_brier,
                row.audit_brier,
                row.audit_excess,
                row.best_ever_audit_excess,
                row.best_final_audit_excess,
            ],
            "regrown audit scores",
        )
        require(
            bool(exact[k]) == row.selected_is_exact
            and sum(exact.values()) == row.n_exact_visited
            and sum(exact[k] for k in set(final)) == row.n_exact_final,
            "observational exactness on both splits",
        )
    if check_poison:
        # Repeat the fit without accepting any audit arrays; replace all audit
        # labels, the audit posterior and the diagnostic fit posterior on scoring.
        other = engine["fit_search"](
            x,
            y,
            groups=groups,
            data_seed=seed,
            search_seed=seed + 1_000_000,
            path=path,
            joint_study=True,
        )
        bad, _ = engine["score_search"](
            other, ax, 1 - ay, fit_q=1 - fq, audit_q=np.full_like(aq, 0.5)
        )
        require(
            other["choices"] == choices and other["snapshots"] == snapshots,
            "unchanged evolution and terminal menus under evaluation changes",
        )
        for b in bad:
            k = choices[b["method"]]
            require(
                key_hash(k) == b["selected_digest"]
                and sha(predict_tree(other["archive"][k]["tree"], ax))
                == predictions[k],
                "unchanged predictions after audit substitution",
            )
            require(
                abs(b["audit_brier"] - audit_loss[k]) > 1e-12,
                "negative control changed measured audit loss",
            )
    return {
        "seed": seed,
        "path": path,
        "unique_structures": len(records),
        "final_structures": len(set(final)),
        "audit_swapped": check_poison,
        **({"min_witness_excluded_from_founders": True} if witness else {}),
    }


def first_round():
    seeds, laws, sizes = range(288100, 288148), ("equal", "signal"), (24, 192)
    paths = PATHS[:2]
    scores = grid(
        OUT / "r1-scores.csv",
        ["seed", "law", "n_scenes", "method"],
        product(seeds, laws, sizes, methods(paths)),
    )
    trace = grid(
        OUT / "r1-trajectories.csv.gz",
        ["seed", "law", "n_scenes", "path", "generation"],
        product(seeds, laws, sizes, paths, range(17)),
    )
    events = grid(
        OUT / "r1-events.csv",
        ["seed", "law", "n_scenes", "path"],
        product(seeds, laws, sizes, paths),
    )
    for seed, law, n in product(seeds, laws, sizes):
        cell = scores.loc[(seed, law, n)]
        matched(cell, paths, trace.loc[(seed, law, n)])
        for path in paths:
            e = events.loc[(seed, law, n, path)]
            a, fold, veto, joint = (
                cell.loc[k, "selected_digest"]
                for k in ("anchor", path + "_fold", path + "_veto", path + "_joint")
            )
            require(
                e.blocked == (veto != fold)
                and e.rescued == (veto != fold and joint != a),
                "R1 blocked/replacement events",
            )
    manifest = json.loads((OUT / "r1-run.json").read_text())
    # Recreate the original selection block and all inputs for one entire seed.
    bank = next(b for b in manifest["banks"] if b["seed"] == seeds[0])
    for meta in bank["cells"]:
        seed, law, n = meta["seed"], meta["law"], meta["n_rows"]
        data, a, slots = checkpoint(seed, law, n)
        x, y, _g, _fq, ax, ay, aq = data
        require(slots == meta["slot_ids"], "historical A-only strict F1 slots")
        for k, v in zip(
            ("fit_x", "fit_y", "group", "fit_q", "audit_x", "audit_y", "audit_q"),
            data,
            strict=True,
        ):
            require(sha(v) == meta[k + "_sha256"], "R1 raw input " + k)
        require(
            sha(a.P) == meta["selection_A_x_sha256"]
            and sha(a.y) == meta["selection_A_y_sha256"],
            "original selection input",
        )
        cell = scores.loc[(seed, law, n // 8)]
        old["control_replay"](x, y, ax, ay, aq, cell)
        if law == "signal" and n == 192:
            replay = audit_replay(data, seed, "fold", cell, check_poison=True)
    delta = scores.xs(("signal", 24), level=("law", "n_scenes")).audit_excess.unstack()
    primary = paired(
        delta.fold_joint - delta.fold_fold,
        read(OUT / "r1-contrasts.csv"),
        97826,
        law="signal",
        n_scenes=24,
        method="fold_joint",
        reference="fold_fold",
        metric="audit_excess",
    )
    return {
        "scores": len(scores),
        "trajectory_rows": len(trace),
        "terminal_events": len(events),
        "banks": len(seeds),
        "primary": primary,
        "native_replay": replay,
    }


def witness_tree(x, parent, child):
    from okapi.node import MinNode, ValueNode
    from okapi.tree import Tree

    operand = ValueNode(None, x[:, child, 1:2], f"m{child}.npy")
    tree = Tree(ValueNode([MinNode([operand])], x[:, parent, 1:2], f"m{parent}.npy"))
    require(
        set(tree.unique_value_node_ids) == {f"m{child}.npy", f"m{parent}.npy"},
        "two distinct native MIN operands",
    )
    return tree


def second_round():
    seeds, laws, sizes = range(288200, 288232), ("singleton", "min"), (192, 1536)
    cells = list(product(seeds, laws, sizes))
    scores = grid(
        OUT / "r2-scores.csv",
        ["seed", "law", "n_rows", "method"],
        product(seeds, laws, sizes, methods(("fold",))),
    )
    keys = ["seed", "law", "n_rows"]
    witness = grid(OUT / "r2-witness.csv", keys, cells)
    diagnostics = grid(OUT / "r2-diagnostics.csv", keys, cells)
    for seed, law, n in cells:
        cell = scores.loc[(seed, law, n)]
        matched(cell, ("fold",))
        data, parent, child = switch(seed, law, n)
        x, y, g, fq, ax, ay, aq = data
        meta = diagnostics.loc[(seed, law, n)]
        require(
            sha(x) == meta.fit_x_sha256
            and sha(ax) == meta.audit_x_sha256
            and [parent, child] == [meta.parent_index, meta.child_index],
            "R2 rows and roles",
        )
        folds = assign_folds(g, seed)
        require(sha(folds) == meta.fold_digest, "R2 independent row grouping")
        founders = [x[:, j, 1] for j in range(x.shape[1])] + [x[:, :, 1].mean(axis=1)]
        i = min(
            range(len(founders)), key=lambda j: (row_brier(founders[j], y).mean(), j)
        )
        anchor = founders[i]
        ref = cell.loc["anchor"]
        close(ref.fit_brier, row_brier(anchor, y).mean(), "empirical founder")
        ap = ax[:, :, 1].mean(axis=1) if i == x.shape[1] else ax[:, i, 1]
        close(ref.audit_excess, 2 * np.square(ap - aq).mean(), "same anchor deployed")
        w = witness.loc[(seed, law, n)]
        p, f, four = gains(row_brier(anchor, y) - row_brier(fq, y), folds)
        close(
            [p, f, min(p, f), *four],
            [
                w.pooled_gain,
                w.fold_gain,
                w.joint_gain,
                *[w[f"fold_gain_{j}"] for j in range(4)],
            ],
            "separate fit-only witness",
        )
        require(
            (w.accept_pooled, w.accept_fold, w.accept_joint, w.anchor_is_witness)
            == (p > 0, f > 0, p > 0 and f > 0, np.array_equal(anchor, fq)),
            "witness strict acceptance",
        )
        if law == "min":
            require(
                not any(np.array_equal(p, fq) for p in founders),
                "no founder realizes MIN in this bank",
            )
        close(
            [meta.retention_gap, meta.joint_choice_gap, meta.fold_choice_gap],
            [
                meta.best_menu_audit_excess - meta.best_ever_audit_excess,
                cell.loc["fold_joint", "audit_excess"] - meta.best_menu_audit_excess,
                cell.loc["fold_fold", "audit_excess"] - meta.best_menu_audit_excess,
            ],
            "oracle-only diagnostic decomposition",
        )
        if seed == seeds[0]:
            old["control_replay"](x, y, ax, ay, aq, cell)
            if law == "min" and n == 192:
                tree = witness_tree(x, parent, child)
                for z, q in ((x, fq), (ax, aq)):
                    require(
                        np.array_equal(predict_tree(tree, z)[:, 1], q),
                        "native exact MIN on fit and audit",
                    )
                ticks = np.arange(1, 64) / 64
                u, v = np.meshgrid(ticks, ticks, indexing="ij")
                z = np.full((u.size, 6), 0.5)
                z[:, parent], z[:, child] = u.ravel(), v.ravel()
                require(
                    np.array_equal(
                        predict_tree(tree, z)[:, 1], np.minimum(u, v).ravel()
                    ),
                    "bivariate MIN on all 3969 ordered grid pairs",
                )
                replay = audit_replay(data, seed, "fold", cell, witness=True)
    delta = scores.xs(("min", 192), level=("law", "n_rows")).audit_excess.unstack()
    primary = paired(
        delta.fold_joint - delta.fold_fold,
        read(OUT / "r2-summary.csv"),
        97827,
        law="min",
        n_rows=192,
        method="fold_joint",
        reference="fold_fold",
        metric="audit_excess",
    )
    return {
        "scores": len(scores),
        "banks": len(seeds),
        "fit_only_witness_decisions": len(witness),
        "min_grid_pairs": 63**2,
        "noninferior": primary["ci_high"] < 0.002,
        "upper_bound_below_zero": primary["ci_high"] < 0.0,
        "primary": primary,
        "native_replay": replay,
    }


def controlled():
    seeds, laws = range(288300, 288324), ("signal", "min")
    names = methods(PATHS)
    cells = [
        (i, law, i + d * 1_000_000) for i, law, d in product(seeds, laws, (1, 2, 3))
    ]
    scores = grid(
        OUT / "r3-controlled-scores.csv",
        ["seed", "law", "search_seed", "method"],
        ((*c, m) for c in cells for m in names),
    )
    banks = grid(
        OUT / "r3-controlled-banks.csv",
        ["law", "seed", "method"],
        product(laws, seeds, names),
    )
    records = json.loads((OUT / "r3-controlled-run.json").read_text())["inputs"]
    ids = {(r["seed"], r["law"]): r for r in records}
    require(
        len(ids) == len(records) == len(seeds) * len(laws),
        "R3 controlled source coverage",
    )
    replays = []
    for seed, law in product(seeds, laws):
        sub = scores.loc[(seed, law)]
        data = (
            checkpoint(seed, law, 192)[0]
            if law == "signal"
            else switch(seed, law, 192)[0]
        )
        x, y, g, _fq, ax, ay, aq = data
        for k, v in zip(
            ("fit_x", "fit_y", "group", "fit_q", "audit_x", "audit_y", "audit_q"),
            data,
            strict=True,
        ):
            require(sha(v) == ids[seed, law][k], "R3 raw input " + k)
        n = sub[sub.anchor_digest.notna()]
        require(
            n.fold_digest.nunique() == n.anchor_digest.nunique() == 1
            and n.iloc[0].fold_digest == sha(assign_folds(g, seed)),
            "matched folds and fixed anchor across randomized searches",
        )
        for sr in (seed + offset * 1_000_000 for offset in (1, 2, 3)):
            matched(sub.loc[sr], PATHS)
        for ref in REFS:
            a = sub.xs(ref, level="method")
            require(
                a[["audit_excess", "audit_brier", "fit_brier", "n_providers"]]
                .nunique()
                .eq(1)
                .all(),
                "references held fixed across search seeds",
            )
        cell = sub.loc[seed + 1_000_000]
        old["control_replay"](x, y, ax, ay, aq, cell)
        if seed == seeds[0] and law == "min":
            for path in PATHS:
                replays.append(audit_replay(data, seed, path, cell))
    wide = scores.audit_excess.groupby(["law", "seed", "method"]).agg(
        ["mean", "min", "max"]
    )
    wide["range"] = wide["max"] - wide["min"]
    wide["worst"] = wide["max"]
    close(
        wide[["mean", "range", "worst"]],
        banks[["mean", "range", "worst"]],
        "per-bank repeated-search sensitivity",
    )
    contrasts = read(OUT / "r3-controlled-contrasts.csv")
    for r in contrasts.itertuples():
        v = wide.loc[r.law, r.metric].unstack()
        close(
            old["bootstrap"](v[r.method] - v[r.reference], 97828),
            [r.estimate, r.ci_low, r.ci_high],
            "bootstrap across independent data banks",
        )
    return {
        "scores": len(scores),
        "banks": len(seeds),
        "searches_per_bank_law": 3,
        "paired_law_search_cells": len(cells),
        "intervals_recomputed": len(contrasts),
        "independent_native_replays": replays,
    }


def panel():
    families = tuple(load_preset("standard-v4")["scenarios"])
    seeds = range(288400, 288404)
    require(len(families) == len(set(families)) == 25, "25 fixed families")
    scores = grid(
        OUT / "r3-panel-scores.csv",
        ["scenario_id", "data_seed", "method"],
        product(families, seeds, methods(PATHS)),
    )
    trace = grid(
        OUT / "r3-panel-trajectories.csv.gz",
        ["scenario_id", "data_seed", "path", "generation"],
        product(families, seeds, PATHS, range(17)),
    )
    for family, seed in product(families, seeds):
        data = generate_dataset(family, n_per_split=512, seed=seed)
        x, y, ax, ay = data.fit_preds, data.fit_y, data.test_preds, data.test_y
        sub = scores.loc[(family, seed)]
        matched(sub, PATHS, trace.loc[(family, seed)])
        for k, v in (
            ("fit_inputs", x),
            ("fit_labels", y),
            ("test_inputs", ax),
            ("test_labels", ay),
        ):
            require(sub[k + "_sha256"].eq(sha(v)).all(), "panel bank " + k)
        require(
            sub.search_seed.eq(0).all()
            and sub.loc["anchor", "fold_digest"]
            == sha(assign_folds(np.arange(len(y)), seed)),
            "panel fixed search and independent folds",
        )
        rank = sorted(range(x.shape[1]), key=lambda j: row_brier(x[:, j], y).mean())
        n = min(
            range(1, len(rank) + 1),
            key=lambda n: row_brier(x[:, rank[:n]].mean(axis=1), y).mean(),
        )
        for method, cols in (("top_n", rank[:n]), ("mean", list(range(x.shape[1])))):
            close(
                [
                    row_brier(z[:, cols].mean(axis=1), lab).mean()
                    for z, lab in ((x, y), (ax, ay))
                ],
                sub.loc[method, ["fit_brier", "test_brier"]],
                ("panel independent controls", family, seed, method),
            )

    contrasts = read(OUT / "r3-panel-contrasts.csv")
    pairs = tuple(contrasts[["method", "reference"]].itertuples(index=False, name=None))
    require(
        len(set(pairs)) == len(pairs) and ("joint_joint", "fold_joint") in pairs,
        "unique family contrast specifications",
    )
    by = grid(
        OUT / "r3-panel-family-contrasts.csv",
        ["scenario_id", "method", "reference"],
        ((f, *pair) for f, pair in product(families, pairs)),
    )
    vals = scores.test_brier.unstack().reindex(
        pd.MultiIndex.from_product((families, seeds))
    )
    indices = np.random.default_rng(97828).integers(4, size=(10_000, 25, 4))
    for r in contrasts.itertuples():
        delta = (vals[r.method] - vals[r.reference]).to_numpy().reshape(25, 4)
        means = delta.mean(axis=1)
        report = by.xs((r.method, r.reference), level=("method", "reference")).loc[
            list(families)
        ]
        close(means, report.estimate, ("family means", r.method, r.reference))
        for v, k in (
            (delta < -1e-12, "n_seed_better"),
            (abs(delta) <= 1e-12, "n_seed_tied"),
            (delta > 1e-12, "n_seed_worse"),
        ):
            close(v.sum(axis=1), report[k], k)
        resampled = delta[np.arange(25)[None, :, None], indices].mean(axis=(1, 2))
        interval = [
            float(means.mean()),
            *map(float, np.quantile(resampled, [0.025, 0.975])),
        ]
        close(interval, [r.estimate, r.ci_low, r.ci_high], "within-family resampling")
        signs = [
            int((means < -1e-12).sum()),
            int((abs(means) <= 1e-12).sum()),
            int((means > 1e-12).sum()),
        ]
        require(signs == [r.n_better, r.n_tied, r.n_worse], "fixed-family signs")
        if (r.method, r.reference) == ("joint_joint", "fold_joint"):
            primary = {
                **dict(zip(CI, interval, strict=True)),
                "better": signs[0],
                "tied": signs[1],
                "worse": signs[2],
                "broad_gain": interval[2] < 0 and signs[0] > signs[2],
            }
    return {
        "families": len(families),
        "banks_per_family": len(seeds),
        "scores": len(scores),
        "trajectory_rows": len(trace),
        "independent_mean_top_n_banks": len(families) * len(seeds),
        "intervals_recomputed": len(contrasts),
        "primary": primary,
    }


def main():
    with threadpool_limits(limits=1):
        result = {
            "status": "passed",
            "terminal_boundaries": corners(),
            "round_1": first_round(),
            "round_2": second_round(),
            "round_3_controlled": controlled(),
            "round_3_panel": panel(),
        }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (OUT / "verification.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
