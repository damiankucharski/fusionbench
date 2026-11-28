"""Independent numerical verification of the three fixed terminal-probe comparisons."""

import json
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PREVIOUS = HERE.parent / "fold-promotion"
audit = run_path(str(PREVIOUS / "verify.py"))
close, require, read, grid, sha, key_hash, bootstrap = (
    audit[k]
    for k in ("close", "require", "read", "grid", "sha", "key_hash", "bootstrap")
)
REFS = ("mean", "top_n", "prob_lr", "regularized")
CI = ["estimate", "ci_low", "ci_high"]


@dataclass
class Bank:
    x: np.ndarray
    y: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    fq: np.ndarray | None = None
    aq: np.ndarray | None = None
    roles: tuple | None = None

    @property
    def n_classes(self):
        return self.x.shape[2]


def binary_data(series, seed, law, m):
    if series == 1:
        a, b = np.random.default_rng([seed, 15]).permutation(m)[:2]
    else:
        order = np.random.default_rng([seed, m, 15]).permutation(m)
        a, b = np.argsort(order)[:2]

    def split(index, n):
        if series == 1:
            draw = np.random.default_rng([seed, index])
            p = draw.integers(1, 64, (n, m)) / 64
            uniform = draw.random(n)
        else:
            p = (
                np.random.default_rng([seed, index, 0]).integers(1, 64, (n, 12))[
                    :, order
                ]
                / 64
            )
            uniform = np.random.default_rng([seed, index, 1]).random(n)
        q = (
            p[:, a]
            if law == "singleton"
            else (np.minimum if law == "min" else np.maximum)(p[:, a], p[:, b])
        )
        return np.stack((1 - p, p), axis=-1), (uniform < q).astype(int), q

    x, y, fq = split(1, 192)
    ax, ay, aq = split(2, 4096)
    return Bank(x, y, ax, ay, fq, aq, tuple(sorted((int(a), int(b)))))


def schedule(m, n, op, seed, unique):
    palette = list(combinations(range(m), 2))
    rng = np.random.default_rng(seed)
    if unique:
        draws = []
        while len(draws) < n:
            draws.extend(rng.permutation(len(palette)))
    else:
        draws = rng.integers(len(palette), size=n)
    return [(op, *palette[i]) for i in draws[:n]]


def menus(series, seed, m):
    if series == 1:
        return {
            p: schedule(m, 8, "min", seed + 4_000_000, p == "unique")
            for p in ("iid", "unique")
        }
    if series == 2:

        def arm(op, k):
            s = int(np.random.SeedSequence([seed, 4_000_000, k]).generate_state(1)[0])
            return schedule(m, 8, op, s, True)

        mn, mx = arm("min", 0), arm("max", 1)
        return {
            "min8": mn,
            "balanced8": [k for t in zip(mn[:4], mx[:4], strict=True) for k in t],
            "max8": mx,
        }
    result = {}
    for policy in ("iid", "unique"):
        ops = []
        for k, op in enumerate(("min", "max")):
            s = int(np.random.SeedSequence([seed, k, 6103]).generate_state(1)[0])
            ops.append(schedule(m, 4, op, s, policy == "unique"))
        result[policy] = [k for t in zip(*ops, strict=True) for k in t]
    return result


def full(p):
    return np.column_stack((1.0 - p, p))


def normalized(p):
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=1, keepdims=True)


def probe(x, instruction):
    op, a, b = instruction
    value = (np.minimum if op == "min" else np.maximum)(x[:, a], x[:, b])
    if x.shape[-1] == 2:
        return full(value[:, 1])
    mass = value.sum(axis=1, keepdims=True)
    value = value + (mass == 0)
    return normalized(value / value.sum(axis=1, keepdims=True))


def risk(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean())


def excess(p, q):
    return float(2 * np.mean((p[:, 1] - q) ** 2))


def interval(delta, seed):
    return dict(zip(CI, bootstrap(delta, seed), strict=True))


def refs(bank, score, seed, split):
    """Fit all four references. Independently implement the first three."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x, y, ax = bank.x, bank.y, bank.ax
    rank = sorted(range(x.shape[1]), key=lambda j: risk(normalized(x[:, j]), y))
    k = min(
        range(1, len(rank) + 1),
        key=lambda k: risk(normalized(x[:, rank[:k]].mean(1)), y),
    )
    scaler = StandardScaler().fit(x.reshape(len(x), -1))

    def features(z):
        return scaler.transform(z.reshape(len(z), -1))

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(features(x), y)
    native = RegularizedNativePatternOKAPI(seed=seed, split_seed=seed)
    native.fit(
        x,
        y,
        task="binary" if bank.n_classes == 2 else "multiclass",
        n_classes=bank.n_classes,
    )

    def predictions(z):
        p = np.zeros((len(z), bank.n_classes))
        p[:, lr.classes_] = lr.predict_proba(features(z))
        return {
            "mean": normalized(z.mean(1)),
            "top_n": normalized(z[:, rank[:k]].mean(1)),
            "prob_lr": normalized(p),
            "regularized": native.predict(z),
        }

    fits, tests = predictions(x), predictions(ax)
    for name in REFS:
        row = score.loc[name]
        close(
            [risk(fits[name], y), risk(tests[name], bank.ay)],
            [row.fit_brier, row[split + "_brier"]],
            ("reference", name),
        )
        if bank.aq is not None:
            close(
                excess(tests[name], bank.aq),
                row.audit_excess,
                ("reference excess", name),
            )
        if (
            name == "regularized"
            and "predictions_sha256" in row
            and pd.notna(row.predictions_sha256)
        ):
            require(
                sha(tests[name]) == row.predictions_sha256,
                "regularized predictions bytes",
            )


def regrow(bank, seed, search_seed):
    fit = run_path(str(PREVIOUS / "search.py"))["fit_search"](
        bank.x,
        bank.y,
        groups=np.arange(len(bank.y)),
        data_seed=seed,
        search_seed=search_seed,
        path="pooled",
    )
    archive = fit["archive"]
    final = fit["snapshots"][-1]["keys"]
    by = {}
    for k in final:
        t = archive[k]["tree"]
        by[k] = risk(predict_tree(t, bank.x), bank.y), t.nodes_count
    key = min(final, key=by.__getitem__)
    require(key == fit["choices"]["pooled_final"], "unrounded fit-only GP incumbent")
    tree = archive[key]["tree"]
    return key_hash(key), predict_tree(tree, bank.x), predict_tree(tree, bank.ax), fit


def schedule_boundaries():
    from okapi.node import ValueNode
    from okapi.pair_coverage import pair_schedule, pair_tree

    count = 0
    for m, n, op, unique in product(
        (2, 3, 6), (1, 4, 8), ("min", "max"), (True, False)
    ):
        expected = schedule(m, n, op, 1729, unique)
        actual = pair_schedule(
            m, n, operators=(op,), policy="unique" if unique else "iid", seed=1729
        )
        require(
            list(actual) == expected, "finite schedule including recycling at M=2/3"
        )
        count += 1
    x = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.5, 0.25, 0.25], [0.25, 0.75, 0.0]]]
    )
    roots = [ValueNode(None, x[:, i], f"m{i}.npy") for i in range(2)]
    for op in ("min", "max"):
        instr = (op, 0, 1)
        tree = pair_tree(roots, instr)
        close(predict_tree(tree, x), probe(x, instr), "multiclass pair normalization")
    close(
        probe(x, ("min", 0, 1))[0], np.repeat(1 / 3, 3), "disjoint-support MIN fallback"
    )
    return {"schedule_cases": count, "multiclass_disjoint_support": True}


def structure(instr):
    from okapi.node import MaxNode, MinNode, ValueNode
    from okapi.tree import Tree

    op, a, b = instr
    node = (MinNode if op == "min" else MaxNode)([ValueNode(None, None, f"m{b}.npy")])
    return key_hash(Tree(ValueNode([node], None, f"m{a}.npy")).structural_key())


def check_menus(bank, seed, series, rows, offered, gp_meta):
    """Use independently drawn slots, fresh GP predictions and no held-out tie break."""
    plan = menus(series, seed, bank.x.shape[1])
    h, gfit, gtest, gp = regrow(bank, seed, 0 if series == 3 else seed + 1_000_000)
    incumbent = rows.loc["gp"]
    require(
        incumbent.digest == h and incumbent.calls == 0 and incumbent.chosen == -1,
        "unmodified frozen incumbent",
    )
    for k in ("path_digest", "objective_evaluations", "offspring_proposals"):
        require(gp_meta[k] == gp["snapshots"][-1][k], ("regrown GP", k))
    close(
        [risk(gfit, bank.y), risk(gtest, bank.ay)],
        [incumbent.fit_brier, incumbent.audit_brier],
        "replayed GP scores",
    )
    if bank.aq is not None:
        close(excess(gtest, bank.aq), incumbent.audit_excess, "GP excess")

        def exact(fp, ap):
            return (
                np.max(abs(fp[:, 1] - bank.fq)) <= 1e-12
                and np.max(abs(ap[:, 1] - bank.aq)) <= 1e-12
            )

        archive = gp["archive"]
        exact_set = set()
        for k, r in archive.items():
            p = predict_tree(r["tree"], bank.x)
            if np.max(abs(p[:, 1] - bank.fq)) <= 1e-12 and exact(
                p, predict_tree(r["tree"], bank.ax)
            ):
                exact_set.add(k)
        require(
            (bool(exact_set), bool(exact_set.intersection(gp["snapshots"][-1]["keys"])))
            == (gp_meta["ever"], gp_meta["final_exact"]),
            "GP functional coverage before proposals",
        )
    checked_ties = 0
    for policy, specs in plan.items():
        menu = offered.loc[policy]
        row = rows.loc[policy]
        require(
            len(menu) == row.calls == 8 and list(menu.index) == list(range(8)),
            "eight actual scored slots",
        )
        if "distinct" in row:
            require(len(set(specs)) == row.distinct, "unique typed pairs per menu")
        fit = [gfit]
        test = [gtest]
        ids = [h]
        for index, (op, a, b) in enumerate(specs):
            actual = menu.loc[index]
            require(
                (actual.operator, actual.left, actual.right) == (op, a, b),
                "exact generated schedule in public provider order",
            )
            p, t = probe(bank.x, (op, a, b)), probe(bank.ax, (op, a, b))
            fit.append(p)
            test.append(t)
            ids.append(structure((op, a, b)))
            require(
                actual.digest == ids[-1], "fresh native structural fingerprint of pair"
            )
            close(
                risk(p, bank.y),
                actual.fit_brier,
                "independently reconstructed probe fit score",
            )
            if "audit_brier" in actual:
                close(risk(t, bank.ay), actual.audit_brier, "probe audit risk")
            if "audit_excess" in actual:
                close(excess(t, bank.aq), actual.audit_excess, "probe conditional risk")
                require(
                    exact(p, t) == actual.is_exact, "functional identity of MIN/MAX"
                )
            if "predictions_sha256" in actual:
                require(
                    sha(t) == actual.predictions_sha256, "pair audit prediction bytes"
                )
            if "pair_hit" in actual:
                target_op = "min" if series == 1 else "max"
                require(
                    actual.pair_hit == ((a, b) == bank.roles and op == target_op),
                    "actual coverage of hidden pair",
                )
        costs = [risk(p, bank.y) for p in fit]
        winner = min(range(9), key=costs.__getitem__)
        checked_ties += int(costs.count(costs[winner]) > 1)
        require(
            row.chosen == winner - 1 and row.digest == ids[winner],
            ("fit-only rank and structural choice", series, seed, policy),
        )
        if "selected" in menu:
            require(
                menu.selected.tolist() == [i == winner - 1 for i in range(8)],
                "selected marker only on deployed pair",
            )
        close(
            [row.fit_brier, row.audit_brier],
            [costs[winner], risk(test[winner], bank.ay)],
            "frozen selected risk",
        )
        if "predictions_sha256" in row:
            require(
                sha(test[winner]) == row.predictions_sha256,
                "selected held-out prediction bytes",
            )
        if bank.aq is not None:
            close(
                row.audit_excess,
                excess(test[winner], bank.aq),
                "conditional deployment risk",
            )
            require(
                row.exact == exact(fit[winner], test[winner]),
                "selected observed exact rule",
            )
            has_probe = any(exact(p, t) for p, t in zip(fit[1:], test[1:], strict=True))
            require(
                row.menu_exact == (has_probe or exact(gfit, gtest)),
                "availability after discovery",
            )
        require(
            row.n_used == (incumbent.n_used if winner == 0 else 2),
            "number of deployed providers",
        )
    return checked_ties


def primary_row(frame, **select):
    for name, value in select.items():
        frame = frame.loc[frame[name] == value]
    require(len(frame) == 1, ("primary row", select))
    return frame.iloc[0]


def first_round():
    seeds = range(241100, 241132)
    laws = ("min", "singleton")
    all_methods = ("gp", "iid", "unique", *REFS)
    scores = grid(
        OUT / "r1-scores.csv",
        ["seed", "law", "method"],
        product(seeds, laws, all_methods),
    ).rename(
        columns={
            "choice_slot": "chosen",
            "extra_fit_calls": "calls",
            "program_digest": "digest",
            "n_distinct_probes": "distinct",
            "exact_selected": "exact",
            "menu_has_exact": "menu_exact",
            "n_providers": "n_used",
        }
    )
    banks = grid(OUT / "r1-banks.csv", ["seed", "law"], product(seeds, laws)).rename(
        columns={"exact_ever_visited": "ever", "exact_in_final": "final_exact"}
    )
    offers = grid(
        OUT / "r1-probes.csv",
        ["seed", "law", "policy", "slot"],
        product(seeds, laws, ("iid", "unique"), range(8)),
    ).rename(columns={"program_digest": "digest", "is_true_pair": "pair_hit"})
    offers["operator"] = "min"
    ties = 0
    for seed, law in product(seeds, laws):
        bank = binary_data(1, seed, law, 6)
        m = banks.loc[(seed, law)]
        require(
            (sha(bank.x), sha(bank.ax)) == (m.fit_x_sha256, m.audit_x_sha256)
            and bank.roles == tuple(sorted((m.role_a, m.role_b))),
            "original six anonymous grid columns",
        )
        cell = scores.loc[(seed, law)]
        ties += check_menus(bank, seed, 1, cell, offers.loc[(seed, law)], m)
        refs(bank, cell, seed + 1_000_000, "audit")
    v = scores.xs("min", level="law").audit_excess.unstack()
    delta = v.unique - v.iid
    primary = interval(delta, 97901)
    reported = primary_row(
        read(OUT / "r1-summary.csv"),
        law="min",
        method="unique",
        reference="iid",
        metric="audit_excess",
    )
    close(list(primary.values()), reported[CI], "R1 primary paired banks")
    require(
        reported.negative == (delta < -1e-12).sum()
        and reported.positive == (delta > 1e-12).sum(),
        "R1 bank directions",
    )
    return {
        "independent_banks": len(seeds),
        "regrown_gp_cells": len(banks),
        "reference_refits": 4 * len(banks),
        "probe_calls": len(offers),
        "tied_winning_menus": ties,
        "primary": primary,
    }


def second_round():
    seeds = range(242100, 242124)
    sizes = (6, 12)
    laws = ("max", "singleton")
    names = ("gp", "min8", "balanced8", "max8", *REFS)
    cells = list(product(seeds, sizes, laws))
    keys = ["seed", "n_providers", "law"]
    scores = grid(
        OUT / "r2-scores.csv",
        [*keys, "method"],
        ((*c, m) for c in cells for m in names),
    ).rename(
        columns={
            "selected_slot": "chosen",
            "probe_fit_calls": "calls",
            "selected_digest": "digest",
            "exact_in_menu": "menu_exact",
            "selected_exact": "exact",
        }
    )
    banks = grid(OUT / "r2-banks.csv", keys, cells).rename(
        columns={
            "gp_objective_evaluations": "objective_evaluations",
            "gp_offspring_proposals": "offspring_proposals",
            "gp_exact_ever": "ever",
            "gp_exact_in_final": "final_exact",
        }
    )
    raw = grid(
        OUT / "r2-candidates.csv.gz",
        [*keys, "method", "slot"],
        (
            (*c, m, k)
            for c in cells
            for m in names[:4]
            for k in ([-1] if m == "gp" else range(-1, 8))
        ),
    ).rename(columns={"program_digest": "digest", "planted_max_probe": "pair_hit"})
    offers = raw.loc[raw.index.get_level_values("slot") >= 0]
    ties = 0
    for seed, m, law in cells:
        idx = (seed, m, law)
        bank = binary_data(2, seed, law, m)
        diag = banks.loc[idx]
        require(
            bank.roles == tuple(sorted((diag.role_a, diag.role_b))),
            "public permutation of planted latent roles",
        )
        for key, arr in (
            ("fit_x", bank.x),
            ("fit_y", bank.y),
            ("fit_q", bank.fq),
            ("audit_x", bank.ax),
            ("audit_y", bank.ay),
            ("audit_q", bank.aq),
        ):
            require(sha(arr) == diag[key + "_sha256"], ("R2 deterministic source", key))
        sub = scores.loc[idx]
        shared = raw.loc[idx].xs(-1, level="slot")
        require(
            shared.digest.nunique() == 1
            and shared.iloc[0].digest == sub.loc["gp", "digest"],
            "paired exact same GP incumbent",
        )
        n = sub.loc[list(names[:4])]
        require(
            (n.gp_objective_evaluations == diag.objective_evaluations).all()
            and (n.total_fit_calls == n.calls + diag.objective_evaluations).all(),
            "native plus matched probe fit calls",
        )
        ties += check_menus(bank, seed, 2, sub, offers.loc[idx], diag)
        refs(bank, sub, seed + 1_000_000, "audit")
    v = scores.xs("max", level="law").audit_excess.unstack()
    delta = (v.balanced8 - v.min8).groupby(level="seed").mean()
    primary = interval(delta, 97832)
    reported = primary_row(
        read(OUT / "r2-contrasts.csv"),
        method="balanced8",
        reference="min8",
        metric="audit_excess",
        law="max",
        n_providers="both",
    )
    close(list(primary.values()), reported[CI], "R2 paired whole-bank interval")
    return {
        "independent_banks": len(seeds),
        "sizes": list(sizes),
        "regrown_gp_cells": len(banks),
        "reference_refits": 4 * len(banks),
        "probe_calls": len(offers),
        "tied_winning_menus": ties,
        "primary": primary,
    }


def third_round():
    families = tuple(load_preset("standard-v4")["scenarios"])
    seeds = range(243100, 243104)
    require(len(set(families)) == len(families) == 25, "complete immutable standard-v4")
    keys = ["scenario_id", "data_seed"]
    scores = grid(
        OUT / "r3-scores.csv",
        [*keys, "method"],
        product(families, seeds, ("gp", "iid", "unique", *REFS)),
    ).rename(
        columns={
            "selected_slot": "chosen",
            "probe_evaluations": "calls",
            "selected_digest": "digest",
            "test_brier": "audit_brier",
            "unique_providers": "n_used",
            "n_distinct_probes": "distinct",
        }
    )
    scores["chosen"] -= 1
    raw = (
        grid(
            OUT / "r3-proposals.csv",
            [*keys, "policy", "slot"],
            product(families, seeds, ("iid", "unique"), range(1, 9)),
        )
        .reset_index()
        .rename(
            columns={
                "test_brier": "audit_brier",
                "selected_digest": "digest",
                "provider_left": "left",
                "provider_right": "right",
            }
        )
    )
    raw["slot"] -= 1
    raw = raw.set_index([*keys, "policy", "slot"]).sort_index()
    ties = 0
    for family, seed in product(families, seeds):
        data = generate_dataset(family, n_per_split=512, seed=seed)
        bank = Bank(data.fit_preds, data.fit_y, data.test_preds, data.test_y)
        cell = scores.loc[(family, seed)]
        for k, v in (
            ("fit_inputs", bank.x),
            ("fit_labels", bank.y),
            ("test_inputs", bank.ax),
            ("test_labels", bank.ay),
        ):
            require(cell[k + "_sha256"].eq(sha(v)).all(), ("panel source", family, k))
        native = cell.loc[["gp", "iid", "unique"]]
        meta = {}
        for k in ("path_digest", "objective_evaluations", "offspring_proposals"):
            s = native["gp_" + k]
            require(s.nunique() == 1, "paired identical ordinary GP")
            meta[k] = s.iloc[0]
        ties += check_menus(bank, seed, 3, cell, raw.loc[(family, seed)], meta)
        refs(bank, cell, 0, "audit")
    levels = scores.audit_brier.unstack().reindex(
        pd.MultiIndex.from_product((families, seeds))
    )
    draws = np.random.default_rng(97831).integers(4, size=(10_000, 25, 4))
    pairs = [("unique", ref) for ref in ("gp", "iid", *REFS)] + [("iid", "gp")]
    overall = grid(OUT / "r3-contrasts.csv", ["method", "reference"], pairs)
    grouped = grid(
        OUT / "r3-family-contrasts.csv",
        ["scenario_id", "method", "reference"],
        ((f, *p) for f, p in product(families, pairs)),
    )
    for name, ref in pairs:
        gaps = (levels[name] - levels[ref]).to_numpy().reshape(25, 4)
        means = gaps.mean(axis=1)
        by_boot = gaps[np.arange(25)[None, :, None], draws].mean(axis=(1, 2))
        ci = [float(means.mean()), *map(float, np.quantile(by_boot, [0.025, 0.975]))]
        reported = overall.loc[(name, ref)]
        close(ci, reported[CI], ("fixed-family paired bootstrap", name, ref))
        signs = [
            int((means < -1e-12).sum()),
            int((abs(means) <= 1e-12).sum()),
            int((means > 1e-12).sum()),
        ]
        close(signs, reported[["n_better", "n_tied", "n_worse"]], "family directions")
        per_family = grouped.xs((name, ref), level=("method", "reference")).loc[
            list(families)
        ]
        close(means, per_family.estimate, "individual family deltas")
        if (name, ref) == ("unique", "iid"):
            primary = {
                **dict(zip(CI, ci, strict=True)),
                "better": signs[0],
                "tied": signs[1],
                "worse": signs[2],
            }
    return {
        "fixed_families": 25,
        "banks_per_family": 4,
        "regrown_gp_cells": 100,
        "reference_refits": 400,
        "probe_calls": len(raw),
        "tied_winning_menus": ties,
        "paired_intervals": len(pairs),
        "primary": primary,
    }


def charged_ties():
    """Count real prediction calls including duplicates, restricting selectors to fit views."""
    from unittest.mock import patch

    from okapi.node import ValueNode
    from okapi.tree import Tree

    repeated = np.tile([0.75, 0.25, 0.25, 0.25, 0.25, 0.25], (16, 1))
    x = np.stack((1.0 - repeated, repeated), axis=2)
    y = np.zeros(len(x), dtype=int)
    root = Tree(ValueNode(None, x[:, 0, 1:2], "m0.npy"))
    repeated_menu = [("min", 0, 1), ("min", 0, 2)] * 4
    first = run_path(str(HERE / "round1.py"))["fit_menu"]
    second = run_path(str(HERE / "r2.py"))["fit_menus"]
    third = run_path(str(HERE / "r3_panel.py"))["fit_menus"]

    def guard(fn, args, expected):
        calls = []

        def counted(t, arr):
            require(arr is x, "scoring received only this fit input array")
            calls.append(t.structural_key())
            return predict_tree(t, arr)

        with patch.dict(fn.__globals__, {"predict_tree": counted}):
            answer = fn(*args)
        require(len(calls) == expected, "full forward costs on repeated pairs")
        return answer

    original = risk(full(x[:, 0, 1]), y)
    _, slot, _, _ = guard(first, (x, y, root, original, repeated_menu), 8)
    require(slot == 0, "first tied strictly superior probe")
    # The second identically scoring fit candidate is much better on a separate
    # audit whose labels favor small values. It must still lose the fit-only tie.
    ax = x.copy()
    ax[:, 1, :] = [0.2, 0.8]
    ax[:, 2, :] = [0.95, 0.05]
    require(
        risk(probe(ax, repeated_menu[1]), y) < risk(probe(ax, repeated_menu[0]), y),
        "audit actively prefers later probe",
    )
    exact = Tree(ValueNode(None, x[:, 1, 1:2], "m1.npy"))
    require(
        guard(first, (x, y, exact, risk(full(x[:, 1, 1]), y), repeated_menu), 8)[1]
        == -1,
        "R1 incumbent tie",
    )
    _, _, sel = guard(second, (x, y, 919), 24)
    require(set(sel.values()) == {0}, "R2 incumbent retains ties")
    _, choices, _ = guard(third, (root, x, y, 919), 17)
    require(set(choices.values()) == {1}, "R3 first fit-optimal probe")
    _, choices, _ = guard(third, (exact, x, y, 919), 17)
    require(set(choices.values()) == {0}, "R3 incumbent retains ties")
    return {
        "repeated_pairs_charged": True,
        "fit_forward_counts": {
            "r1_menu": 8,
            "r2_three_menus": 24,
            "r3_incumbent_and_two_menus": 17,
        },
        "incumbent_ties": True,
        "first_probe_ties": True,
        "audit_preferred_later_probe": True,
    }


def reported_bank_differences():
    """Check each published R2 primary observation before its bank bootstrap."""
    frame = read(OUT / "r2-scores.csv")
    risk_table = frame.loc[frame.law.eq("max")].pivot(
        index=["seed", "n_providers"], columns="method", values="audit_excess"
    )
    size_deltas = (risk_table.balanced8 - risk_table.min8).unstack()
    observed = grid(
        OUT / "r2-primary-by-bank.csv",
        ["seed", "n_providers"],
        product(range(242100, 242124), ("6", "12", "both")),
    )
    for size in (6, 12):
        close(
            size_deltas[size],
            observed.xs(str(size), level="n_providers").difference,
            "published size-specific paired bank",
        )
    close(
        size_deltas.mean(axis=1),
        observed.xs("both", level="n_providers").difference,
        "published 24 independent resampling units",
    )
    return {"paired_differences": len(observed), "whole_bank_units": len(size_deltas)}


def main():
    with threadpool_limits(limits=1):
        result = {
            "r2_bank_export": reported_bank_differences(),
            "schedule": schedule_boundaries(),
            "selection": charged_ties(),
            "round_1": first_round(),
            "round_2": second_round(),
            "round_3": third_round(),
            "status": "passed",
        }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (OUT / "verification.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
