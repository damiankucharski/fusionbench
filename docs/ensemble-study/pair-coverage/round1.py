"""Score two fixed-cost pair menus against the same unmodified GP finalist."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import predict_tree
from fusionbench.scenarios.native_switch import draw_native_switch

HERE = Path(__file__).resolve().parent
PRIOR = run_path(str(HERE.parent / "joint-gain/round2.py"))
ENGINE = PRIOR["ENGINE"]
SEEDED = range(241100, 241132)
POLICIES = ("iid", "unique")
METHODS = ("gp", *POLICIES, "mean", "top_n", "prob_lr", "regularized")


def brier(pred, labels):
    return float(np.square(pred - np.eye(2)[labels]).sum(axis=1).mean())


def fit_menu(x, y, base_tree, base_risk, schedule):
    from okapi.node import ValueNode
    from okapi.pair_coverage import pair_tree

    roots = [ValueNode(None, x[:, i, 1:2], f"m{i}.npy") for i in range(x.shape[1])]
    chosen, chosen_slot, winner_risk = base_tree, -1, base_risk
    records = []
    for slot, proposal in enumerate(schedule):
        tree = pair_tree(roots, proposal)
        fit_risk = brier(predict_tree(tree, x), y)
        _, i, j = proposal
        records.append(
            {
                "slot": slot,
                "left": i,
                "right": j,
                "fit_brier": fit_risk,
                "program_digest": ENGINE["digest"](tree.structural_key()),
            }
        )
        if fit_risk < winner_risk:
            chosen, chosen_slot, winner_risk = tree, slot, fit_risk
    return chosen, chosen_slot, winner_risk, records


def bank(seed):
    from okapi.pair_coverage import pair_schedule

    start = perf_counter()
    fit, audit = (
        draw_native_switch(seed, split, n) for split, n in ((1, 192), (2, 4096))
    )
    true_pair = tuple(sorted((fit.parent, fit.child)))
    schedules = {
        p: pair_schedule(6, 8, seed=seed + 4_000_000, policy=p, operators=("min",))
        for p in POLICIES
    }
    scores, proposals, banks = [], [], []
    for law in ("min", "singleton"):
        x = fit.preds
        q, labels = PRIOR["target"](fit, law)
        aq, ay = PRIOR["target"](audit, law)
        cell = {"seed": seed, "law": law}
        search = ENGINE["fit_search"](
            x,
            labels,
            groups=np.arange(len(labels)),
            data_seed=seed,
            search_seed=seed + 1_000_000,
            path="pooled",
            joint_study=True,
        )
        key = search["choices"]["pooled_final"]
        base = search["archive"][key]
        finalist = base["tree"]
        extensions = {
            p: fit_menu(x, labels, finalist, base["pooled_brier"], schedules[p])
            for p in POLICIES
        }

        # All live choices are now frozen. Hidden roles and audit may be used for scoring.
        search["choices"] = {"gp": key}
        (gp,), _ = ENGINE["score_search"](search, audit.preds, ay, audit_q=aq, fit_q=q)
        base_scores = {
            "fit_brier": gp["fit_brier"],
            "audit_brier": gp["audit_brier"],
            "audit_excess": gp["audit_excess"],
            "exact_selected": gp["selected_is_exact"],
            "program_digest": gp["selected_digest"],
            "n_providers": gp["unique_providers"],
        }
        scores.append(
            {
                **cell,
                "method": "gp",
                "choice_slot": -1,
                "extra_fit_calls": 0,
                "menu_has_exact": gp["selected_is_exact"],
                **base_scores,
            }
        )
        banks.append(
            {
                **cell,
                "role_a": fit.parent,
                "role_b": fit.child,
                **{
                    k: gp[k]
                    for k in (
                        "objective_evaluations",
                        "offspring_proposals",
                        "path_digest",
                        "unique_evaluated",
                        "exact_ever_visited",
                        "exact_in_final",
                        "selected_is_exact",
                        "best_ever_audit_excess",
                        "best_final_audit_excess",
                    )
                },
                "fit_x_sha256": ENGINE["array_digest"](x),
                "audit_x_sha256": ENGINE["array_digest"](audit.preds),
            }
        )
        for policy, (tree, slot, loss, records) in extensions.items():
            offered = False
            for row in records:
                hit = (row["left"], row["right"]) == true_pair
                offered |= hit
                proposals.append({**cell, "policy": policy, "is_true_pair": hit, **row})
            ap = predict_tree(tree, audit.preds)
            fp = predict_tree(tree, x)
            assert brier(fp, labels) == loss
            exact = (
                np.max(abs(fp[:, 1] - q)) <= 1e-12
                and np.max(abs(ap[:, 1] - aq)) <= 1e-12
            )
            scores.append(
                {
                    **cell,
                    "method": policy,
                    "choice_slot": slot,
                    "extra_fit_calls": len(records),
                    "n_distinct_probes": len(set(schedules[policy])),
                    "true_pair_offered": offered,
                    "menu_has_exact": gp["selected_is_exact"]
                    or (law == "min" and offered),
                    "exact_selected": bool(exact),
                    "fit_brier": loss,
                    "audit_brier": brier(ap, ay),
                    "audit_excess": float(
                        np.square(ap - np.column_stack((1 - aq, aq))).sum(axis=1).mean()
                    ),
                    "program_digest": ENGINE["digest"](tree.structural_key()),
                    "n_providers": len(tree.unique_value_node_ids),
                }
            )

        refs = PRIOR["BASELINES"](
            x, labels, audit.preds, ay, seed=seed + 1_000_000, audit_q=aq
        )
        scores.extend({**cell, "method": name, **row} for name, row in refs.items())
    return scores, proposals, banks, perf_counter() - start


def summarize(scores, probes, banks):
    assert (
        len(scores) == 32 * 2 * len(METHODS)
        and len(banks) == 32 * 2
        and len(probes) == 32 * 2 * 2 * 8
    )
    assert not scores.duplicated(["seed", "law", "method"]).any()
    assert not probes.duplicated(["seed", "law", "policy", "slot"]).any()
    for (seed, law), part in scores.groupby(["seed", "law"]):
        part = part.set_index("method")
        assert set(part.index) == set(METHODS)
        for policy in POLICIES:
            menu = probes.query(
                "seed==@seed and law==@law and policy==@policy"
            ).sort_values("slot")
            costs = [part.loc["gp", "fit_brier"], *menu.fit_brier]
            winner = int(np.argmin(costs)) - 1
            assert len(menu) == 8 and int(part.loc[policy, "choice_slot"]) == winner
            assert part.loc[policy, "fit_brier"] == min(costs)
            assert (
                not part.loc[policy, "exact_selected"]
                or part.loc[policy, "menu_has_exact"]
            )
        assert part.loc["unique", "n_distinct_probes"] == 8
    draws = np.random.default_rng(97901).integers(32, size=(10000, 32))
    records = []
    for law, frame in scores.groupby("law"):
        for metric in ("fit_brier", "audit_brier", "audit_excess"):
            wide = frame.pivot(index="seed", columns="method", values=metric).loc[
                list(SEEDED)
            ]
            for method in METHODS:
                for reference in ("level", "gp", "iid"):
                    v = (
                        wide[method].to_numpy()
                        if reference == "level"
                        else (wide[method] - wide[reference]).to_numpy()
                    )
                    lo, hi = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
                    records.append(
                        {
                            "law": law,
                            "method": method,
                            "reference": reference,
                            "metric": metric,
                            "estimate": v.mean(),
                            "ci_low": lo,
                            "ci_high": hi,
                            "negative": (v < -1e-12).sum(),
                            "positive": (v > 1e-12).sum(),
                        }
                    )
    return pd.DataFrame(records)


def main():
    from okapi import pair_coverage

    beginning = datetime.now(UTC).isoformat()
    clock = perf_counter()
    with threadpool_limits(limits=1):
        batches = [bank(seed) for seed in SEEDED]
    scores, probes, banks = [
        pd.DataFrame(row for batch in batches for row in batch[i]) for i in range(3)
    ]
    summary = summarize(scores, probes, banks)
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("scores", scores),
        ("probes", probes),
        ("banks", banks),
        ("summary", summary),
    ):
        table.to_csv(out / f"r1-{name}.csv", index=False)
    sources = [
        Path(__file__),
        HERE / "r1-protocol.md",
        HERE.parent / "fold-promotion/search.py",
        Path(pair_coverage.__file__),
    ]
    info = {
        "started_utc": beginning,
        "seconds": perf_counter() - clock,
        "bank_seconds": [b[3] for b in batches],
        "workers": 1,
        "numerical_threads": 1,
        "native_searches": len(banks),
        "probe_fit_calls": len(probes),
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (out / "r1-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"Finished {len(banks)} searches, {len(probes)} fit probes in {info['seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
