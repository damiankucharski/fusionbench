"""Run the frozen terminal pair-coverage experiment on 25 standard FusionBench families."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.eval.transfer_grid import brier
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
old = run_path(str(HERE.parent / "joint-gain" / "round3_panel.py"))
references, digest = old["references"], old["digest"]
from search import fit_search

SEEDS = (243100, 243101, 243102, 243103)
METHODS = ("gp", "iid", "unique", "mean", "top_n", "prob_lr", "regularized")
POLICIES = ("iid", "unique")
BOOTSTRAP_SEED = 97831
TOL = 1e-12


def tree_digest(tree):
    return hashlib.sha256(repr(tree.structural_key()).encode()).hexdigest()


def schedule(n_providers, seed, policy):
    from okapi.pair_coverage import pair_schedule

    schedules = []
    for i, op in enumerate(("min", "max")):
        seed_op = int(np.random.SeedSequence([seed, i, 6103]).generate_state(1)[0])
        batch = pair_schedule(
            n_providers, 4, seed=seed_op, policy=policy, operators=(op,)
        )
        schedules.append([(seed_op, proposal) for proposal in batch])
    return [item for pair in zip(*schedules, strict=True) for item in pair]


def fit_menus(incumbent, x, y, seed):
    from okapi.node import ValueNode
    from okapi.pair_coverage import pair_tree

    roots = [ValueNode(None, value, provider) for provider, value in _bind(x).items()]
    base_risk = brier(predict_tree(incumbent, x), y)
    choices, proposals = {}, {}
    for policy in POLICIES:
        menu = []
        for slot, (seed_op, proposal) in enumerate(
            schedule(len(roots), seed, policy), start=1
        ):
            tree = pair_tree(roots, proposal)
            op, left, right = proposal
            risk = brier(predict_tree(tree, x), y)
            menu.append(
                {
                    "policy": policy,
                    "slot": slot,
                    "operator": op,
                    "provider_left": left,
                    "provider_right": right,
                    "operator_seed": seed_op,
                    "tree": tree,
                    "fit_brier": risk,
                }
            )
        risks = [base_risk] + [item["fit_brier"] for item in menu]
        choices[policy] = int(np.argmin(risks))
        proposals[policy] = menu
    return base_risk, choices, proposals


def evaluate(family, seed, n_rows=512):
    bank = generate_dataset(family, seed=seed, n_per_split=n_rows)
    x, y = bank.fit_preds, bank.fit_y
    context = {
        "scenario_id": family,
        "data_seed": seed,
        "gp_seed": 0,
        "n_providers": x.shape[1],
        "n_classes": x.shape[2],
        "fit_inputs_sha256": digest(x),
        "fit_labels_sha256": digest(y),
        "test_inputs_sha256": digest(bank.test_preds),
        "test_labels_sha256": digest(bank.test_y),
    }
    start = perf_counter()
    fit = fit_search(
        x, y, groups=np.arange(len(y)), data_seed=seed, search_seed=0, path="pooled"
    )
    incumbent = fit["archive"][fit["choices"]["pooled_final"]]["tree"]
    gp_seconds = perf_counter() - start
    search_cost = fit["snapshots"][-1]
    base_risk, choices, menus = fit_menus(incumbent, x, y, seed)

    # Both deployment choices are now frozen; retrospective candidate scores are audit only.
    audit_x, audit_y = bank.test_preds, bank.test_y
    anchor = predict_tree(incumbent, audit_x)
    gp = {
        "fit_brier": base_risk,
        "test_brier": brier(anchor, audit_y),
        "predictions_sha256": digest(anchor),
        "selected_digest": tree_digest(incumbent),
        "unique_providers": len(incumbent.unique_value_node_ids),
        "selected_slot": 0,
        "selected_pair": False,
    }
    cost = {
        "gp_objective_evaluations": search_cost["objective_evaluations"],
        "gp_offspring_proposals": search_cost["offspring_proposals"],
        "gp_seconds": gp_seconds,
        "gp_path_digest": search_cost["path_digest"],
    }
    rows = [
        {
            **context,
            **gp,
            **cost,
            "method": "gp",
            "probe_evaluations": 0,
            "n_distinct_probes": 0,
        }
    ]
    raw = []
    for policy in POLICIES:
        winner = choices[policy]
        decision = gp.copy()
        distinct = set()
        for item in menus[policy]:
            tree = item["tree"]
            predicted = predict_tree(tree, audit_x)
            key = (item["operator"], item["provider_left"], item["provider_right"])
            distinct.add(key)
            assessment = {
                "test_brier": brier(predicted, audit_y),
                "predictions_sha256": digest(predicted),
                "selected_digest": tree_digest(tree),
            }
            raw.append(
                {
                    "scenario_id": family,
                    "data_seed": seed,
                    **{k: v for k, v in item.items() if k != "tree"},
                    **assessment,
                    "selected": item["slot"] == winner,
                }
            )
            if item["slot"] == winner:
                decision = {
                    **assessment,
                    "fit_brier": item["fit_brier"],
                    "selected_slot": winner,
                    "unique_providers": len(tree.unique_value_node_ids),
                    "selected_pair": True,
                }
        rows.append(
            {
                **context,
                **decision,
                **cost,
                "method": policy,
                "probe_evaluations": len(menus[policy]),
                "n_distinct_probes": len(distinct),
            }
        )
    rows.extend({**context, **row} for row in references(bank))
    return rows, raw


def summarize(scores, families):
    idx = pd.MultiIndex.from_product(
        [families, SEEDS], names=["scenario_id", "data_seed"]
    )
    data = scores.pivot(
        index=["scenario_id", "data_seed"], columns="method", values="test_brier"
    ).reindex(index=idx, columns=METHODS)
    if len(scores) != len(idx) * len(METHODS) or not np.isfinite(data).all().all():
        raise ValueError("Missing scenario × seed × method score")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(SEEDS), size=(10_000, len(families), len(SEEDS))
    )
    group = np.arange(len(families))[None, :, None]

    def estimate(series):
        x = series.to_numpy().reshape(len(families), len(SEEDS))
        low, high = np.quantile(x[group, draws].mean(axis=(1, 2)), [0.025, 0.975])
        return {
            "estimate": x.mean(),
            "ci_low": low,
            "ci_high": high,
            "n_families": len(families),
        }

    levels = [{"method": m, **estimate(data[m])} for m in METHODS]
    contrasts, by_family = [], []
    for method, ref in [("unique", r) for r in METHODS if r != "unique"] + [
        ("iid", "gp")
    ]:
        delta = data[method] - data[ref]
        family_means = delta.groupby(level="scenario_id").mean()
        contrasts.append(
            {
                "method": method,
                "reference": ref,
                **estimate(delta),
                "n_better": int((family_means < -TOL).sum()),
                "n_tied": int((abs(family_means) <= TOL).sum()),
                "n_worse": int((family_means > TOL).sum()),
            }
        )
        for f in families:
            x = delta.loc[f]
            by_family.append(
                {
                    "scenario_id": f,
                    "method": method,
                    "reference": ref,
                    "estimate": family_means[f],
                    "bank_better": int((x < -TOL).sum()),
                    "bank_tied": int((abs(x) <= TOL).sum()),
                    "bank_worse": int((x > TOL).sum()),
                }
            )
    return {
        "means": pd.DataFrame(levels),
        "contrasts": pd.DataFrame(contrasts),
        "family-contrasts": pd.DataFrame(by_family),
        "family-means": data.groupby(level="scenario_id")
        .mean()
        .loc[families]
        .reset_index(),
    }


def smoke():
    from okapi.node import ValueNode
    from okapi.tree import Tree

    constant = np.full((32, 6, 2), 0.5)
    incumbent = Tree(ValueNode(None, constant[:, 0, 1:2], "m0.npy"))
    risk, choices, _ = fit_menus(incumbent, constant, np.arange(32) % 2, 81719)
    assert risk == 0.5 and choices == {"iid": 0, "unique": 0}

    # Exercise both native output semantics and an exhausted two-provider pair palette.
    for family in ("calibrated_suppressor", "multiclass_committee"):
        rows, probes = evaluate(family, 81713, n_rows=48)
        if len(rows) != 7 or len(probes) != 16:
            raise AssertionError("Probe smoke lost a method or charged slot")
        if family == "calibrated_suppressor":
            unique = next(row for row in rows if row["method"] == "unique")
            iid = next(row for row in rows if row["method"] == "iid")
            assert unique["n_distinct_probes"] == iid["n_distinct_probes"] == 2
            assert unique["predictions_sha256"] == iid["predictions_sha256"]
        else:
            assert all(np.isfinite(row["test_brier"]) for row in rows)
    print(
        "Passed native endpoint smoke for binary, multiclass and the exhausted pair palette."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    families = list(load_preset("standard-v4")["scenarios"])
    if len(families) != 25 or len(set(families)) != 25:
        raise ValueError("Expected 25 distinct standard-v4 scenarios")
    rows, probes = [], []
    start = perf_counter()
    with threadpool_limits(limits=1):
        if args.smoke:
            smoke()
            return
        for family in families:
            for seed in SEEDS:
                selected, candidates = evaluate(family, seed)
                rows.extend(selected)
                probes.extend(candidates)
            print(f"Finished {family}", flush=True)
    scores = pd.DataFrame(rows)
    candidates = pd.DataFrame(probes)
    if len(candidates) != len(families) * len(SEEDS) * 16:
        raise ValueError("Expected eight charged probes in each menu per bank")
    tables = summarize(scores, families)
    tables.update({"scores": scores, "proposals": candidates})
    args.out.mkdir(exist_ok=True, parents=True)
    for name, table in tables.items():
        table.to_csv(args.out / ("r3-" + name + ".csv"), index=False)
    manifest = {
        "scenarios": families,
        "data_seeds": list(SEEDS),
        "gp_seed": 0,
        "n_fit": 512,
        "n_test": 512,
        "n_banks": len(families) * len(SEEDS),
        "native_searches": len(families) * len(SEEDS),
        "probe_fit_evaluations": len(candidates),
        "probe_audit_evaluations": len(candidates),
        "workers": 1,
        "numerical_threads": 1,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": 10000,
        "wall_seconds": perf_counter() - start,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    (args.out / "r3-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(tables["contrasts"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
