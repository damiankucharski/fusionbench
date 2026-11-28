"""Compare eight native pair probes with different operator coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from runpy import run_path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.eval.suite_point_gp import _bind, predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE.parent / "fold-promotion/search.py"
ENGINE = run_path(str(ENGINE_PATH))
SEEDS = range(242100, 242124)
SIZES = (6, 12)
LAWS = ("max", "singleton")
NATIVE = ("gp", "min8", "balanced8", "max8")
METHODS = (*NATIVE, "mean", "top_n", "prob_lr", "regularized")
TOL = 1e-12


def scores_and_labels(seed, size, law):
    order = np.random.default_rng([seed, size, 15]).permutation(size)
    a, b = (int(np.flatnonzero(order == role)[0]) for role in (0, 1))
    splits = []
    for split, n in ((1, 192), (2, 4096)):
        latent = (
            np.random.default_rng([seed, split, 0]).integers(1, 64, size=(n, 12)) / 64
        )
        positive = latent[:, order]
        q = (
            latent[:, 0]
            if law == "singleton"
            else np.maximum(latent[:, 0], latent[:, 1])
        )
        u = np.random.default_rng([seed, split, 1]).random(n)
        splits.append(
            (np.stack((1 - positive, positive), axis=-1), (u < q).astype(np.int64), q)
        )
    return splits, a, b


def schedule(seed, size):
    from okapi.pair_coverage import pair_schedule

    seq = []
    for op_idx, op in enumerate(("min", "max")):
        op_seed = int(
            np.random.SeedSequence([seed, 4000000, op_idx]).generate_state(1)[0]
        )
        seq.append(
            pair_schedule(size, 8, seed=op_seed, policy="unique", operators=(op,))
        )
    return {
        "min8": seq[0],
        "balanced8": tuple(
            pair for items in zip(seq[0][:4], seq[1][:4], strict=True) for pair in items
        ),
        "max8": seq[1],
    }


def brier(p, y):
    return float(np.mean(np.sum((p - np.eye(2)[y]) ** 2, axis=1)))


def matches(p, q):
    return bool(np.max(np.abs(p[:, 1] - q)) <= TOL)


def fit_menus(x, y, seed):
    """Freeze all native decisions before admitting oracle or test data."""
    from okapi.node import ValueNode
    from okapi.pair_coverage import pair_tree

    gp = ENGINE["fit_search"](
        x,
        y,
        groups=np.arange(len(y)),
        search_seed=seed + 1_000_000,
        data_seed=seed,
        path="pooled",
    )
    key = gp["choices"]["pooled_final"]
    frozen = gp["archive"][key]
    incumbent = (None, frozen["tree"], frozen["pooled_brier"])
    roots = [ValueNode(None, v, k) for k, v in _bind(x).items()]
    menus = {"gp": [incumbent]}
    selected = {"gp": 0}
    for name, sequence in schedule(seed, len(roots)).items():
        offers = [incumbent]
        for spec in sequence:
            tree = pair_tree(roots, spec)
            fit_loss = brier(predict_tree(tree, x), y)
            offers.append((spec, tree, fit_loss))
        menus[name] = offers
        selected[name] = min(range(len(offers)), key=lambda i: offers[i][2])
    return gp, menus, selected


def fit_references(x, y, seed):
    methods = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
        "regularized": RegularizedNativePatternOKAPI(
            seed=seed + 1_000_000, split_seed=seed + 1_000_000
        ),
    }
    for method in methods.values():
        method.fit(x, y, task="binary", n_classes=2)
    return methods


def score_cell(seed, size, law):
    rows, slots, states = [], [], []
    ((x, y, fit_q), (test, test_y, q)), a, b = scores_and_labels(seed, size, law)
    gp, menus, selected = fit_menus(x, y, seed)
    refs = fit_references(x, y, seed)
    common = {"seed": seed, "n_providers": size, "law": law}
    final = gp["snapshots"][-1]
    cache = {}

    def evaluate(tree):
        key = tree.structural_key()
        if key not in cache:
            test_p = predict_tree(tree, test)
            exact = matches(test_p, q) and matches(predict_tree(tree, x), fit_q)
            cache[key] = {
                "audit_brier": brier(test_p, test_y),
                "audit_excess": float(2 * np.mean((test_p[:, 1] - q) ** 2)),
                "is_exact": exact,
                "program_digest": ENGINE["digest"](key),
                "n_used": len(tree.unique_value_node_ids),
                "nodes": tree.nodes_count,
            }
        return cache[key]

    for record in gp["archive"].values():
        evaluate(record["tree"])
    base = menus["gp"][0][1]
    states.append(
        {
            **common,
            "role_a": a,
            "role_b": b,
            "search_seed": seed + 1_000_000,
            "fit_x_sha256": ENGINE["array_digest"](x),
            "fit_y_sha256": ENGINE["array_digest"](y),
            "fit_q_sha256": ENGINE["array_digest"](fit_q),
            "audit_x_sha256": ENGINE["array_digest"](test),
            "audit_y_sha256": ENGINE["array_digest"](test_y),
            "audit_q_sha256": ENGINE["array_digest"](q),
            "founder_digest": gp["founder_digest"],
            "path_digest": final["path_digest"],
            "gp_objective_evaluations": final["objective_evaluations"],
            "gp_offspring_proposals": final["offspring_proposals"],
            "gp_unique_genotypes": len(gp["archive"]),
            "gp_exact_ever": any(cache[k]["is_exact"] for k in gp["archive"]),
            "gp_exact_in_final": any(cache[k]["is_exact"] for k in final["keys"]),
            "gp_selected_exact": evaluate(base)["is_exact"],
        }
    )
    for method, offers in menus.items():
        menu = []
        for i, (spec, tree, risk) in enumerate(offers):
            op, left, right = spec if spec else ("gp", -1, -1)
            row = {
                **common,
                "method": method,
                "slot": i - 1,
                "operator": op,
                "left": left,
                "right": right,
                "planted_max_probe": op == "max"
                and (left, right) == tuple(sorted((a, b))),
                "fit_brier": risk,
                **evaluate(tree),
                "selected": i == selected[method],
            }
            menu.append(row)
        slots.extend(menu)
        chosen = menu[selected[method]]
        rows.append(
            {
                **common,
                "method": method,
                "selected_slot": chosen["slot"],
                "n_used": chosen["n_used"],
                "fit_brier": chosen["fit_brier"],
                "audit_brier": chosen["audit_brier"],
                "audit_excess": chosen["audit_excess"],
                "selected_exact": chosen["is_exact"],
                "selected_digest": chosen["program_digest"],
                "probe_fit_calls": len(offers) - 1,
                "gp_objective_evaluations": final["objective_evaluations"],
                "total_fit_calls": final["objective_evaluations"] + len(offers) - 1,
                "exact_in_menu": any(row["is_exact"] for row in menu),
                "exact_probe_offered": any(row["is_exact"] for row in menu[1:]),
                "planted_max_offered": any(
                    row["planted_max_probe"] for row in menu[1:]
                ),
                "selected_probe": selected[method] > 0,
            }
        )
    for name, method in refs.items():
        fit_p, test_p = method.predict(x), method.predict(test)
        count = size
        if name == "top_n":
            count = method.n_selected_
        if name == "regularized":
            count = len(method.trees_[method.chosen_].unique_value_node_ids)
        rows.append(
            {
                **common,
                "method": name,
                "n_used": count,
                "fit_brier": brier(fit_p, y),
                "audit_brier": brier(test_p, test_y),
                "audit_excess": float(2 * np.mean((test_p[:, 1] - q) ** 2)),
                "selected_exact": matches(test_p, q) and matches(fit_p, fit_q),
            }
        )
    return rows, slots, states


def score_bank(seed):
    cells = [score_cell(seed, size, law) for size, law in product(SIZES, LAWS)]
    return tuple([row for cell in cells for row in cell[i]] for i in range(3))


def interval(values, draws):
    vals = np.asarray(values, dtype=float)
    low, high = np.percentile(vals[draws].mean(axis=1), [2.5, 97.5])
    return {
        "estimate": float(vals.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_lower": int(np.count_nonzero(vals < -TOL)),
        "n_tied": int(np.count_nonzero(abs(vals) <= TOL)),
        "n_higher": int(np.count_nonzero(vals > TOL)),
    }


def analyze(out):
    scores = pd.read_csv(out / "r2-scores.csv")
    probes = pd.read_csv(out / "r2-candidates.csv.gz")
    states = pd.read_csv(out / "r2-banks.csv")
    cells = len(SEEDS) * len(SIZES) * len(LAWS)
    if (
        len(states) != cells
        or len(scores) != cells * len(METHODS)
        or len(probes) != cells * (1 + 9 * 3)
    ):
        raise ValueError("Incomplete experiment grid")
    draws = np.random.default_rng(97832).integers(len(SEEDS), size=(10_000, len(SEEDS)))
    summaries, paired, primary = [], [], []
    comparisons = [
        ("balanced8", ref)
        for ref in ("min8", "max8", "gp", "mean", "top_n", "prob_lr", "regularized")
    ]
    comparisons.extend([("max8", "gp"), ("min8", "gp")])
    for law in LAWS:
        for size in (*SIZES, "both"):
            part = scores[scores.law == law]
            if size != "both":
                part = part[part.n_providers == size]
            for metric in ("audit_excess", "audit_brier", "fit_brier", "n_used"):
                wide = (
                    part.pivot(
                        index=["seed", "n_providers"], columns="method", values=metric
                    )
                    .groupby(level="seed")
                    .mean()
                    .loc[list(SEEDS), list(METHODS)]
                )
                if not np.isfinite(wide).all().all():
                    raise ValueError("Incomplete or nonfinite risk grid")
                for method in METHODS:
                    summaries.append(
                        {
                            "law": law,
                            "n_providers": size,
                            "method": method,
                            "metric": metric,
                            "estimate": wide[method].mean(),
                        }
                    )
                for method, reference in comparisons:
                    change = (wide[method] - wide[reference]).to_numpy()
                    paired.append(
                        {
                            "law": law,
                            "n_providers": size,
                            "method": method,
                            "reference": reference,
                            "metric": metric,
                            "n_banks": len(change),
                            **interval(change, draws),
                        }
                    )
                    if (law, metric, method, reference) == (
                        "max",
                        "audit_excess",
                        "balanced8",
                        "min8",
                    ):
                        primary.extend(
                            {"seed": seed, "n_providers": size, "difference": delta}
                            for seed, delta in zip(SEEDS, change, strict=True)
                        )
    coverage = []
    for (law, size, name), part in scores[scores.method.isin(NATIVE)].groupby(
        ["law", "n_providers", "method"]
    ):
        diag = states[(states.law == law) & (states.n_providers == size)]
        enabled = part.exact_in_menu.eq(True)
        offered = part.exact_probe_offered.eq(True)
        hit = part.selected_exact.eq(True)
        coverage.append(
            {
                "law": law,
                "n_providers": size,
                "method": name,
                "banks": len(part),
                "gp_exact_ever": int(diag.gp_exact_ever.sum()),
                "gp_exact_final": int(diag.gp_exact_in_final.sum()),
                "gp_selected_exact": int(diag.gp_selected_exact.sum()),
                "planted_max_offered": int(part.planted_max_offered.eq(True).sum()),
                "exact_probe_offered": int(offered.sum()),
                "exact_menu": int(enabled.sum()),
                "deployed_exact": int(hit.sum()),
                "deployed_exact_when_offered_by_probe": int((offered & hit).sum()),
                "deployed_exact_probe_when_offered": int(
                    (offered & hit & part.selected_probe.eq(True)).sum()
                ),
                "exact_menu_bypassed": int((enabled & ~hit).sum()),
                "deployed_any_probe": int(part.selected_probe.eq(True).sum()),
                "probe_fit_calls": int(part.probe_fit_calls.sum()),
                "gp_objective_evaluations": int(part.gp_objective_evaluations.sum()),
            }
        )
    for name, table in (
        ("summary", summaries),
        ("contrasts", paired),
        ("coverage", coverage),
        ("primary-by-bank", primary),
    ):
        pd.DataFrame(table).to_csv(out / f"r2-{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.out)
        return
    start = perf_counter()
    utc = datetime.now(UTC).isoformat()
    batches = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            batches.append(score_bank(seed))
            print(f"Completed {len(batches)}/{len(SEEDS)} banks.", flush=True)
    elapsed = perf_counter() - start
    args.out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(("r2-scores.csv", "r2-candidates.csv.gz", "r2-banks.csv")):
        df = pd.DataFrame(row for batch in batches for row in batch[i])
        df.to_csv(
            args.out / name,
            index=False,
            compression={"method": "gzip", "mtime": 0}
            if name.endswith(".gz")
            else None,
        )
    from okapi import pair_coverage

    manifest = {
        "started_utc": utc,
        "elapsed_seconds": elapsed,
        "workers": 1,
        "numerical_threads_per_worker": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "banks": len(SEEDS),
        "actual_gp_fits": len(SEEDS) * len(SIZES) * len(LAWS),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("protocol", HERE / "r2-protocol.md"),
                ("runner", Path(__file__)),
                ("gp_engine", ENGINE_PATH),
                ("pair_coverage", Path(pair_coverage.__file__)),
            )
        },
    }
    (args.out / "r2-run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    analyze(args.out)
    print(f"Completed {len(SEEDS)} paired banks in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
