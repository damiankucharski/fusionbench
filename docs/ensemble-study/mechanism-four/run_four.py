"""Fit synthetic-only study policies and score the resulting immutable decisions."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from banks_study import banks, score
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE.parent / "full-data-five"))
from engine import fit_full, fit_model
from interventions import fit_candidate
from run_full import digest, forecast


def fit_models(r, x, y, groups=None, chosen=None):
    from sparse_gp import fit_sparse, fit_sparse_gp, fit_sparse_pool

    models = {
        "mean": fit_model(SimpleAverage(), x, y),
        "top_n": fit_model(TopN(metric="brier"), x, y),
        "lr": fit_model(StackingLR(c_value=1.0), x, y),
        "full": fit_full(x, y),
        "pull_mean": fit_candidate("pull_mean", x, y, seed=0, groups=groups),
        "previous8": fit_model(
            RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8),
            x,
            y,
        ),
    }
    if r == 1:
        models["sparse_pool"] = fit_sparse_pool(x)
    elif r == 2:
        models.update(fit_sparse(x, y))
    else:
        from calibrate_gp import fit_calibrated

        if r == 3 or chosen in {"sparse_gp", "cal_sparse"}:
            models["sparse_gp"] = fit_sparse_gp("sparse_gp", x, y)
        for name, raw in (
            ("cal_pull", "pull_mean"),
            ("raw_pull", "pull_mean"),
            ("cal_mean", "mean"),
            ("cal_sparse", "sparse_gp"),
        ):
            if r == 3 or name == chosen:
                models[name] = fit_calibrated(
                    models[raw], x, y, groups=groups, penalized=(name != "raw_pull")
                )
    return models


def measure(r, b, chosen=None):
    start = perf_counter()
    models = fit_models(r, b.x, b.y, b.groups, chosen)
    fit_seconds = perf_counter() - start
    return score_models(r, b, models), fit_seconds


def score_models(r, b, models):
    """Score immutable models after every fit and selection decision has finished."""
    truth = np.eye(b.x.shape[-1])[b.y]
    rows = []
    for name, m in models.items():
        p, f = forecast(m, b.audit_x), forecast(m, b.x)
        tree = getattr(m, "best_tree", None)
        if tree is None and hasattr(m, "chosen_"):
            tree = m.trees_[m.chosen_]
        if tree is not None:
            providers = len(tree.unique_value_node_ids)
        elif name == "top_n":
            providers = m.n_selected_
        else:
            providers = b.x.shape[1]
        cal = getattr(m, "calibration_", None)
        beta = cal.get("beta") if isinstance(cal, dict) else getattr(cal, "beta", None)
        row = {
            **b.context,
            "round": r,
            "method": name,
            **score(b, p),
            "fit_brier": float(np.square(f - truth).sum(1).mean()),
            "providers": providers,
            "nodes": None if tree is None else tree.nodes_count,
            "fit_hash": digest(f),
            "test_hash": digest(p),
            "beta": beta,
            "subset_seed": json.dumps([int(i) for i in m.selected_indices_])
            if hasattr(m, "selected_indices_")
            else None,
            "program": None if tree is None else repr(tree.structural_key()),
        }
        if b.audit_regime is not None:
            # Expected full Brier conditional on the same latent examples, split by planted regime.
            for region, mask in (("on", b.audit_regime), ("off", ~b.audit_regime)):
                target = b.q[mask]
                prob = p[mask, 1]
                row[f"regime_{region}_brier"] = (
                    float(2 * np.mean((prob - target) ** 2 + target * (1 - target)))
                    if mask.any()
                    else None
                )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 5))
    r = parser.parse_args().round
    chosen = None
    if r == 4:
        from choose_four import locked

        chosen = locked()["chosen"]
    start = perf_counter()
    fit_seconds = 0
    rows = []
    with threadpool_limits(limits=1):
        for i, b in enumerate(banks(r), 1):
            batch, seconds = measure(r, b, chosen)
            rows.extend(batch)
            fit_seconds += seconds
            if i % 30 == 0:
                print(r, i, flush=True)
    assert i == (408 if r == 4 else 156)
    tab = pd.DataFrame(rows)
    assert not tab.duplicated(["cohort", "family", "seed", "method"]).any()
    OUT.mkdir(exist_ok=True)
    tab.to_csv(OUT / f"r{r}-scores.csv", index=False)
    (OUT / f"r{r}-time.json").write_text(
        json.dumps(
            {
                "round": r,
                "banks": i,
                "scores": len(tab),
                "seconds": perf_counter() - start,
                "fit_seconds": fit_seconds,
                "numerical_threads": 1,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
