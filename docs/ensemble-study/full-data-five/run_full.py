"""Fit native controls and one registered intervention set before scoring any audit rows."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from banks_full import audit_scores, banks
from engine import fit_full, fit_model, fit_round1
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
INTERVENTIONS = {
    2: ("seeded_mean", "sham"),
    3: ("seeded_mean", "pull_mean", "pull_zero"),
    4: ("seeded_mean", "paired_se", "paired_zero"),
}


def fit_methods(r, x, y, groups, chosen=None):
    if r == 1:
        models = fit_round1(x, y, groups=groups)
    else:
        from interventions import fit_candidate

        models = {"full": fit_full(x, y)}
        for name in (chosen,) if r == 5 else INTERVENTIONS[r]:
            models[name] = fit_candidate(name, x, y, seed=0, groups=groups)
    for name, reference in (
        ("mean", SimpleAverage()),
        ("top_n", TopN(metric="brier")),
        ("lr", StackingLR(c_value=1.0)),
        (
            "previous8",
            RegularizedNativePatternOKAPI(seed=0, split_seed=0, gp_validation_top_k=8),
        ),
    ):
        models[name] = fit_model(reference, x, y)
    return models


def forecast(model, x):
    p = np.asarray(model.predict(x), dtype=np.float64)
    if p.shape != (len(x), x.shape[-1]) or not np.isfinite(p).all():
        raise ValueError("Invalid shape or non-finite forecast")
    if p.min() < -1e-8 or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError("Unnormalized forecast")
    return p


def digest(p):
    return hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()


def measure(r, b, chosen):
    start = perf_counter()
    models = fit_methods(r, b.x, b.y, getattr(b, "groups", None), chosen)
    fit_time = perf_counter() - start
    # These accesses occur after every method has finished fitting and selecting its program.
    avg = models["mean"]
    source_mean = forecast(avg, b.x)
    target_mean = forecast(avg, b.audit_x)
    target = np.eye(b.x.shape[-1])[b.y]
    rows = []
    fits = {}
    for name, model in models.items():
        p, f = forecast(model, b.audit_x), forecast(model, b.x)
        audit = audit_scores(b, p)
        tree = getattr(model, "best_tree", None)
        if tree is None and hasattr(model, "chosen_"):
            tree = model.trees_[model.chosen_]
        native = getattr(model, "okapi", None)
        info = {}
        if native is not None:
            info["n_evolution_rows"] = int(np.asarray(native.gt_tensor).size)
            info["final_population"] = len(native.population)
            info["unique_final"] = len(
                {tr.structural_key() for tr in native.population}
            )
            info["front_size"] = len(native.pareto_trees)
            if hasattr(model, "selection_"):
                info["holdout"] = model.selection_
        if tree is not None:
            info["program"] = repr(tree.structural_key())
        if hasattr(model, "n_selected_"):
            info["top_n"] = int(model.n_selected_)
        fits[name] = info
        rows.append(
            {
                **b.context,
                "round": r,
                "method": name,
                "brier": audit["test_brier"],
                "fit_brier": np.square(f - target).sum(1).mean(),
                "macro_f1": audit["test_macro_f1"],
                "average_precision": audit["test_ap"],
                "test_prediction_hash": digest(p),
                "fit_prediction_hash": digest(f),
                "n_evolution_rows": info.get("n_evolution_rows"),
                "nodes": tree.nodes_count if tree else None,
                "providers_used": len(tree.unique_value_node_ids)
                if tree
                else info.get("top_n"),
                "equals_mean_fit": bool(
                    np.allclose(f, source_mean, rtol=0, atol=1e-12)
                ),
                "equals_mean_audit": bool(
                    np.allclose(p, target_mean, rtol=0, atol=1e-12)
                ),
            }
        )
    return rows, {
        **b.context,
        "fits": fits,
        "fit_seconds": fit_time,
        "seconds": perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=range(1, 6))
    r = parser.parse_args().round
    chosen = None
    if r == 5:
        from decide import read_locked

        chosen = read_locked()["chosen"]
    rows, details = [], []
    start = perf_counter()
    with threadpool_limits(limits=1):
        for i, bank in enumerate(banks(r), 1):
            batch, info = measure(r, bank, chosen)
            rows.extend(batch)
            details.append(info)
            if i % 25 == 0:
                print(f"Round {r}: {i} banks", flush=True)
    assert i == (284 if r == 5 else 118)
    OUT.mkdir(exist_ok=True)
    scores = pd.DataFrame(rows)
    assert not scores.duplicated(["cohort", "family", "seed", "method"]).any()
    scores.to_csv(OUT / f"round{r}-scores.csv", index=False)
    with (OUT / f"round{r}-fits.jsonl").open("w") as stream:
        for item in details:
            stream.write(
                json.dumps(item, allow_nan=False, default=lambda x: x.item()) + "\n"
            )
    (OUT / f"round{r}-run.json").write_text(
        json.dumps(
            {
                "round": r,
                "banks": i,
                "score_rows": len(rows),
                "seconds": perf_counter() - start,
                "numerical_threads": 1,
                "optimizer_seed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Round {r} finished", flush=True)


if __name__ == "__main__":
    main()
