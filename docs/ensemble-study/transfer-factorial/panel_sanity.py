"""Two-seed descriptive transfer sanity check. Run with one numerical thread."""

import hashlib
import json
import platform
import runpy
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import okapi
import pandas as pd
import sklearn
from gate import gate_choices
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.transfer import load_preset

STUDY = Path(__file__).resolve().parent
OUT = STUDY / "results"
PANEL = runpy.run_path(str(STUDY.parent / "fold-promotion" / "panel.py"))
SEEDS = (254100, 254101)
ROWS = 512


def evaluate(scenario, seed, scope):
    bank = generate_dataset(scenario, seed=seed, n_per_split=ROWS)
    start = perf_counter()
    decision = gate_choices(bank.fit_preds, bank.fit_y)
    gate_seconds = perf_counter() - start
    rows = PANEL["references"](bank)
    by_name = {row["method"]: row for row in rows}
    context = {
        "scenario_id": scenario,
        "panel_scope": scope,
        "data_seed": seed,
        "n_fit": len(bank.fit_y),
        "n_test": len(bank.test_y),
        "provider_count": bank.fit_preds.shape[1],
        "n_classes": bank.fit_preds.shape[2],
        "fit_inputs_sha256": PANEL["digest"](bank.fit_preds),
        "fit_labels_sha256": PANEL["digest"](bank.fit_y),
        "test_inputs_sha256": PANEL["digest"](bank.test_preds),
        "test_labels_sha256": PANEL["digest"](bank.test_y),
    }
    # Both choices reuse the same validation fit and one already fitted full-data model.
    for policy in ("plain", "guard"):
        actual = "regularized" if decision[policy] else "mean"
        rows.append(
            {
                **by_name[actual],
                "method": policy,
                "deployed_method": actual,
                "gate_gain": decision["gain"],
                "gate_se": decision["se"],
                "gate_plain": decision["plain"],
                "gate_guard": decision["guard"],
                "shared_gate_fit_seconds": gate_seconds,
            }
        )
    for row in rows:
        row.setdefault("deployed_method", row["method"])
    return [{**context, **row} for row in rows]


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    scenarios = tuple(load_preset("standard-v5")["scenarios"])
    assert scenarios == (
        *load_preset("standard-v4")["scenarios"],
        "rare_shortcut_noise",
        "stratified_pool",
    )
    start = perf_counter()
    all_rows = []
    with threadpool_limits(limits=1):
        for i, scenario in enumerate(scenarios):
            scope = "planned_26" if i < 26 else "appended_27th"
            for seed in SEEDS:
                all_rows.extend(evaluate(scenario, seed, scope))
    OUT.mkdir(exist_ok=True)
    table = pd.DataFrame(all_rows)
    name = OUT / "panel-sanity.csv"
    table.to_csv(name, index=False)
    assert len(table) == 27 * 2 * 6
    timings = table.query(
        "method in ['mean','top_n','prob_lr','regularized']"
    ).fit_seconds.sum()
    meta = {
        "protocol_commit": "8d4dae2",
        "suite_extension_protocol_commit": "02d81b6",
        "fusionbench_base_commit": revision(STUDY),
        "okapi_commit": revision(Path(okapi.__file__).resolve().parent),
        "scenarios": list(scenarios),
        "planned_scenarios": 26,
        "appended_scenario": "stratified_pool",
        "seeds": SEEDS,
        "rows_per_split": ROWS,
        "methods": list(table.method.unique()),
        "generation_api": "generate_dataset(scenario, seed=seed, n_per_split=512)",
        "gate": "gate_choices(fit_x, fit_y) once per bank; compare gain > 0 or gain > SE; share all full-data predictions",
        "fit_seconds_column": "time for deployed full-data method; identical reused values on decision rows",
        "ordinary_fit_seconds": float(timings),
        "additional_gate_seconds": float(
            table.query("method == 'guard'").shared_gate_fit_seconds.sum()
        ),
        "wall_seconds": perf_counter() - start,
        "numerical_threads": 1,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "sha256": hashlib.sha256(name.read_bytes()).hexdigest(),
    }
    for scope, subset in [
        ("planned_26", table.query("panel_scope == 'planned_26'")),
        ("appended_27th", table.query("panel_scope == 'appended_27th'")),
        ("all_27", table),
    ]:
        meta[scope + "_mean_test_brier"] = (
            subset.groupby("method").test_brier.mean().to_dict()
        )
    (OUT / "panel-sanity-manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in meta.items()
                if k.startswith(
                    (
                        "wall_seconds",
                        "planned_26_mean",
                        "appended_27th_mean",
                        "all_27_mean",
                    )
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
