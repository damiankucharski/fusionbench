"""Compare ordinary and anchored Brier selection on a fixed provider menu."""

import argparse
import gzip
import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fusionbench import generate_dataset
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.registry import STANDARD_V3_SCENARIOS

HERE = Path(__file__).resolve().parent
FIT_ROWS = 192
TEST_ROWS = 1024
WEIGHT = 0.25
TOL = 1e-12


def tensor_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def candidates(preds):
    x = np.asarray(preds, dtype=np.float64)
    k = x.shape[1]
    indices = list(combinations(range(k), 2))
    prior = x.mean(axis=1)
    values = [
        prior,
        *x.transpose(1, 0, 2),
        *[(x[:, i] + x[:, j]) * 0.5 for i, j in indices],
    ]
    names = [
        "mean",
        *[f"single_{i}" for i in range(k)],
        *[f"pair_{i}_{j}" for i, j in indices],
    ]
    return names, np.stack(values)


def measures(predictions, prior, labels):
    target = np.eye(prior.shape[1])[labels]
    return {
        "brier": np.square(predictions - target).sum(axis=-1).mean(axis=-1),
        "displacement": np.square(predictions - prior).sum(axis=-1).mean(axis=-1),
        "accuracy": (predictions.argmax(axis=-1) == labels).mean(axis=-1),
    }


def assess_menu(x, y):
    names, raw = candidates(x)
    prior = raw[0]
    mixture = prior + WEIGHT * (raw - prior)
    return names, {
        "raw": measures(raw, prior, y),
        "mixture": measures(mixture, prior, y),
    }


def run_bank(scenario, index, rep):
    seed = 4_810_000 + 1000 * index + rep
    ds = generate_dataset(scenario, n_per_split=TEST_ROWS, seed=seed)
    x, y = ds.fit_preds[:FIT_ROWS], ds.fit_y[:FIT_ROWS]
    names, fit = assess_menu(x, y)
    ordinary = int(np.argmin(fit["raw"]["brier"]))
    anchored = int(np.argmin(fit["mixture"]["brier"]))
    if (
        fit["raw"]["displacement"][anchored] + TOL
        < fit["raw"]["displacement"][ordinary]
    ):
        raise AssertionError("The anchored winner has smaller raw fit displacement")
    models = {
        "top_n": TopN("brier"),
        "prob_lr": StackingLR(c_value=1),
        "logit_lr": StackingLogit(c_value=1),
    }
    for model in models.values():
        model.fit(x, y, task=ds.task, n_classes=ds.n_classes)
    test_names, test = assess_menu(ds.test_preds, ds.test_y)
    assert test_names == names
    key = {"scenario": scenario, "rep": rep, "seed": seed}
    bank = {
        **key,
        "n_candidates": len(names),
        "ordinary_winner": names[ordinary],
        "anchored_winner": names[anchored],
        "n_providers": len(ds.model_ids),
        "n_classes": ds.n_classes,
        "n_top": models["top_n"].n_selected_,
        "fit_inputs_sha256": tensor_hash(x),
        "fit_labels_sha256": tensor_hash(y),
        "test_inputs_sha256": tensor_hash(ds.test_preds),
        "test_labels_sha256": tensor_hash(ds.test_y),
    }
    rows, details = [], []
    methods = {
        "mean": ("raw", 0),
        "ordinary_raw": ("raw", ordinary),
        "ordinary_mixed": ("mixture", ordinary),
        "anchored_raw": ("raw", anchored),
        "anchored_mixed": ("mixture", anchored),
    }
    for domain, v, inputs, labels in (
        ("fit", fit, x, y),
        ("test", test, ds.test_preds, ds.test_y),
    ):
        for i, name in enumerate(names):
            details.append(
                {
                    **key,
                    "domain": domain,
                    "candidate": name,
                    "raw_brier": v["raw"]["brier"][i],
                    "mixed_brier": v["mixture"]["brier"][i],
                    "raw_displacement": v["raw"]["displacement"][i],
                }
            )
        for name, (form, winner) in methods.items():
            rows.append(
                {
                    **key,
                    "domain": domain,
                    "method": name,
                    **{m: val[winner] for m, val in v[form].items()},
                }
            )
        prior = np.asarray(inputs, dtype=np.float64).mean(1)
        for name, model in models.items():
            rows.append(
                {
                    **key,
                    "domain": domain,
                    "method": name,
                    **measures(model.predict(inputs), prior, labels),
                }
            )
        bank[domain + "_raw_displacement_delta"] = (
            v["raw"]["displacement"][anchored] - v["raw"]["displacement"][ordinary]
        )
        bank[domain + "_mixed_brier_delta"] = (
            v["mixture"]["brier"][anchored] - v["mixture"]["brier"][ordinary]
        )
        bank[domain + "_raw_brier_delta"] = (
            v["raw"]["brier"][anchored] - v["raw"]["brier"][ordinary]
        )
    return bank, rows, details


def summarize(banks):
    fields = [k for k in banks if k.endswith("_delta")]
    grouped = banks.groupby("scenario")[fields].mean()
    stats = []
    rng = np.random.default_rng(4810999)
    draws = rng.integers(8, size=(10000, len(grouped), 8))
    for field in fields:
        values = banks.pivot(index="scenario", columns="rep", values=field)
        v = values.to_numpy()
        avg = v[np.arange(len(grouped))[None, :, None], draws].mean(axis=(1, 2))
        ci = np.quantile(avg, [0.025, 0.975])
        nums = grouped[field]
        stats.append(
            {
                "contrast": field,
                "mean": v.mean(),
                "ci025": ci[0],
                "ci975": ci[1],
                "negative_banks": (v < -TOL).sum(),
                "tie_banks": (np.abs(v) <= TOL).sum(),
                "positive_banks": (v > TOL).sum(),
                "negative_scenarios": (nums < -TOL).sum(),
                "tie_scenarios": (nums.abs() <= TOL).sum(),
                "positive_scenarios": (nums > TOL).sum(),
            }
        )
    return grouped, pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    suite = tuple(STANDARD_V3_SCENARIOS)
    if len(suite) != 24:
        raise ValueError(
            "The frozen design requires the standard-v3 list of 24 scenarios"
        )
    tick = perf_counter()
    banks, rows, details = [], [], []
    with threadpool_limits(limits=1):
        for i, scenario in enumerate(suite):
            for rep in range(8):
                bank, scores, menu = run_bank(scenario, i, rep)
                banks.append(bank)
                rows.extend(scores)
                details.extend(menu)
    banks = pd.DataFrame(banks)
    frame = pd.DataFrame(rows)
    banks.to_csv(out / "banks.csv", index=False)
    frame.to_csv(out / "scores.csv", index=False)
    # Fixed gzip time supplies a reproducible candidate-metric archive.
    with (
        (out / "candidates.csv.gz").open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz,
    ):
        gz.write(pd.DataFrame(details).to_csv(index=False).encode())
    scenarios, summary = summarize(banks)
    scenarios.to_csv(out / "scenario-means.csv")
    summary.to_csv(out / "contrasts.csv", index=False)
    frame.groupby(["domain", "method"])[
        ["brier", "accuracy", "displacement"]
    ].mean().to_csv(out / "means.csv")
    manifest = {
        "scenarios": suite,
        "seed_root": 4810000,
        "replicates": 8,
        "selection_rows": FIT_ROWS,
        "test_rows": TEST_ROWS,
        "tree_weight": WEIGHT,
        "candidate_domain_rows": len(details),
        "bank_count": len(banks),
        "bootstrap_seed": 4810999,
        "bootstrap_draws": 10000,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "elapsed_seconds": perf_counter() - tick,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
