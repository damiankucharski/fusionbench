"""Secondary real-data check for a frozen native candidate on sklearn digits.

Run only after the choice of challenger and its parameters is recorded.
The eight test partitions repeatedly sample one dataset. All summaries are
descriptive and the fitted fusion methods consume only provider probabilities.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from fusionbench import __version__, load_dataset
from fusionbench.eval.transfer_grid import brier
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import NativePatternOKAPI
from fusionbench.methods.registry import FACTORIES
from fusionbench.methods.stacking import StackingLogit, StackingLR

SEEDS = tuple(range(2_810_000, 2_810_008))
SOURCE = Path(__file__).resolve().parents[3] / "examples" / "digits_predictions.py"
PROVIDERS = ("lr", "neighbours", "forest")


def digest(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def split_indices(labels, seed):
    """Reconstruct the original builder's row IDs for overlap and checksum audits."""
    indices = np.arange(len(labels))
    base, rest = train_test_split(
        indices, stratify=labels, train_size=0.5, random_state=seed
    )
    fit, test = train_test_split(
        rest, stratify=labels[rest], test_size=0.5, random_state=seed + 1
    )
    assert (len(base), len(fit), len(test)) == (898, 449, 450)
    np.testing.assert_array_equal(np.sort(np.concatenate((base, fit, test))), indices)
    return base, fit, test


def make_methods(seed, challenger_id, kwargs):
    if challenger_id not in FACTORIES or not challenger_id.startswith("okapi"):
        raise ValueError("Choose an explicitly registered native OKAPI challenger")
    if "seed" in kwargs or "fitness" in kwargs:
        raise ValueError(
            "The paired research seed and default Brier objective are fixed"
        )
    if challenger_id == "okapi_native_pattern" and not kwargs:
        raise ValueError(
            "Supply the frozen challenger's parameters to distinguish it from previous"
        )
    methods = {
        "previous": NativePatternOKAPI(seed=seed),
        "challenger": FACTORIES[challenger_id](seed=seed, **kwargs),
        "mean": SimpleAverage(),
        "top_n": TopN("brier"),
        "stack_p": StackingLR(),
        "stack_logit": StackingLogit(),
    }
    for label, model in methods.items():
        if not model.available():
            raise RuntimeError(
                f"{label} is unavailable; install the study's pinned versions"
            )
    return methods


def score_outputs(p, labels):
    risk = brier(p, labels)
    decision = np.argmax(p, axis=1)
    loss = np.square(np.asarray(p, dtype=float) - np.eye(10)[labels]).sum(axis=1)
    return risk, float(np.mean(decision == labels)), decision, loss


def bank(builder, all_labels, seed, challenger_id, kwargs):
    base, fit_idx, test_idx = split_indices(all_labels, seed)
    start = perf_counter()
    with TemporaryDirectory() as temp:
        # Original fixed base suite and float32 save/reload convention.
        ds = load_dataset(builder(temp, seed))
    provider_seconds = perf_counter() - start
    assert tuple(ds.model_ids) == PROVIDERS and ds.meta["base_train_n"] == len(base)
    np.testing.assert_array_equal(ds.fit_y, all_labels[fit_idx])
    np.testing.assert_array_equal(ds.test_y, all_labels[test_idx])
    assert ds.fit_preds.shape == (449, 3, 10) and ds.test_preds.shape == (450, 3, 10)
    ds.fit_preds.setflags(write=False)
    ds.fit_y.setflags(write=False)
    ds.test_preds.setflags(write=False)
    methods = make_methods(seed, challenger_id, kwargs)
    fits = {}
    # Freeze all six fits before scoring any fusion method on the test split.
    for name, model in methods.items():
        start = perf_counter()
        model.fit(ds.fit_preds, ds.fit_y, task=ds.task, n_classes=ds.n_classes)
        elapsed = perf_counter() - start
        counts = getattr(model, "counts_", {})
        fits[name] = {
            "fit_seconds": elapsed,
            "n_selected": getattr(model, "n_selected_", None),
            "chosen_tree": getattr(model, "chosen_", None),
            "score_calls": counts.get("total_score_calls"),
            "fit_brier": brier(model.predict(ds.fit_preds), ds.fit_y),
        }

    scores, cases, providers = [], [], []
    for name, model in methods.items():
        p = model.predict(ds.test_preds)
        risk, accuracy, decision, losses = score_outputs(p, ds.test_y)
        scores.append(
            {
                "seed": seed,
                "method": name,
                **fits[name],
                "test_brier": risk,
                "accuracy": accuracy,
                "prediction_sha256": digest(p),
            }
        )
        for index, target, label, loss in zip(test_idx, ds.test_y, decision, losses):
            cases.append(
                {
                    "seed": seed,
                    "row": index,
                    "method": name,
                    "label": target,
                    "prediction": label,
                    "squared_loss": loss,
                }
            )
    for j, provider in enumerate(ds.model_ids):
        risk, accuracy, _, _ = score_outputs(ds.test_preds[:, j, :], ds.test_y)
        providers.append(
            {
                "seed": seed,
                "provider": provider,
                "fit_brier": brier(ds.fit_preds[:, j, :], ds.fit_y),
                "test_brier": risk,
                "accuracy": accuracy,
            }
        )
    provenance = {
        "seed": seed,
        "base_rows": len(base),
        "fit_rows": len(fit_idx),
        "test_rows": len(test_idx),
        "base_idx_sha256": digest(base.astype("<i8")),
        "fit_idx_sha256": digest(fit_idx.astype("<i8")),
        "test_idx_sha256": digest(test_idx.astype("<i8")),
        "fit_x_sha256": digest(ds.fit_preds),
        "test_x_sha256": digest(ds.test_preds),
        "provider_seconds": provider_seconds,
    }
    return scores, cases, providers, provenance, test_idx


def run(out, challenger_id, kwargs):
    start = perf_counter()
    builder = run_path(str(SOURCE))["write_digits_predictions"]
    all_labels = load_digits().target
    scores, cases, providers, provenance, tests = [], [], [], [], []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            s, c, b, p, idx = bank(builder, all_labels, seed, challenger_id, kwargs)
            scores.extend(s)
            cases.extend(c)
            providers.extend(b)
            provenance.append(p)
            tests.append(idx)
    frame = pd.DataFrame(scores)
    results = frame.pivot(index="seed", columns="method", values="test_brier")
    contrasts = []
    for name in results.columns.drop("challenger"):
        gains = results[name] - results["challenger"]
        contrasts.append(
            {
                "comparator": name,
                "challenger_brier_gain": gains.mean(),
                "wins": int((gains > 1e-12).sum()),
                "ties": int((gains.abs() <= 1e-12).sum()),
                "losses": int((gains < -1e-12).sum()),
                "n_splits": len(gains),
            }
        )
    overlap = [
        len(np.intersect1d(tests[i], tests[j]))
        for i in range(len(tests))
        for j in range(i)
    ]
    manifest = {
        "dataset": "sklearn.datasets.load_digits",
        "seeds": SEEDS,
        "providers": PROVIDERS,
        "previous": "okapi_native_pattern",
        "challenger": challenger_id,
        "challenger_kwargs": kwargs,
        "n_images": len(all_labels),
        "distinct_test_images": len(np.unique(tests)),
        "mean_pairwise_test_overlap": float(np.mean(overlap)),
        "split_banks": provenance,
        "generator_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "fusionbench": __version__,
        "seconds": perf_counter() - start,
    }
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(cases).to_csv(
        out / "cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.DataFrame(providers).to_csv(out / "providers.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    means = frame.groupby("method")[
        ["fit_brier", "test_brier", "accuracy", "fit_seconds"]
    ].mean()
    means.to_csv(out / "means.csv")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", default="okapi_native_pattern")
    parser.add_argument(
        "--challenger-kwargs",
        required=True,
        help="JSON object fixed from the synthetic rounds before opening held-out digits",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    options = json.loads(args.challenger_kwargs)
    if not isinstance(options, dict):
        parser.error("--challenger-kwargs must be a JSON object")
    run(args.out, args.challenger, options)
