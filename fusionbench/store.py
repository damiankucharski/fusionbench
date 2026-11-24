"""On-disk dataset format.

Layout (mirrors OKAPI's value-node convention, one file per base model)::

    data/<name>/
        meta.json                  # task, n_classes, model_ids, oracle, knobs, ...
        gt/fit.npy   gt/test.npy   # int [N]
        fit/<model_id>.npy         # float [N, C], rows sum to 1
        test/<model_id>.npy

Predictions are stored as ``.npy`` (lightweight, no torch dependency). The OKAPI
adapter converts to whatever tensor format OKAPI expects at run time.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from fusionbench.eval.metrics import METRIC_NAMES

from fusionbench.types import Dataset

SPLITS = ("fit", "test")


def save_dataset(ds: Dataset, root: str | Path) -> Path:
    """Write ``ds`` under ``root/<name>/`` and return that directory."""
    for component in (ds.name, *ds.model_ids):
        if (not isinstance(component, str) or not component or component in (".", "..")
                or Path(component).name != component or "\\" in component or "\0" in component):
            raise ValueError("Dataset and provider names must be single file names")
    base = Path(root) / ds.name
    (base / "gt").mkdir(parents=True, exist_ok=True)
    (base / "oracle").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (base / split).mkdir(parents=True, exist_ok=True)

    preds = {"fit": ds.fit_preds, "test": ds.test_preds}
    ys = {"fit": ds.fit_y, "test": ds.test_y}
    oracles = {"fit": ds.fit_oracle, "test": ds.test_oracle}
    for split in SPLITS:
        np.save(base / "gt" / f"{split}.npy", ys[split].astype(np.int64))
        for k, model_id in enumerate(ds.model_ids):
            arr = preds[split][:, k, :].astype(np.float32)
            np.save(base / split / f"{model_id}.npy", arr)
        oracle_file = base / "oracle" / f"{split}.npy"
        if oracles[split] is not None:
            np.save(oracle_file, oracles[split].astype(np.float32))
        else:
            oracle_file.unlink(missing_ok=True)

    meta = {
        "name": ds.name,
        "task": ds.task,
        "n_classes": ds.n_classes,
        "model_ids": ds.model_ids,
        "oracle_description": ds.oracle_description,
        "expected_winners": ds.expected_winners,
        "has_oracle": ds.test_oracle is not None,
        "n_fit": int(ds.fit_y.shape[0]),
        "n_test": int(ds.test_y.shape[0]),
        "format": "npy",
        "knobs": ds.meta,
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    return base


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset previously written by :func:`save_dataset`."""
    return _read_dataset_dir(Path(path))


def _read_dataset_dir(base: Path | zipfile.Path) -> Dataset:
    """Read the native directory layout from disk or an open reference archive."""
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    model_ids = meta["model_ids"]

    if not isinstance(model_ids, list) or not model_ids or any(
        not isinstance(m, str) or not m or Path(m).name != m or m in (".", "..")
        for m in model_ids
    ) or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be a nonempty list of unique file names")
    classes = meta["n_classes"]
    if (not isinstance(classes, int) or classes < 2 or meta["task"] not in ("binary", "multiclass")
            or (meta["task"] == "binary" and classes != 2)):
        raise ValueError("task and n_classes must describe binary or multiclass probabilities")
    if meta.get("knobs", {}).get("primary_metric", "macro_f1") not in METRIC_NAMES:
        raise ValueError("unknown primary metric in meta.json")

    def probabilities(file, n):
        with file.open("rb") as stream:
            p = np.load(stream, allow_pickle=False)
        if (p.shape != (n, classes) or p.dtype.kind not in "fiu"
                or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1
                or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
            location = file.at if isinstance(file, zipfile.Path) else file
            raise ValueError(f"{location}: expected finite probabilities [N, {classes}] with rows summing to 1")
        return p

    labels = {}
    for split in SPLITS:
        with (base / "gt" / f"{split}.npy").open("rb") as stream:
            y = np.load(stream, allow_pickle=False)
        if (y.ndim != 1 or not len(y) or y.dtype.kind not in "biu"
                or y.min() < 0 or y.max() >= classes):
            raise ValueError(f"{split}: expected nonempty integer class labels [N] in [0, {classes})")
        labels[split] = y

    def stack(split: str) -> np.ndarray:
        cols = [probabilities(base / split / f"{m}.npy", len(labels[split])) for m in model_ids]
        return np.stack(cols, axis=1)  # [N, K, C]

    def oracle(split: str) -> np.ndarray | None:
        path = base / "oracle" / f"{split}.npy"
        return probabilities(path, len(labels[split])) if path.exists() else None

    return Dataset(
        name=meta["name"],
        task=meta["task"],
        n_classes=meta["n_classes"],
        model_ids=model_ids,
        fit_preds=stack("fit"),
        fit_y=labels["fit"],
        test_preds=stack("test"),
        test_y=labels["test"],
        fit_oracle=oracle("fit"),
        test_oracle=oracle("test"),
        oracle_description=meta.get("oracle_description", ""),
        expected_winners=meta.get("expected_winners", []),
        meta=meta.get("knobs", {}),
    )
