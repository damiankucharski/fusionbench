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
from pathlib import Path

import numpy as np

from fusionbench.types import Dataset

SPLITS = ("fit", "test")


def save_dataset(ds: Dataset, root: str | Path) -> Path:
    """Write ``ds`` under ``root/<name>/`` and return that directory."""
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
        if oracles[split] is not None:
            np.save(base / "oracle" / f"{split}.npy", oracles[split].astype(np.float32))

    meta = {
        "name": ds.name,
        "task": ds.task,
        "n_classes": ds.n_classes,
        "model_ids": ds.model_ids,
        "oracle_description": ds.oracle_description,
        "expected_winners": ds.expected_winners,
        "has_oracle": ds.fit_oracle is not None,
        "n_fit": int(ds.fit_y.shape[0]),
        "n_test": int(ds.test_y.shape[0]),
        "format": "npy",
        "knobs": ds.meta,
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    return base


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset previously written by :func:`save_dataset`."""
    base = Path(path)
    meta = json.loads((base / "meta.json").read_text())
    model_ids = meta["model_ids"]

    def stack(split: str) -> np.ndarray:
        cols = [np.load(base / split / f"{m}.npy") for m in model_ids]
        return np.stack(cols, axis=1)  # [N, K, C]

    def oracle(split: str) -> np.ndarray | None:
        path = base / "oracle" / f"{split}.npy"
        return np.load(path) if path.exists() else None

    return Dataset(
        name=meta["name"],
        task=meta["task"],
        n_classes=meta["n_classes"],
        model_ids=model_ids,
        fit_preds=stack("fit"),
        fit_y=np.load(base / "gt" / "fit.npy"),
        test_preds=stack("test"),
        test_y=np.load(base / "gt" / "test.npy"),
        fit_oracle=oracle("fit"),
        test_oracle=oracle("test"),
        oracle_description=meta.get("oracle_description", ""),
        expected_winners=meta.get("expected_winners", []),
        meta=meta.get("knobs", {}),
    )
