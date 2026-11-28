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
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import numpy as np

from fusionbench.eval.metrics import METRIC_NAMES
from fusionbench.types import Dataset, _numpy_json

try:
    import fcntl
except ImportError:  # Windows does not support flock on directory descriptors.
    fcntl = None

SPLITS = ("fit", "test")


_store_lock = RLock()


@contextmanager
def _directory_lock(directory, *, writing):
    # Lock the existing parent directory so read-only datasets need no sidecar.
    # Shared reads may proceed together while complete banks are being staged.
    if fcntl is None:
        with _store_lock:
            yield
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if writing else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)


def _file_name(name):
    return (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and Path(name).name == name
        and "\\" not in name
        and "\0" not in name
    )


def _metadata(ds):
    if (
        not isinstance(ds.meta, dict)
        or ds.meta.get("primary_metric", "macro_f1") not in METRIC_NAMES
    ):
        raise ValueError(
            "Dataset metadata must be a mapping with a known primary metric"
        )
    return json.dumps(
        {
            "name": ds.name,
            "task": ds.task,
            "n_classes": ds.n_classes,
            "model_ids": ds.model_ids,
            "oracle_description": ds.oracle_description,
            "expected_winners": ds.expected_winners,
            "has_oracle": ds.test_oracle is not None,
            "n_fit": len(ds.fit_y),
            "n_test": len(ds.test_y),
            "format": "npy",
            "knobs": ds.meta,
        },
        indent=2,
        allow_nan=False,
        default=_numpy_json,
    )


def save_dataset(ds: Dataset, root: str | Path) -> Path:
    """Write the complete bank under root/name and return that directory.

    Array conversion and serialization finish in a staging directory before
    updating any published files. Replacing a bank keeps unrelated sidecars.
    """
    ds.validate()
    if not all(_file_name(part) for part in (ds.name, *ds.model_ids)):
        raise ValueError("Dataset and provider names must be single file names")
    metadata = _metadata(ds)
    base = Path(root) / ds.name
    base.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fusionbench-", dir=base.parent) as tmp:
        staged = Path(tmp) / "data"
        staged.mkdir()
        for folder in (*SPLITS, "gt", "oracle"):
            (staged / folder).mkdir()
        for split in SPLITS:
            preds, y = getattr(ds, split + "_preds"), getattr(ds, split + "_y")
            np.save(staged / "gt" / f"{split}.npy", y.astype(np.int64, copy=False))
            for k, name in enumerate(ds.model_ids):
                np.save(
                    staged / split / f"{name}.npy",
                    preds[:, k, :].astype(np.float32, copy=False),
                )
            oracle = getattr(ds, split + "_oracle")
            if oracle is not None:
                np.save(
                    staged / "oracle" / f"{split}.npy",
                    oracle.astype(np.float32, copy=False),
                )
        (staged / "meta.json").write_text(metadata, encoding="utf-8")
        with _directory_lock(base.parent, writing=True):
            if not base.exists():
                staged.rename(base)
            else:
                _replace_files(staged, base, Path(tmp) / "old")
    return base


def _replace_files(staged, base, backup):
    """Install staged files, undoing completed moves on a failed replacement."""
    paths = [p.relative_to(staged) for p in staged.rglob("*.npy")]
    # Drop obsolete published providers, leaving other files in the bank alone.
    manifest = base / "meta.json"
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            previous = {}
        old_ids = previous.get("model_ids", []) if isinstance(previous, dict) else []
        for model in old_ids if isinstance(old_ids, list) else ():
            if _file_name(model):
                for split in SPLITS:
                    rel = Path(split) / f"{model}.npy"
                    if (
                        not (staged / rel).exists()
                        and (base / rel).exists()
                        and rel not in paths
                    ):
                        paths.append(rel)
    # Old optional oracles must disappear if a replacement has none.
    for split in SPLITS:
        oracle = Path("oracle") / f"{split}.npy"
        if not (staged / oracle).exists() and (base / oracle).exists():
            paths.append(oracle)
    paths.append(Path("meta.json"))
    old, installed = [], []
    try:
        for rel in paths:
            dest, prev, src = base / rel, backup / rel, staged / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not dest.is_file():
                    raise ValueError(f"Expected a dataset file at {dest}")
                prev.parent.mkdir(parents=True, exist_ok=True)
                dest.replace(prev)
                old.append(rel)
            if src.exists():
                src.replace(dest)
                installed.append(rel)
    except BaseException:
        for rel in reversed(installed):
            (base / rel).unlink()
        for rel in reversed(old):
            (backup / rel).replace(base / rel)
        raise


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset previously written by :func:`save_dataset`."""
    directory = Path(path)
    with _directory_lock(directory.parent, writing=False):
        return _read_dataset_dir(directory)


def _read_dataset_dir(base: Path | zipfile.Path) -> Dataset:
    """Read the native directory layout from disk or an open reference archive."""
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    model_ids = meta["model_ids"]

    if (
        not isinstance(model_ids, list)
        or not model_ids
        or any(not _file_name(m) for m in model_ids)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise ValueError("model_ids must be a nonempty list of unique file names")
    classes = meta["n_classes"]
    if (
        not isinstance(classes, int)
        or classes < 2
        or meta["task"] not in ("binary", "multiclass")
        or (meta["task"] == "binary" and classes != 2)
    ):
        raise ValueError(
            "task and n_classes must describe binary or multiclass probabilities"
        )
    if (
        not isinstance(meta.get("knobs", {}), dict)
        or meta.get("knobs", {}).get("primary_metric", "macro_f1") not in METRIC_NAMES
    ):
        raise ValueError("unknown primary metric in meta.json")

    def probabilities(file, n):
        with file.open("rb") as stream:
            p = np.load(stream, allow_pickle=False)
        if (
            p.shape != (n, classes)
            or p.dtype.kind not in "fiu"
            or not np.isfinite(p).all()
            or p.min() < 0
            or p.max() > 1
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)
        ):
            location = file.at if isinstance(file, zipfile.Path) else file
            raise ValueError(
                f"{location}: expected finite probabilities [N, {classes}] with rows summing to 1"
            )
        return p

    labels = {}
    for split in SPLITS:
        with (base / "gt" / f"{split}.npy").open("rb") as stream:
            y = np.load(stream, allow_pickle=False)
        if (
            y.ndim != 1
            or not len(y)
            or y.dtype.kind not in "biu"
            or y.min() < 0
            or y.max() >= classes
        ):
            raise ValueError(
                f"{split}: expected nonempty integer class labels [N] in [0, {classes})"
            )
        if f"n_{split}" in meta and meta[f"n_{split}"] != len(y):
            raise ValueError(f"{split}: row count disagrees with meta.json")
        labels[split] = y

    def stack(split: str) -> np.ndarray:
        cols = [
            probabilities(base / split / f"{m}.npy", len(labels[split]))
            for m in model_ids
        ]
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
