"""Versioned, offline FusionBench reference samples for interoperability checks."""

from __future__ import annotations

import hashlib
import io
import json
from importlib.resources import files
from zipfile import Path as ZipPath
from zipfile import ZipFile

from fusionbench.store import _read_dataset_dir
from fusionbench.types import Dataset

_VERSION = "reference-v1"
_MANIFEST_DIGEST = "ab310db529e7316fc0507b6e971de4b72d550fa9f6444175f1b386cdac75b180"


def load_reference_dataset(scenario_id: str) -> Dataset:
    """Read the published 128-row fit and test banks for a canonical scenario ID.

    The packaged banks store each generator's seed-1729 draw in the native
    dataset format (float32 probabilities and int64 class labels). The returned
    arrays are independent on each call. Metadata contains the bank version and
    archive SHA-256 for reproducing comparisons.
    """
    version, digest = _VERSION, _MANIFEST_DIGEST
    if scenario_id == "similar_classes_38":
        version = "reference-v2"
        digest = "27ec27bb0cd031048c15f029d170061f8a462b0d74ac364217df8f71fc1307ed"
    directory = files("fusionbench").joinpath("data", version)
    raw_manifest = directory.joinpath("manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != digest:
        raise ValueError(f"{version} manifest checksum mismatch")
    manifest = json.loads(raw_manifest)
    if scenario_id not in manifest["banks"]:
        raise ValueError(f"Unknown {_VERSION} scenario {scenario_id!r}; choose from {', '.join(manifest['banks'])}")
    expected_hash = manifest["banks"][scenario_id]
    data = directory.joinpath(f"{scenario_id}.zip").read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"{scenario_id}: reference archive checksum mismatch")
    # No member is extracted. _read_dataset_dir validates the same labels,
    # probabilities and model IDs as load_dataset() on ordinary files.
    with ZipFile(io.BytesIO(data)) as archive:
        ds = _read_dataset_dir(ZipPath(archive))
    ds.meta["reference_bank"] = version
    ds.meta["reference_sha256"] = expected_hash
    return ds
