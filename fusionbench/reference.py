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

_MANIFEST_DIGESTS = {
    "reference-v1": "ab310db529e7316fc0507b6e971de4b72d550fa9f6444175f1b386cdac75b180",
    "reference-v2": "27ec27bb0cd031048c15f029d170061f8a462b0d74ac364217df8f71fc1307ed",
    "reference-v3": "0fd0fe41c0770d6a27b622e702c6624f0911da0367a0698b40144e5622798df5",
    "reference-v4": "a1327152ffd02702008ff5635d4b375195c8400491a01257df0b52e2ace77468",
    "reference-v5": "7ea9db3455cefec627a3a0117a6d532648f07b4ea074db57c7a99bc21967c12d",
    "reference-v6": "fffde9e2d6793b40b512ee9d05443730a10e869a2498f280077be6556e1ba70c",
}


def load_reference_dataset(scenario_id: str) -> Dataset:
    """Read the published 128-row fit and test banks for a canonical scenario ID.

    The packaged banks store each generator's seed-1729 draw in the native
    dataset format (float32 probabilities and int64 class labels). The returned
    arrays are independent on each call. Metadata contains the bank version and
    archive SHA-256 for reproducing comparisons.
    """
    available = []
    for version, digest in _MANIFEST_DIGESTS.items():
        directory = files("fusionbench").joinpath("data", version)
        raw_manifest = directory.joinpath("manifest.json").read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != digest:
            raise ValueError(f"{version} manifest checksum mismatch")
        banks = json.loads(raw_manifest)["banks"]
        if scenario_id in banks:
            expected_hash = banks[scenario_id]
            break
        available.extend(banks)
    else:
        raise ValueError(
            f"Unknown reference scenario {scenario_id!r}; choose from {', '.join(available)}"
        )
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
