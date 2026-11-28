"""Rebuild an offline reference bank collection from its frozen scenario IDs.

From the source checkout, run
    PYTHONPATH=. python tools/build_reference.py --out /tmp/reference-v1
    PYTHONPATH=. python tools/build_reference.py --version reference-v3 --out /tmp/reference-v3

The original build used the project lock on Linux x86-64, Python 3.12.14,
NumPy 2.5.0 and SciPy 1.18.0. Use that environment and the v1 source revision
to reproduce existing checksums after generator or library upgrades.

The archive paths, NPY encoding, member order, file modes and timestamps form
part of the versioned fixture. ZIP_STORED avoids compression-library dependence.
The wheel compresses the resulting assets. Compare manifest SHA-256 to the
digest in fusionbench/reference.py before publishing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from fusionbench import generate_dataset
from fusionbench.store import save_dataset

# Frozen inventory. Future additions to the live registry do not alter v1.
SCENARIO_IDS = (
    "calibrated_suppressor", "committee", "compositional_switch", "compound",
    "correlated", "deadweight", "gated_experts", "hardrescue", "hardveto",
    "miscalibration", "multiclass_committee", "native_close_switch",
    "native_switch", "needle", "parity", "polarity_fault", "saturated",
    "two_pathways", "weakest_link",
)
SEED = 1729
ROWS = 128


def build(out: Path, version="reference-v1") -> str:
    out.mkdir(parents=True, exist_ok=True)
    banks = {}
    ids = {
        "reference-v1": SCENARIO_IDS,
        "reference-v2": ("similar_classes_38",),
        "reference-v3": ('rare_specialist', 'redundant_pool', 'environment_shortcut', 'selective_fallback'),
        "reference-v4": ("cloned_shortcut",),
        "reference-v5": ("rare_shortcut_noise", "stratified_pool"),
        "reference-v6": ("grouped_rare_event", "shared_error_bloc", "rare_scene_specialist"),
    }[version]
    for scenario_id in ids:
        dataset = generate_dataset(scenario_id, n_per_split=ROWS, seed=SEED)
        target = out / f"{scenario_id}.zip"
        with TemporaryDirectory() as temp:
            directory = save_dataset(dataset, temp)
            with ZipFile(target, "w", compression=ZIP_STORED) as archive:
                for file in sorted(directory.rglob("*")):
                    if file.is_file():
                        member = ZipInfo(file.relative_to(directory).as_posix())
                        member.create_system = 3
                        member.external_attr = 0o100644 << 16
                        archive.writestr(member, file.read_bytes())
        banks[scenario_id] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "bank_version": version,
        "seed": SEED,
        "n_per_split": ROWS,
        "banks": banks,
    }
    serialized = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (out / "manifest.json").write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", choices=("reference-v1", "reference-v2", "reference-v3", "reference-v4", "reference-v5", "reference-v6"), default="reference-v1")
    args = parser.parse_args()
    print(build(args.out, args.version))
