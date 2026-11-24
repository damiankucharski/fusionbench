"""Export one published prediction bank as a standard FusionBench dataset."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().with_name("external-results")
SUITES = ("digits", "digits_lr_bag", "wine")


def export_bank(suite, seed, out, *, source=SOURCE, test_source=None):
    """Use the fit manifest in source and the test archive in test_source (default source)."""
    from fusionbench import Dataset, save_dataset

    source = Path(source)
    test_source = source if test_source is None else Path(test_source)
    manifest = json.loads((source / "fit_manifest.json").read_text())
    records = [r for r in manifest["banks"] if (r["suite"], r["seed"]) == (suite, seed)]
    if len(records) != 1:
        raise ValueError(f"Expected one published prediction bank for {suite} seed {seed}, found {len(records)}")
    record = records[0]
    name = f"{suite}_{seed}"
    # The split-directory release includes UCI Seeds provenance in its manifest.
    # Use this marker so copies of the release retain their original study ID.
    study = "runner-up-transfer" if "seeds_source" in manifest else "external-coverage"
    meta = {"source": study, "primary_metric": "brier", **record}
    if suite == "seeds" and "seeds_source" in manifest:
        meta["seeds_source"] = manifest["seeds_source"]
    arrays, row_ids = {}, {}
    for split, folder in (("fit", source), ("test", test_source)):
        with np.load(folder / f"{split}_banks.npz", allow_pickle=False) as archive:
            inputs, labels, rows = (archive[f"{name}_{key}"] for key in ("inputs", "labels", "rows"))
        count = record[f"n_{split}"] if f"n_{split}" in record else record[f"{split}_rows"]
        if rows.dtype.kind not in "iu" or rows.shape != (count,) or labels.shape != rows.shape:
            raise ValueError(f"Row IDs and labels must match the published {split} count")
        for key, value in (("rows", rows), ("inputs", inputs), ("labels", labels)):
            field = f"{split}_{key}_sha256"
            if field in record and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != record[field]:
                raise ValueError(f"Published hash mismatch for {name} {split} {key}")
        arrays[f"{split}_preds"], arrays[f"{split}_y"] = inputs, labels
        row_ids[f"{split}_row_ids"] = rows.tolist()
    all_ids = row_ids["fit_row_ids"] + row_ids["test_row_ids"]
    if min(all_ids) < 0 or len(set(all_ids)) != len(all_ids):
        raise ValueError("Fit and test row IDs must be distinct nonnegative source indices")

    dataset = Dataset(
        name=name,
        task="binary" if record["n_classes"] == 2 else "multiclass",
        n_classes=record["n_classes"],
        model_ids=record["providers"],
        meta={**meta, **row_ids},
        **arrays,
    )
    return save_dataset(dataset, out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Suite ID in the fit manifest")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path, help="Parent directory for the exported dataset")
    parser.add_argument("--source", type=Path, default=SOURCE, help="Directory containing the published bank archives and fit manifest")
    parser.add_argument("--test-source", type=Path, help="Directory containing test_banks.npz (defaults to --source)")
    args = parser.parse_args()
    print(export_bank(args.suite, args.seed, args.out, source=args.source, test_source=args.test_source))
