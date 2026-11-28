"""Verify bank provenance and row reuse without decoding provider predictions.

From the project root, run python docs/ensemble-study/validation-curse/real/verify_real.py
to print the audit. Optionally supply --seeds-zip to verify the original UCI file too.
"""

import argparse
import hashlib
import io
import json
from itertools import combinations, product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
OLD = STUDY / "external-coverage" / "external-results"
NEW = STUDY / "runner-up-transfer"
HASHES = {
    OLD
    / "fit_banks.npz": "163b70904a03c314348de2652f81ac61909d62fa246c13ba760cd20df0fae886",
    OLD
    / "test_banks.npz": "835bce8241dd773683174e60d7882c4d7ebd94df04fa3a0757f788df793aac83",
    OLD
    / "fit_manifest.json": "bd273d64e36fabb14c0276cfccdaaef60180353015aa1cb8e3441ecfb2f701c5",
    NEW
    / "external-fit/fit_banks.npz": "7caa774d7bb2022173838d12f9d45222fcbc9da196e511e6df4e185cb9134d3e",
    NEW
    / "external-results/test_banks.npz": "c01b4cb376af61c740997c859afe556905d4e628ec733c6cdde1439c7783f97c",
    NEW
    / "external-fit/fit_manifest.json": "c1d8343af41822188ccbb11c227a095e013a0278d2381bc4a029a87d332970a6",
}
FIRST_SEED = {
    "digits_A": 3_140_000,
    "digits_lr_bag": 3_140_000,
    "wine": 3_150_000,
    "digits_B": 3_300_000,
    "iris": 3_310_000,
    "seeds": 3_320_000,
}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def subsets(labels, seed, n_base, n_fit):
    ids = np.arange(len(labels), dtype=np.int64)
    base, rest = train_test_split(
        ids, stratify=labels, train_size=n_base, random_state=seed
    )
    fit, test = train_test_split(
        rest, stratify=labels[rest], train_size=n_fit, random_state=seed + 1
    )
    np.testing.assert_array_equal(np.sort(np.concatenate([base, fit, test])), ids)
    return base, fit, test


def summarize_overlap(pairs):
    overlaps = [len(left & right) for left, right in pairs]
    return {
        "pairs": len(overlaps),
        "nonempty_intersections": sum(x > 0 for x in overlaps),
        "min": min(overlaps),
        "mean": sum(overlaps) / len(overlaps),
        "max": max(overlaps),
    }


def audit(seeds_zip=None):
    hashes = {}
    for path, expected in HASHES.items():
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"File hash mismatch at {path}")
        hashes[str(path.relative_to(STUDY))] = actual

    # The original ordered Seeds label vector has three contiguous blocks. Verify its
    # full digest against the signed-off fit manifest in every Seeds partition below.
    labels_by_dataset = {
        "digits": load_digits().target,
        "wine": load_wine().target,
        "iris": load_iris().target,
        "seeds": np.repeat(np.arange(3, dtype=np.int64), 70),
    }
    if seeds_zip:
        source = json.loads((NEW / "external-fit/fit_manifest.json").read_text())[
            "seeds_source"
        ]
        archive = Path(seeds_zip).read_bytes()
        assert hashlib.sha256(archive).hexdigest() == source["archive_sha256"]
        with ZipFile(io.BytesIO(archive)) as data:
            content = data.read("seeds_dataset.txt")
        assert hashlib.sha256(content).hexdigest() == source["seeds_dataset_txt_sha256"]
        np.testing.assert_array_equal(
            np.loadtxt(io.BytesIO(content))[:, -1] - 1, labels_by_dataset["seeds"]
        )

    per_pool = {pool: {} for pool in FIRST_SEED}
    for fit_dir, test_dir, prefix in (
        (OLD, OLD, "A"),
        (NEW / "external-fit", NEW / "external-results", "B"),
    ):
        banks = json.loads((fit_dir / "fit_manifest.json").read_text())["banks"]
        with (
            np.load(fit_dir / "fit_banks.npz", allow_pickle=False) as fit_data,
            np.load(test_dir / "test_banks.npz", allow_pickle=False) as test_data,
        ):
            assert len(banks) == 24
            for bank in banks:
                suite, seed = bank["suite"], bank["seed"]
                dataset = "digits" if suite.startswith("digits") else suite
                pool = f"digits_{prefix}" if suite == "digits" else suite
                key = f"{suite}_{seed}"
                source = labels_by_dataset[dataset].astype(np.int64, copy=False)
                if "source_labels_sha256" in bank:
                    assert digest(source) == bank["source_labels_sha256"]
                n_base, n_fit, n_test = [
                    bank.get(f"n_{part}", bank.get(f"{part}_rows"))
                    for part in ("base", "fit", "test")
                ]
                base, fit, test = subsets(source, seed, n_base, n_fit)
                assert len(test) == n_test
                for part, ids in (("base", base), ("fit", fit), ("test", test)):
                    assert digest(ids) == bank[f"{part}_rows_sha256"]
                for data, ids in ((fit_data, fit), (test_data, test)):
                    np.testing.assert_array_equal(data[key + "_rows"], ids)
                    np.testing.assert_array_equal(data[key + "_labels"], source[ids])
                assert digest(fit_data[key + "_labels"]) == bank["fit_labels_sha256"]
                assert seed not in per_pool[pool]
                rng = np.random.default_rng([seed, 96301])
                selected = (
                    fit[np.sort(rng.choice(n_fit, 64, replace=False))]
                    if n_fit > 64
                    else fit
                )
                per_pool[pool][seed] = (base, fit, selected, test)

    summaries = {}
    test_sets = {}
    for pool, by_seed in per_pool.items():
        seeds = range(FIRST_SEED[pool], FIRST_SEED[pool] + 8)
        assert sorted(by_seed) == list(seeds)
        records = [by_seed[seed] for seed in seeds]
        sizes = list(map(len, records[0]))
        assert all(list(map(len, record)) == sizes for record in records)
        test_sets[pool] = [set(record[-1].tolist()) for record in records]
        counts = np.unique(
            np.concatenate([record[-1] for record in records]), return_counts=True
        )[1]
        summaries[pool] = {
            "partitions": len(records),
            "n_base_fit_small_test": sizes,
            "test_positions": sum(map(len, test_sets[pool])),
            "distinct_test_ids": len(counts),
            "ids_tested_more_than_once": int(sum(counts > 1)),
            "max_tests_per_id": int(max(counts)),
            "test_intersections_across_partitions": summarize_overlap(
                combinations(test_sets[pool], 2)
            ),
        }
    for seed, rows in per_pool["digits_A"].items():
        for left, right in zip(rows, per_pool["digits_lr_bag"][seed], strict=True):
            np.testing.assert_array_equal(left, right)

    return {
        "verified_files_sha256": hashes,
        "verified_banks": sum(len(records) for records in per_pool.values()),
        "six_pool_counts": summaries,
        "digits_A_bag_equal_ordered_base_fit_small_test": 8,
        "digits_A_or_bag_vs_B_test_intersections": summarize_overlap(
            product(test_sets["digits_A"], test_sets["digits_B"])
        ),
        "digits_A_or_bag_and_B_union_intersection": len(
            set.union(*test_sets["digits_A"]) & set.union(*test_sets["digits_B"])
        ),
        "all_digits_distinct_test_ids": len(
            set.union(*test_sets["digits_A"], *test_sets["digits_B"])
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds-zip", type=Path)
    print(json.dumps(audit(parser.parse_args().seeds_zip), indent=2))
