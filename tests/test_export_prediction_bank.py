import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from fusionbench import load_dataset

STUDY = Path(__file__).resolve().parents[1] / "docs" / "ensemble-study" / "external-coverage"
SOURCE = STUDY / "external-results"


@pytest.mark.parametrize("suite,seed", [("digits", 3140000), ("digits_lr_bag", 3140000), ("wine", 3150000)])
def test_published_bank_roundtrip(tmp_path, suite, seed):
    bank = f"{suite}_{seed}"
    script = STUDY / "export_bank.py"
    if suite == "digits":
        output = subprocess.check_output(
            [sys.executable, str(script), "--suite", suite, "--seed", str(seed), "--out", str(tmp_path)], cwd=tmp_path, text=True
        )
        folder = Path(output.strip())
    else:
        folder = run_path(str(script))["export_bank"](suite, seed, tmp_path)
    assert folder == tmp_path / bank
    ds = load_dataset(folder)
    records = json.loads((SOURCE / "fit_manifest.json").read_text())["banks"]
    original = next(r for r in records if (r["suite"], r["seed"]) == (suite, seed))
    assert ds.model_ids == original["providers"] and ds.n_classes == original["n_classes"]
    assert ds.meta["seed"] == seed and ds.meta["primary_metric"] == "brier"
    for split in ("fit", "test"):
        with np.load(SOURCE / f"{split}_banks.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(getattr(ds, split + "_preds"), archive[bank + "_inputs"])
            np.testing.assert_array_equal(getattr(ds, split + "_y"), archive[bank + "_labels"])
            np.testing.assert_array_equal(ds.meta[split + "_row_ids"], archive[bank + "_rows"])


def test_runner_up_seeds_bank_from_separate_archives(tmp_path):
    study = STUDY.parent / "runner-up-transfer"
    fit, test = study / "external-fit", study / "external-results"
    name = "seeds_3320000"
    output = subprocess.check_output(
        [
            sys.executable, str(STUDY / "export_bank.py"), "--suite", "seeds", "--seed", "3320000",
            "--source", str(fit), "--test-source", str(test), "--out", str(tmp_path),
        ],
        cwd=tmp_path,
        text=True,
    )
    assert Path(output.strip()) == tmp_path / name
    ds = load_dataset(tmp_path / name)
    manifest = json.loads((fit / "fit_manifest.json").read_text())
    bank = next(r for r in manifest["banks"] if (r["suite"], r["seed"]) == ("seeds", 3320000))
    assert ds.model_ids == bank["providers"]
    assert ds.meta["source"] == "runner-up-transfer" and ds.meta["primary_metric"] == "brier"
    assert ds.meta["seeds_source"] == manifest["seeds_source"]
    assert all(ds.meta[k] == value for k, value in bank.items())
    for split, source in (("fit", fit), ("test", test)):
        assert getattr(ds, split + "_preds").shape == (bank["n_" + split], bank["n_providers"], bank["n_classes"])
        with np.load(source / f"{split}_banks.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(getattr(ds, split + "_preds"), archive[name + "_inputs"])
            np.testing.assert_array_equal(getattr(ds, split + "_y"), archive[name + "_labels"])
            np.testing.assert_array_equal(ds.meta[split + "_row_ids"], archive[name + "_rows"])
