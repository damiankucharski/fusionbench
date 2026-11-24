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
