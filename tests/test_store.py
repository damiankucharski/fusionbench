"""Dataset save/load round-trip preserves predictions, labels and oracle."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fusionbench.scenarios.committee import Committee
from fusionbench.store import load_dataset, save_dataset


def test_round_trip(tmp_path):
    ds = Committee(k=5).build(n_per_split=1000, seed=0)
    path = save_dataset(ds, tmp_path)
    back = load_dataset(path)

    assert back.name == ds.name
    assert back.task == ds.task
    assert back.model_ids == ds.model_ids
    np.testing.assert_allclose(back.fit_preds, ds.fit_preds, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(back.test_y, ds.test_y)
    np.testing.assert_allclose(back.test_oracle, ds.test_oracle, rtol=1e-5, atol=1e-6)


def test_rows_sum_to_one():
    ds = Committee(k=4).build(n_per_split=500, seed=0)
    sums = ds.fit_preds.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


@pytest.mark.parametrize("name,provider", [("../escape", "m0"), ("safe", "../escape")])
def test_save_rejects_path_components_before_writing(tmp_path, name, provider):
    ds = Committee(k=1).build(n_per_split=8, seed=1)
    ds.name = name
    ds.model_ids = [provider]
    with pytest.raises(ValueError, match="file names"):
        save_dataset(ds, tmp_path / "output")
    assert not list(tmp_path.iterdir())


def test_replacement_removes_old_bank_files_and_preserves_user_sidecars(tmp_path):
    old = Committee(k=2).build(n_per_split=12, seed=2)
    bank = save_dataset(old, tmp_path)
    sidecar = bank / "fit" / "annotation.npy"
    sidecar.write_bytes(b"user file")
    new = replace(
        old,
        model_ids=["replacement"],
        fit_preds=old.fit_preds[:, :1],
        test_preds=old.test_preds[:, :1],
        fit_oracle=None,
        test_oracle=None,
    )
    save_dataset(new, tmp_path)
    loaded = load_dataset(bank)
    assert loaded.model_ids == ["replacement"]
    assert loaded.test_oracle is loaded.fit_oracle is None
    assert sidecar.read_bytes() == b"user file"
    for split in ("fit", "test"):
        for name in old.model_ids:
            assert not (bank / split / f"{name}.npy").exists()


def test_invalid_metadata_and_interrupted_replacement_restore_the_original(
    tmp_path, monkeypatch
):
    old = Committee(k=2).build(n_per_split=16, seed=2)
    bank = save_dataset(old, tmp_path)
    files = {
        p.relative_to(bank): p.read_bytes() for p in bank.rglob("*") if p.is_file()
    }
    changed = replace(Committee(k=1).build(n_per_split=7, seed=3), name=old.name)
    changed.meta["invalid"] = object()
    with pytest.raises(TypeError, match="as JSON"):
        save_dataset(changed, tmp_path)
    del changed.meta["invalid"]
    original_replace = Path.replace

    def fail_one_install(src, dest):
        if dest == bank / "meta.json" and src.parent.name == "data":
            raise OSError("simulated installation failure")
        return original_replace(src, dest)

    monkeypatch.setattr(Path, "replace", fail_one_install)
    with pytest.raises(OSError, match="simulated"):
        save_dataset(changed, tmp_path)
    assert files == {
        p.relative_to(bank): p.read_bytes() for p in bank.rglob("*") if p.is_file()
    }
    np.testing.assert_array_equal(load_dataset(bank).fit_y, old.fit_y)


def test_numpy_metadata_survives_the_storage_round_trip(tmp_path):
    ds = Committee(k=1).build(n_per_split=4, seed=5)
    ds.meta["weights"] = np.array([0.25, 0.75])
    ds.meta["tag"] = np.int64(2)
    restored = load_dataset(save_dataset(ds, tmp_path))
    assert restored.meta["weights"] == [0.25, 0.75] and restored.meta["tag"] == 2
