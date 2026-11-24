"""Published bank integrity, validation and compatibility with live generators."""

import io
import json
from importlib.resources import files
from zipfile import Path as ZipPath
from zipfile import ZipFile

import numpy as np
import pytest

from fusionbench import reference
from fusionbench.scenarios.registry import SCENARIOS
from fusionbench.store import _read_dataset_dir

DATA = files("fusionbench").joinpath("data", "reference-v1")


def test_frozen_banks_cover_the_v1_suite_and_match_generators():
    spec = json.loads(DATA.joinpath("manifest.json").read_text())
    assert spec["n_per_split"] == 128 and spec["seed"] == 1729
    assert len(spec["banks"]) == 19
    for scenario_id, digest in spec["banks"].items():
        frozen = reference.load_reference_dataset(scenario_id)
        live = SCENARIOS[scenario_id]().build(n_per_split=128, seed=1729)
        assert frozen.model_ids == live.model_ids
        assert frozen.task == live.task
        assert frozen.meta["scenario_id"] == scenario_id
        assert frozen.meta["reference_sha256"] == digest
        assert frozen.meta["reference_bank"] == "reference-v1"
        for split in ("fit", "test"):
            y = getattr(frozen, f"{split}_y")
            assert y.dtype == np.int64
            np.testing.assert_array_equal(y, getattr(live, f"{split}_y"))
            for field in ("preds", "oracle"):
                arr = getattr(frozen, f"{split}_{field}")
                source = getattr(live, f"{split}_{field}")
                if source is None:
                    assert arr is None
                else:
                    assert arr.dtype == np.float32
                    np.testing.assert_allclose(arr, source, atol=1e-6, rtol=1e-6)


def test_unknown_id_does_not_resolve_a_path():
    with pytest.raises(ValueError, match="Unknown reference-v1"):
        reference.load_reference_dataset("../committee")


@pytest.mark.parametrize("filename,error", [
    ("manifest.json", "manifest checksum"),
    ("committee.zip", "archive checksum"),
])
def test_reference_rejects_modified_bytes(tmp_path, monkeypatch, filename, error):
    bank_root = tmp_path / "data" / "reference-v1"
    bank_root.mkdir(parents=True)
    (bank_root / "manifest.json").write_bytes(DATA.joinpath("manifest.json").read_bytes())
    (bank_root / filename).write_bytes(b"corrupt")
    monkeypatch.setattr(reference, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match=error):
        reference.load_reference_dataset("committee")


def test_archived_native_reader_validates_provider_ids():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("meta.json", json.dumps({"model_ids": ["../outside"]}))
    with ZipFile(buffer) as archive, pytest.raises(ValueError, match="model_ids"):
        _read_dataset_dir(ZipPath(archive))


def test_reference_bank_for_similar_classes_matches_live_draw():
    from fusionbench import generate_dataset

    frozen = reference.load_reference_dataset("similar_classes_38")
    live = generate_dataset("similar_classes_38", n_per_split=128, seed=1729)
    assert frozen.meta["reference_bank"] == "reference-v2"
    for split in ("fit", "test"):
        np.testing.assert_array_equal(getattr(frozen, split + "_y"), getattr(live, split + "_y"))
        np.testing.assert_array_equal(getattr(frozen, split + "_preds"), getattr(live, split + "_preds").astype(np.float32))
