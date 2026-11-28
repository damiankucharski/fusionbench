import numpy as np
import pandas as pd
import pytest

from fusionbench.cli import main
from fusionbench.store import load_dataset, save_dataset
from fusionbench.types import Dataset


def probabilistic_dataset(metric):
    labels = np.array([0, 1, 0, 1, 1, 0])
    sharp = np.eye(2)[labels] * 0.8 + 0.1
    # The weak base has the wrong class on half the data.
    weak = np.array([[0.7, 0.3]] * len(labels))
    preds = np.stack([sharp, weak], axis=1)
    return Dataset(
        "external",
        "binary",
        2,
        ["sharp", "weak"],
        preds,
        labels,
        preds.copy(),
        labels.copy(),
        None,
        None,
        meta={"primary_metric": metric},
    )


@pytest.mark.parametrize("metric", ["log_loss", "accuracy"])
def test_external_cli_without_oracle(tmp_path, metric):
    folder = save_dataset(probabilistic_dataset(metric), tmp_path)
    output = tmp_path / "results"
    main(
        [
            "evaluate",
            "--dataset",
            str(folder),
            "--methods",
            "best_single",
            "simple_average",
            "--out",
            str(output),
        ]
    )
    rows = pd.read_csv(output / "results.csv")
    best = pd.read_csv(output / "winners.csv").iloc[0]
    assert best.best_method == "best_single"  # first method wins tied accuracy
    expected = (
        rows.test_primary.max() if metric == "accuracy" else rows.test_primary.min()
    )
    assert best.best_test_score == pytest.approx(expected)
    assert rows.regret.isna().all()


def test_overwrite_removes_obsolete_oracle(tmp_path):
    ds = probabilistic_dataset("log_loss")
    ds.test_oracle = ds.test_preds[:, 0, :]
    folder = save_dataset(ds, tmp_path)
    ds.test_oracle = None
    save_dataset(ds, tmp_path)
    assert load_dataset(folder).test_oracle is None


@pytest.mark.parametrize("bad", [np.nan, -0.5, 0.8])
def test_invalid_external_probabilities_identify_file(tmp_path, bad):
    folder = save_dataset(probabilistic_dataset("accuracy"), tmp_path)
    file = folder / "test/sharp.npy"
    matrix = np.load(file)
    matrix[0, 0] = bad
    np.save(file, matrix)
    with pytest.raises(ValueError, match="sharp.npy"):
        load_dataset(folder)


def test_failed_method_keeps_diagnostic_report(tmp_path, monkeypatch):
    from fusionbench.methods.registry import FACTORIES
    from fusionbench.methods.stacking import StackingLR

    class FailingStacker(StackingLR):
        def fit(self, *args, **kwargs):
            raise ValueError("failure during fitting")

    monkeypatch.setitem(FACTORIES, "stacking_lr", FailingStacker)
    ds = probabilistic_dataset("accuracy")
    folder = save_dataset(ds, tmp_path)
    output = tmp_path / "results"
    with pytest.raises(SystemExit, match="stacking_lr"):
        main(
            [
                "evaluate",
                "--dataset",
                str(folder),
                "--methods",
                "stacking_lr",
                "--out",
                str(output),
            ]
        )
    assert "Failed methods" in (output / "report.md").read_text()
    failed = pd.read_csv(output / "results.csv")
    assert failed.loc[0, "error"]
    assert pd.read_csv(output / "winners.csv").empty
