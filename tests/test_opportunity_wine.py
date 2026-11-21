import numpy as np
import pytest
from sklearn.datasets import load_wine

from fusionbench.eval import opportunity_wine as wine


def test_original_specimens_and_frozen_partitions():
    x, y = load_wine(return_X_y=True)
    assert x.shape == (178, 13)
    assert np.bincount(y).tolist() == [59, 71, 48]
    assert wine.wine_hashes() == {
        "features": "8edcf3903afd97c64d51e0212eb10b213f7943da650574d1c055b836c5c35d37",
        "labels": "b6514e07a6d27854e0da000ca82d8d8cc3d0cb904c51600344ca34f1ee04884b",
        "splits": "5d5836ef8966d07e338f9c854bc048790d822ee229d16f2ffa4f6f66bc14deca",
    }
    plans = wine.wine_plans()
    np.testing.assert_array_equal(
        np.sort(np.concatenate([p.test_ids for p in plans])), np.arange(178)
    )
    for p in plans:
        expected_sizes = (106, 36, 36) if p.fold < 3 else (107, 36, 35)
        ids = (p.fit_ids, p.validation_ids, p.test_ids)
        assert tuple(map(len, ids)) == expected_sizes
        np.testing.assert_array_equal(np.sort(np.concatenate(ids)), np.arange(178))
        for group in ids:
            assert set(y[group]) == {0, 1, 2}


def test_every_actual_fit_and_query_is_disjoint(monkeypatch):
    x, y = load_wine(return_X_y=True)
    original = {row.tobytes(): i for i, row in enumerate(x)}
    assert len(original) == len(x)
    calls = []

    class RecordingProvider:
        def fit(self, features, labels):
            self.train = [original[row.tobytes()] for row in features]
            np.testing.assert_array_equal(labels, y[self.train])
            self.classes_ = np.unique(labels)
            return self

        def predict_proba(self, features):
            query = [original[row.tobytes()] for row in features]
            calls.append((self.train, query))
            return np.tile([0.2, 0.3, 0.5], (len(query), 1))

    monkeypatch.setattr(
        wine, "_providers", lambda: [RecordingProvider() for _ in wine.PROVIDER_NAMES]
    )
    for fold in range(5):
        calls.clear()
        data = wine.build_wine_fold(fold)
        assert len(calls) == 30
        fit, valid, test = (
            set(s.row_ids) for s in (data.fit, data.validation, data.test)
        )
        for provider in range(6):
            predicted = []
            jobs = calls[provider::6]
            for train, query in jobs[:4]:
                train, query = set(train), set(query)
                assert train.isdisjoint(query)
                assert train | query == fit
                predicted.extend(query)
            assert sorted(predicted) == sorted(fit)
            train, query = map(set, jobs[-1])
            assert train == fit
            assert query == valid | test
            assert train.isdisjoint(query)


def test_real_heldout_and_oof_provider_columns_agree_with_direct_refit():
    x, y = load_wine(return_X_y=True)
    data = wine.build_wine_fold(0)
    for split in [data.fit, data.validation, data.test]:
        np.testing.assert_array_equal(split.y, y[split.row_ids])
        assert split.preds.shape == (len(split.y), 6, 3)
        np.testing.assert_allclose(split.preds.sum(axis=-1), 1, atol=1e-12, rtol=0)
    baseline = wine._providers()[0].fit(x[data.fit.row_ids], y[data.fit.row_ids])
    for held in [data.validation, data.test]:
        np.testing.assert_allclose(
            held.preds[:, 0],
            baseline.predict_proba(x[held.row_ids]),
            atol=1e-13,
            rtol=0,
        )
    index = data.plan.inner_holdout[data.fit.row_ids]
    inner_train = data.fit.row_ids[index != 0]
    lda = wine._providers()[5].fit(x[inner_train], y[inner_train])
    np.testing.assert_allclose(
        data.fit.preds[index == 0, 5],
        lda.predict_proba(x[data.fit.row_ids[index == 0]]),
        atol=1e-13,
        rtol=0,
    )


@pytest.mark.parametrize("bad", [-1, 5])
def test_unknown_fold_rejected(bad):
    with pytest.raises(ValueError, match="Wine fold"):
        wine.build_wine_fold(bad)


def test_additional_arm_uses_archived_inputs_and_writes_a_separate_file(
    monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    import pandas as pd

    class FixedProvider:
        def fit(self, x, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, x):
            return np.tile([0.1, 0.7, 0.2], (len(x), 1))

    monkeypatch.setattr(
        wine, "_providers", lambda: [FixedProvider() for _ in wine.PROVIDER_NAMES]
    )
    wine.write_wine_inputs(tmp_path)
    previous = tmp_path / "wine-runs.csv.gz"
    previous.write_bytes(b"existing primary results")
    calls = []

    def record_call(name, fold, splits, **options):
        calls.append((name, fold, options))
        assert set(splits) == {"fit", "validation", "iid"}
        for preds, labels, oracle in splits.values():
            assert preds.shape == (len(labels), 6, 3)
            assert oracle is None
        return [{"fold": fold}]

    def unexpected_refit():
        pytest.fail("The supplement should reuse the primary-run provider bank")

    monkeypatch.setattr(wine, "_providers", unexpected_refit)
    monkeypatch.setitem(
        sys.modules,
        "fusionbench.eval.mutation_opportunity",
        SimpleNamespace(evaluate_bank=record_call),
    )
    wine.run_wine(tmp_path, independent_refresh=True)
    assert calls == [
        (
            "wine",
            fold,
            {
                "gp_seeds": (661, 673),
                "arms": ("refresh_independent",),
                "include_baselines": False,
            },
        )
        for fold in range(5)
    ]
    assert previous.read_bytes() == b"existing primary results"
    assert pd.read_csv(tmp_path / "wine-refresh-independent-runs.csv.gz")[
        "fold"
    ].tolist() == list(range(5))
