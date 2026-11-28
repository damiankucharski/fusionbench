"""Fold isolation, fallback and the one-full-model confirmation interface."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import r4


@pytest.mark.parametrize("family", ["simplex", "regularized", "tie"])
def test_shared_oof_fits_and_single_deploy(monkeypatch, family):
    labels = np.tile([0, 1], 16)
    ids = np.arange(len(labels))
    x = np.zeros((len(labels), 2, 2))
    # In this fixture, columns hold row identity and a perfectly predictive sign.
    x[:, 0, 0] = ids
    x[:, 1, 1] = labels
    calls = []

    def fit(name, x_train, y_train):
        train = x_train[:, 0, 0].astype(int)
        np.testing.assert_equal(y_train, labels[train])
        calls.append((name, len(train)))

        def predict(held):
            rows = held[:, 0, 0].astype(int)
            assert not np.intersect1d(train, rows).size
            p = np.eye(2)[held[:, 1, 1].astype(int)]
            return p if name == family or family == "tie" else 1 - p

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(r4, "_fit", fit)
    chosen = "simplex" if family == "tie" else family
    models, info = r4.fit_variants(x, labels)
    assert Counter(calls) == {
        ("simplex", 24): 4,
        ("regularized", 24): 4,
        ("simplex", 32): 1,
        ("regularized", 32): 1,
    }
    assert models["cf_margin"] is models["cf_gain"] is models[chosen]
    assert info["selected_family"]["cf_margin"] == chosen
    assert info["oof_gain"] == {"simplex": -2, "regularized": 2, "tie": 0}[family]
    assert info["oof_se"] == 0
    json.dumps(info, allow_nan=False)

    calls.clear()
    _, one = r4.fit_variant(x, labels, "cf_margin")
    assert one["fold_id_sha256"] == info["fold_id_sha256"]
    assert one["full_fits"][chosen] == 1 and sum(one["full_fits"].values()) == 1
    assert len(calls) == 9
    calls.clear()
    _, direct = r4.fit_variant(x, labels, chosen)
    assert calls == [(chosen, 32)] and sum(direct["oof_fits"].values()) == 0


def test_minority_singleton_stays_in_one_held_fold():
    cases = [([0] * 11 + [1], "stratified"), ([0, 1, 2, 3], "kfold")]
    for labels, expected in cases:
        labels = np.array(labels)
        kind, folds = r4._folds(labels)
        assert kind == expected
        held = [valid for _, valid in folds]
        np.testing.assert_array_equal(
            np.sort(np.concatenate(held)), np.arange(len(labels))
        )
        assert all(len(v) for v in held)


def test_one_se_retains_simplex_when_small_positive_gain_is_noisy(monkeypatch):
    y = np.tile([0, 1], 16)
    x = np.zeros((len(y), 2, 2))
    x[:, 0, 0] = np.arange(len(y))

    def fit(name, x_train, y_train):
        def predict(held):
            ids = held[:, 0, 0].astype(int)
            pred = np.full((len(held), 2), 0.5)
            pick = ids < 3
            accurate = (ids[pick] != 2) if name == "regularized" else (ids[pick] == 2)
            pred[pick] = np.eye(2)[np.where(accurate, y[ids[pick]], 1 - y[ids[pick]])]
            return pred

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(r4, "_fit", fit)
    models, info = r4.fit_variants(x, y)
    assert 0 < info["oof_gain"] < info["oof_se"]
    assert models["cf_gain"] is models["regularized"]
    assert models["cf_margin"] is models["simplex"]
