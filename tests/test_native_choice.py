import json

import numpy as np
import pytest

from fusionbench.methods import native_choice as routing


def comparison(gain, fold, disagreement, *, best_simple="mean"):
    gain = np.asarray(gain, dtype=float)
    simple_loss = np.full(len(gain), 0.45)
    other = "top_n" if best_simple == "mean" else "mean"
    return routing._summarize(
        {
            "native": simple_loss + gain,
            best_simple: simple_loss,
            other: simple_loss + 0.20,
        },
        np.asarray(fold),
        np.asarray(disagreement),
    )


def test_a_favorable_average_can_hide_a_reversed_fold():
    evidence = comparison(
        [-0.02] * 4 + [0.08] * 8,
        np.repeat(np.arange(3), 4),
        np.arange(12),
        best_simple="top_n",
    )
    assert routing.routes(evidence) == {
        "cv": "top_n",
        "paired": "top_n",
        "sign": "native",
        "stress": "top_n",
        "one_se": "top_n",
    }


def test_agreement_stratum_can_reverse_an_otherwise_consistent_gain():
    evidence = comparison(
        [-0.01] * 12 + [0.08] * 12, np.tile(np.arange(3), 8), np.arange(24)
    )
    choices = routing.routes(evidence)
    assert choices["sign"] == choices["paired"] == "mean"
    assert choices["stress"] == "native"


def test_paired_standard_error_and_simplicity_order():
    # CV prefers the native model slightly and the simpler action is within one SE.
    gain = np.tile([-0.1, 0.095], 6)
    evidence = comparison(gain, np.tile(np.arange(3), 4), np.ones(12))
    assert evidence["paired_se"]["native"]["mean"] == pytest.approx(
        gain.std(ddof=1) / np.sqrt(12)
    )
    assert evidence["unit_stratum"] == [0] * 12
    assert len(evidence["simple_gain_by_stratum"]) == 1
    assert routing.routes(evidence) == {
        "cv": "native",
        "one_se": "mean",
        "sign": "native",
        "stress": "native",
        "paired": "native",
    }

    evidence = routing._summarize(
        dict.fromkeys(routing.ACTIONS, np.zeros(6)),
        np.tile(np.arange(3), 2),
        np.zeros(6),
    )
    assert routing.routes(evidence) == {
        "cv": "mean",
        "paired": "mean",
        "sign": "native",
        "stress": "mean",
        "one_se": "mean",
    }


def test_grouped_selection_shares_fits_and_weights_units_equally(monkeypatch):
    sizes = np.array([10, 2, 1, 4, 1, 1])
    group = np.repeat(np.arange(len(sizes)), sizes)
    # The first provider encodes the scene so the fake model can detect a split scene.
    x = np.column_stack((0.1 + 0.05 * group, 0.60 - 0.06 * group))
    y = (group == 0).astype(int)
    calls = []
    constant = {"mean": 0.3, "top_n": 0.4, "native": 0.2}

    def fit(train, labels, name, *, seed):
        assert seed == 7
        calls.append((name, len(labels)))

        class Predictor:
            def predict(self, held):
                assert not set(held[:, 0, 1]).intersection(train[:, 0, 1])
                return np.tile([1 - constant[name], constant[name]], (len(held), 1))

        return Predictor()

    monkeypatch.setattr(routing, "fit_action", fit)
    evidence = routing.source_evidence(x, y, group, seed=7)
    assert len(calls) == evidence["oof_fits"] == 9
    assert sum(f["held_units"] for f in evidence["folds"]) == len(sizes)
    assert len(np.unique(evidence["unit_fold"])) == 3
    assert evidence["n_units"] == len(sizes)
    per_row = 2 * (constant["mean"] - y) ** 2
    expected = [np.mean(per_row[group == i]) for i in range(len(sizes))]
    np.testing.assert_allclose(evidence["unit_loss"]["mean"], expected)
    assert evidence["oof_brier"]["mean"] == pytest.approx(np.mean(expected))
    assert evidence["oof_brier"]["mean"] != pytest.approx(per_row.mean())
    # Two binary providers: mean squared full-class deviation is half their
    # squared positive-class difference. No label is needed for this measure.
    expected_disagreement = [
        0.5 * (x[group == i, 0][0] - x[group == i, 1][0]) ** 2
        for i in range(len(sizes))
    ]
    np.testing.assert_allclose(evidence["unit_disagreement"], expected_disagreement)
    assert json.loads(json.dumps(evidence)) == evidence


@pytest.mark.parametrize(
    "x,groups", [(np.full((4, 2), 0.5), [0, 0, 1, 1]), (np.full((4, 1), 0.5), None)]
)
def test_inadequate_units_or_single_provider(x, groups):
    evidence = routing.source_evidence(x, [0, 1, 0, 1], groups)
    assert evidence["oof_fits"] == 0
    assert set(routing.routes(evidence).values()) == {"mean"}


def test_final_refit_uses_full_source_and_registered_native_seed():
    pytest.importorskip("okapi.pattern_logit")
    rng = np.random.default_rng(615)
    x = rng.uniform(0.01, 0.99, (48, 3))
    y = (rng.random(48) < x.mean(axis=1)).astype(int)
    model = routing.fit_action(x, y, "native", seed=17)
    assert model.split_seed == 0 and model.seed == 17 and model.gp_validation_top_k == 8
    assert model.counts_["refit_rows"] == len(x)
    assert model.predict(x).shape == (48, 2)


def test_public_helper_refits_exactly_one_selected_action_on_all_source_rows(
    monkeypatch,
):
    calls = []

    def fake_fit(x, y, name, *, seed):
        calls.append((len(y), name, seed))
        return object()

    monkeypatch.setattr(routing, "fit_action", fake_fit)
    x = np.full((10, 1), 0.6)
    model, action, evidence = routing.fit_source_choice(
        x, np.arange(10) % 2, rule="paired", seed=5
    )
    assert model is not None and action == "mean"
    assert evidence["fallback_reason"] == "single_provider"
    assert calls == [(10, "mean", 5)]
    with pytest.raises(ValueError, match="configuration rule"):
        routing.fit_source_choice(x, np.arange(10) % 2, rule="unknown")
