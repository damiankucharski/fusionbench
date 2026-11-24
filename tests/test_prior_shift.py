import numpy as np
import pytest
from scipy.special import expit, softmax
from scipy.stats import multivariate_normal, norm

from fusionbench.eval.prior_shift import generate_prior_shift
from fusionbench.scenarios.calibrated import mc_strength


@pytest.mark.parametrize(
    ("scenario", "kwargs", "first", "second"),
    [
        ("committee", {"k": 3, "accuracy": 0.79}, [0.5, 0.5], [0.2, 0.8]),
        (
            "multiclass_committee",
            {"k": 2, "n_classes": 3},
            [0.25, 0.25, 0.5],
            [0.5, 0.25, 0.25],
        ),
        (
            "saturated",
            {"k": 2, "n_classes": 3, "corr": 0.65},
            [0.25, 0.25, 0.5],
            [0.5, 0.25, 0.25],
        ),
    ],
)
def test_paired_priors_and_folds(scenario, kwargs, first, second):
    def draw(prior, fold=1, fit=first):
        return generate_prior_shift(
            scenario,
            fit_prior=fit,
            test_prior=prior,
            n_fit=101,
            n_test=351,
            seed=4,
            fold=fold,
            **kwargs,
        )

    iid, shifted = draw(first), draw(second)
    np.testing.assert_array_equal(iid.fit_preds, shifted.fit_preds)
    np.testing.assert_array_equal(iid.fit_y, shifted.fit_y)
    same = iid.test_y == shifted.test_y
    assert same.any() and (~same).any()
    np.testing.assert_array_equal(iid.test_preds[same], shifted.test_preds[same])
    expected = iid.test_oracle[same] * (np.array(second) / first)
    np.testing.assert_allclose(
        shifted.test_oracle[same], expected / expected.sum(axis=1, keepdims=True)
    )
    for field in (
        "fit_preds",
        "test_preds",
        "fit_y",
        "test_y",
        "fit_oracle",
        "test_oracle",
    ):
        np.testing.assert_array_equal(
            getattr(shifted, field), getattr(draw(second), field)
        )
    np.testing.assert_array_equal(
        shifted.test_preds, draw(second, fit=second).test_preds
    )
    assert not np.array_equal(shifted.fit_y, draw(second, fold=2).fit_y)
    assert not np.array_equal(shifted.test_preds, draw(second, fold=2).test_preds)
    assert (
        shifted.meta["test_class_counts"]
        == np.bincount(shifted.test_y, minlength=len(second)).tolist()
    )


def test_clipped_binary_scores_have_tail_mass_likelihoods():
    # At accuracy .999 the emitter often clips an induced Gaussian likelihood ratio.
    ds = generate_prior_shift(
        "committee",
        k=1,
        accuracy=0.999,
        fit_prior=[0.8, 0.2],
        test_prior=[0.8, 0.2],
        n_test=1024,
        seed=11,
    )
    p = ds.test_preds[:, 0, 1]
    s = norm.ppf(0.999)
    prior_odds = np.log(0.2 / 0.8)
    for bound in (1e-6, 1 - 1e-6):
        mask = p == bound
        assert mask.any()
        cut = np.log(bound / (1 - bound)) / (2 * s)
        distribution = norm.logsf if bound > 0.5 else norm.logcdf
        expected = expit(
            prior_odds + distribution(cut, loc=s) - distribution(cut, loc=-s)
        )
        np.testing.assert_allclose(ds.test_oracle[mask, 1], expected, atol=1e-15)
        assert abs(expected - expit(prior_odds + np.log(bound / (1 - bound)))) > 1e-7


@pytest.mark.parametrize("rho", [0, 0.64, 1])
def test_multiclass_observed_likelihood(rho):
    prior = np.array([0.15, 0.6, 0.25])
    ds = generate_prior_shift(
        "saturated",
        n_classes=3,
        k=2,
        accuracy=0.72,
        corr=rho,
        fit_prior=prior,
        test_prior=prior,
        n_test=25,
        seed=1,
    )
    # Observable log ratios remove the unidentifiable per-provider logit offset.
    x = np.log(ds.test_preds[:, :, :-1] / ds.test_preds[:, :, -1:])
    if rho == 1:
        np.testing.assert_array_equal(x[:, 0], x[:, 1])
        x = x[:, :1, :]
    k = x.shape[1]
    x = x.reshape(len(x), -1)
    cov = np.kron(np.eye(k) * (1 - rho) + rho, np.eye(2) + np.ones((2, 2)))
    s = mc_strength(0.72, 3)
    logp = []
    for cls in range(3):
        boost = np.eye(3)[cls] * s
        mu = np.tile(boost[:-1] - boost[-1], k)
        logp.append(
            np.log(prior[cls]) + multivariate_normal.logpdf(x, mean=mu, cov=cov)
        )
    np.testing.assert_allclose(
        ds.test_oracle, softmax(np.column_stack(logp), axis=1), atol=2e-14
    )


def test_prior_endpoints_and_split_independence():
    kw = {
        "fit_prior": [0, 1, 0],
        "test_prior": [0, 0, 1],
        "n_classes": 3,
        "k": 3,
        "n_test": 100,
        "seed": 12,
    }
    source = generate_prior_shift("multiclass_committee", n_fit=15, **kw)
    other = generate_prior_shift("multiclass_committee", n_fit=95, **kw)
    assert (source.fit_y == 1).all() and (source.test_y == 2).all()
    np.testing.assert_array_equal(source.fit_oracle, np.eye(3)[source.fit_y])
    np.testing.assert_array_equal(source.test_oracle, np.eye(3)[source.test_y])
    np.testing.assert_array_equal(source.test_preds, other.test_preds)
    uniform = generate_prior_shift(
        "committee", n_fit=20, n_test=20, fit_prior=[0, 1], test_prior=[0, 1]
    )
    assert (uniform.test_oracle[:, 1] == 1).all()
    assert not np.array_equal(uniform.fit_preds, uniform.test_preds)


@pytest.mark.parametrize(
    "bad", [[0.5, 0.3], [0.1, np.nan], [-0.2, 1.2], [1.0], [True, False]]
)
def test_prior_validation(bad):
    with pytest.raises(ValueError, match="test_prior"):
        generate_prior_shift("committee", fit_prior=[0.5, 0.5], test_prior=bad)


@pytest.mark.parametrize(
    "name,value", [("seed", -1), ("fold", True), ("n_fit", 0), ("n_test", 2.5)]
)
def test_split_validation(name, value):
    with pytest.raises(ValueError, match=name):
        generate_prior_shift(
            "committee", fit_prior=[0.5, 0.5], test_prior=[0.5, 0.5], **{name: value}
        )


def test_unsupported_law():
    with pytest.raises(ValueError, match="Prior shift supports"):
        generate_prior_shift("compound", fit_prior=[0.5, 0.5], test_prior=[0.5, 0.5])


@pytest.mark.parametrize(
    "scenario,params,prior",
    [
        ("committee", {"k": 3}, [0.3, 0.7]),
        ("multiclass_committee", {"k": 3, "n_classes": 3}, [0.2, 0.5, 0.3]),
        ("saturated", {"k": 3, "n_classes": 3, "corr": 0.7}, [0.2, 0.5, 0.3]),
    ],
)
def test_emissions_match_original_generator_with_the_same_labels_and_noise(
    scenario, params, prior
):
    from fusionbench import get_scenario

    ds = generate_prior_shift(
        scenario,
        fit_prior=prior,
        test_prior=prior,
        seed=7,
        fold=4,
        n_fit=32,
        n_test=45,
        **params,
    )

    class FixedLabels:
        def __init__(self, y, split):
            self.y = y
            self.noise = np.random.default_rng([7, 4, split, 1])

        def integers(self, low, high, size):
            assert (low, high, size) == (0, len(prior), len(self.y))
            return self.y

        def standard_normal(self, size):
            return self.noise.standard_normal(size)

    base = get_scenario(scenario, **params)
    for split, (y, scores) in enumerate(
        [(ds.fit_y, ds.fit_preds), (ds.test_y, ds.test_preds)]
    ):
        expected, _, _ = base.sample(len(y), FixedLabels(y, split))
        np.testing.assert_array_equal(scores, expected)
