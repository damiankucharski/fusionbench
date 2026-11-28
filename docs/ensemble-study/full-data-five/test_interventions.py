import numpy as np
import pytest
from interventions import _row_brier, fit_candidate

from fusionbench.methods.okapi import _task_postprocessing


@pytest.mark.parametrize("classes", [2, 4])
def test_seed_matches_mean_in_native_task_and_sham_has_same_quota(classes):
    rng = np.random.default_rng(27)
    x = rng.dirichlet(np.ones(classes), size=(32, 4))
    y = rng.integers(classes, size=len(x))
    first = fit_candidate("seeded_mean", x, y)
    sham = fit_candidate("sham", x, y)
    assert first.initial_population_size_ == sham.initial_population_size_ == 5
    with _task_postprocessing(first.task):
        p = first.seed_tree_.predict()
        expected = x.mean(1)
        np.testing.assert_allclose(p, expected[:, 1:] if classes == 2 else expected, rtol=0, atol=1e-14)
        duplicate = sham.seed_tree_.predict()
        assert any(np.allclose(duplicate, v, rtol=0, atol=1e-14) for v in sham._to_okapi(x).transpose(1, 0, 2))
    assert first.okapi.selection_fitness_decimals == 3


@pytest.mark.parametrize("policy", ["pull_mean", "pull_zero", "paired_se", "paired_zero"])
def test_stored_terminal_fitness_is_penalized_all_row_score(policy):
    rng = np.random.default_rng(47)
    x = rng.dirichlet([1., 1.], size=(24, 4))
    y = rng.integers(2, size=len(x))
    groups = np.repeat(np.arange(6), 4)
    fitted = fit_candidate(policy, x, y, groups=groups)
    native = fitted.okapi
    pred = fitted.predict(x)
    a = x.mean(1)
    target = np.eye(2)[y]
    losses = np.square(pred - target).sum(1)
    if policy.startswith("pull"):
        strength = np.sqrt(8 / len(y))
        distance = np.square(pred - a).sum(1).mean() if policy == "pull_mean" else 0.
        score = 1 - (losses.mean() + strength * distance) / (2 * (1 + strength))
    else:
        gains = np.square(a - target).sum(1) - losses
        by_group = np.array([gains[groups == i].mean() for i in np.unique(groups)])
        g = by_group.mean()
        if policy == "paired_se":
            g -= 1.645 * by_group.std(ddof=1) / np.sqrt(len(by_group))
        score = .5 + np.arctan(g) / np.pi
    assert native.pareto_fitnesses[:, 0].max() == pytest.approx(round(score, 3))
    np.testing.assert_allclose(_row_brier(pred[:, 1:], y), losses)
