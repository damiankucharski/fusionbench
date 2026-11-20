import numpy as np

from fusionbench.eval.sentinel_audit import batch_scores


def test_label_reversal_changes_only_supervised_scores():
    p = np.array([[.95, .1, .9], [.02, .8, .7]])
    y = p > .5
    good = batch_scores(p, y, 3)
    reversed_ = batch_scores(p, ~y, 3)
    np.testing.assert_array_equal(good["unlabeled_uncertainty"], reversed_["unlabeled_uncertainty"])
    for method in ["error", "brier", "log_loss", "likelihood_ratio"]:
        assert np.all(reversed_[method] > good[method])


def test_at_uninformative_probabilities_probability_scores_are_label_invariant():
    p = np.full((2, 4), .5)
    y = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
    original = batch_scores(p, y, 4)
    changed = batch_scores(p, ~y, 4)
    for method in ["brier", "log_loss", "likelihood_ratio"]:
        np.testing.assert_array_equal(original[method], changed[method])
