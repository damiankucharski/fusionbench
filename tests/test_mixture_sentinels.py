import numpy as np
from scipy.special import expit, softmax

from fusionbench.eval.mixture_sentinels import (
    choose_brier,
    intercept_adjustment,
    probabilities_given_labels,
)


def test_equal_class_conditional_weights_reduce_to_binomial_evidence():
    prior = np.array([0.25, 0.5, 0.75])
    logits = np.repeat(np.log(prior / (1 - prior))[:, None], 5, axis=1)
    labels = np.array([1, 1, 0, 1, 1])
    evidence = softmax(probabilities_given_labels(logits, labels))
    exact = prior**4 * (1 - prior)
    np.testing.assert_allclose(evidence, exact / exact.sum())
    assert np.all(probabilities_given_labels(logits[:, :0], labels[:0]) == 0)


def test_intercept_and_tie_rules_for_sparse_labels():
    logits = np.array([-1.0, 0.2, 1.0, 0.4])
    labels = np.ones(len(logits))
    delta = intercept_adjustment(logits, labels)
    assert delta > 0
    assert abs((expit(logits + delta) - labels).sum() + delta / 4) < 1e-12
    assert intercept_adjustment(np.array([]), np.array([])) == 0.0
    probs = np.array([[0.5, 0.5], [0.75, 0.75], [0.25, 0.25]])
    density = np.array([2.0, 3.0, 3.0])
    assert choose_brier(probs[:, :0], np.array([]), density) == 1
    assert choose_brier(probs, np.array([1, 0]), density) == 0
