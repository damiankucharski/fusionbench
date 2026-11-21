"""Exact Bayes risks for two noiseless worlds sharing two majority votes."""

import math
from numbers import Integral, Real

import numpy as np
from scipy.special import stdtr


def two_world_bounds(theta, chi_df, n):
    """Expected risks under an equal prior and n independent final-label observations.

    The two unchanged votes are independent fair signs. In the remaining plane,
    the training coordinates are an independent signed chi radius and standard
    normal. Test coordinates are independent standard normals. The world changes
    the last vote by theta radians. Brier uses the positive-class probability.
    """
    if (
        isinstance(theta, bool)
        or not isinstance(theta, Real)
        or not 0 <= theta <= math.pi
    ):
        raise ValueError("theta must be a real angle in [0, pi]")
    if isinstance(chi_df, bool) or not isinstance(chi_df, Integral) or chi_df <= 0:
        raise ValueError("chi_df must be a positive integer")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    theta, chi_df, n = float(theta), int(chi_df), int(n)
    p_test = theta / (2 * math.pi)
    if chi_df == 1 or theta in (0.0, math.pi / 2, math.pi):
        p_train = p_test
    elif chi_df == 2:
        p_train = 0.5 * math.sin(theta / 2) ** 2
    else:
        # The ratio of the orthogonal normal and radius / sqrt(df) is Student t.
        threshold = -math.sqrt(chi_df) * math.cos(theta) / math.sin(theta)
        p_train = 0.5 * float(stdtr(chi_df, threshold))
    unresolved = 1.0
    if n and p_train:
        hazard = -math.log1p(-p_train)
        # The log-domain product also accepts sample counts beyond float range.
        log_exposure = math.log(n) + math.log(hazard)
        unresolved = (
            0.0
            if log_exposure >= math.log(746.0)
            else math.exp(-math.exp(log_exposure))
        )
    return {
        "p_train": p_train,
        "p_test": p_test,
        "unresolved": unresolved,
        "world_error": 0.5 * unresolved,
        "target_error": 0.5 * p_test * unresolved,
        "target_brier": 0.25 * p_test * unresolved,
    }


def shared_majority_probability(a, b, c0, c1, weight):
    """Predict the positive class with a posterior weight for the second world.

    a, b are the shared binary votes. c0 and c1 are the third votes in the two
    worlds. All inputs broadcast. If the shared votes split, their predictions
    are averaged with the posterior weight; otherwise their agreement decides.
    """
    votes = []
    for vote in (a, b, c0, c1):
        value = np.asarray(vote)
        if value.dtype.kind not in "biuf" or not np.isin(value, (0, 1)).all():
            raise ValueError("all four votes must be binary (0 or 1)")
        votes.append(value.astype(np.float64, copy=False))
    w = np.asarray(weight)
    if w.dtype.kind not in "biuf" or not ((w >= 0) & (w <= 1)).all():
        raise ValueError("weight must be a probability in [0, 1]")
    # The fused alternative vote lies in [0, 1], so clipping a single linear
    # sum evaluates the posterior mixture without duplicating the shared votes.
    left, right, alt0, alt1 = votes
    return np.clip((left + right - 1) + ((1 - w) * alt0 + w * alt1), 0.0, 1.0)
