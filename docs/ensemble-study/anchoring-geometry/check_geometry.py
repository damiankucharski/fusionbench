"""Exact checks for the anchoring note. Run with the Python standard library."""

from fractions import Fraction as F
from itertools import product
from math import comb

# The five frozen providers have a uniform mean of 1/2. A raw menu item
# can be deployed by reading a single provider.
PROVIDERS = tuple(F(p, 10) for p in (1, 4, 5, 6, 9))
PRIOR = sum(PROVIDERS) / len(PROVIDERS)


def mix(raw, tree_share):
    return PRIOR + tree_share * (raw - PRIOR)


def select(observed_rate, tree_share):
    # Full-class binary Brier is twice the scalar loss. For a constant
    # prediction its candidate-dependent part is twice this squared distance.
    return min(PROVIDERS, key=lambda raw: (mix(raw, tree_share) - observed_rate) ** 2)


def excess(prediction, true_rate):
    return 2 * (prediction - true_rate) ** 2


def expected_excess(true_rate, n, deployed_share):
    before = after = F(0)
    for positives in range(n + 1):
        rate = F(positives, n)
        probability = (
            comb(n, positives)
            * true_rate**positives
            * (1 - true_rate) ** (n - positives)
        )
        raw = select(rate, F(1))
        inside = select(rate, deployed_share)
        assert abs(inside - PRIOR) >= abs(raw - PRIOR)
        after += probability * excess(mix(raw, deployed_share), true_rate)
        before += probability * excess(mix(inside, deployed_share), true_rate)
    return before, after


def check():
    share = F(1, 4)
    rate = F(12, 20)
    assert select(rate, F(1)) == F(3, 5)
    assert select(rate, share) == F(9, 10)
    post = mix(select(rate, F(1)), share)
    inside = mix(select(rate, share), share)
    assert (post, inside) == (F(21, 40), F(3, 5))
    assert (excess(post, PRIOR), excess(inside, PRIOR)) == (F(1, 800), F(1, 50))
    assert (excess(post, rate), excess(inside, rate)) == (F(9, 800), 0)

    # Threshold and rank changes at the dissertation's 85% prior weight.
    # Verify the extremal tree outputs, which attain the interval endpoints.
    share = F(3, 20)
    threshold = F(1, 2)
    for k in range(101):
        prior = F(k, 100)
        min_score = (1 - share) * prior
        max_score = min_score + share
        can_cross = min_score <= threshold < max_score
        assert can_cross == (F(7, 17) < prior <= F(10, 17))
        for j in range(k):
            gap = prior - F(j, 100)
            flipped_or_tied = min_score <= (1 - share) * F(j, 100) + share
            assert flipped_or_tied == (gap <= F(3, 17))

    # Best predictor observing only the first provider. The full binary
    # posterior averages two independent fair bits with equal weights.
    pairs = list(product((0, 1), repeat=2))
    floor = sum(excess((F(x) + PRIOR) / 2, F(x + y, 2)) for x, y in pairs) / 4
    assert floor == F(1, 8)
    return floor


if __name__ == "__main__":
    print("Exact full-class Brier excess averaged over 20 validation labels")
    print("q      search inside blend    blend after selection")
    for q in (F(1, 2), F(3, 5), F(9, 10)):
        inside, after = expected_excess(q, 20, F(1, 4))
        print(f"{float(q):.2f}   {float(inside):.8f}             {float(after):.8f}")
    print("One omitted independent provider, irreducible excess:", check())
