"""Fit-side diagnostics for class confidence and repeatable wrong-class preferences."""

import numpy as np

from fusionbench.types import _check_probabilities


def runner_up_profile(preds, labels) -> dict[str, float]:
    """Profile a fit bank shaped [row, provider, class] and its true class labels.

    top_two_margin averages the largest minus second-largest probability over
    rows and providers. For wrong_class_repeatability, every row votes for its
    strongest incorrect class, sharing ties equally. When top-1 is correct,
    this is the runner-up. When top-1 is wrong, the winning wrong class votes.

    Within each true class, alternate rows in their supplied order between two
    halves. For each provider and half, add two pseudocounts in total, uniform
    over the C-1 wrong classes, then subtract the uniform distribution from the
    frequencies. Average the two halves' dot product equally across providers
    and classes with at least two rows. Divide by 1-1/(C-1) and clip to [0, 1].
    Binary banks and banks without an eligible class have repeatability zero.
    """
    p = np.asarray(preds)
    y = np.asarray(labels)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[2] < 2:
        raise ValueError("Expected nonempty [row, provider, class] predictions with at least two classes")
    n, _, classes = p.shape
    _check_probabilities("preds", p, p.shape)
    if y.shape != (n,) or y.dtype.kind not in "biu" or np.any(y < 0) or np.any(y >= classes):
        raise ValueError("Expected one integer class label in range per row")

    top = np.partition(p, -2, axis=-1)
    margin = float(np.subtract(top[..., -1], top[..., -2], dtype=np.float64).mean())
    del top
    dots = []
    if classes > 2:
        uniform = 1.0 / (classes - 1)
        for true_class, count in zip(*np.unique(y, return_counts=True), strict=True):
            if count < 2:
                continue
            wrong = p[y == true_class][:, :, np.arange(classes) != true_class]
            ties = wrong == wrong.max(axis=2, keepdims=True)
            votes = ties / ties.sum(axis=2, keepdims=True)
            deviations = []
            for half in (votes[::2], votes[1::2]):
                frequencies = (half.sum(axis=0) + 2 * uniform) / (len(half) + 2)
                deviations.append(frequencies - uniform)
            dots.append((deviations[0] * deviations[1]).sum(axis=1).mean())
    repeatability = float(np.clip(np.mean(dots) / (1 - 1.0 / (classes - 1)), 0, 1)) if dots else 0.0
    return {"top_two_margin": margin, "wrong_class_repeatability": repeatability}
