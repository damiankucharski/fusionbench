"""Base predictions must be calibrated: empirically P(y=1 | p) ~= p."""

import numpy as np

from fusionbench.scenarios import calibrated as cal


def test_binary_calibrated_is_calibrated():
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.integers(0, 2, n)
    p1 = cal.binary_calibrated(y, accuracy=0.8, rng=rng)
    # Bin by predicted probability; empirical positive rate should match the bin.
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p1, bins) - 1
    for b in range(10):
        m = idx == b
        if m.sum() < 500:
            continue
        emp = y[m].mean()
        mid = p1[m].mean()
        assert abs(emp - mid) < 0.03, f"bin {b}: emp={emp:.3f} pred={mid:.3f}"


def test_accuracy_matches_target():
    rng = np.random.default_rng(1)
    n = 200_000
    y = rng.integers(0, 2, n)
    for target in (0.7, 0.85):
        p1 = cal.binary_calibrated(y, accuracy=target, rng=rng)
        acc = ((p1 > 0.5).astype(int) == y).mean()
        assert abs(acc - target) < 0.01, f"target {target} got {acc:.3f}"
