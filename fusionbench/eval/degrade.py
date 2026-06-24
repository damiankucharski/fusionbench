"""Validation-reliability degradation.

The main suite holds reliability at its maximum (50k clean fit rows). This module
*deliberately degrades* the fit (selection/training) split — subsampling it and/or
injecting label noise — while leaving the large clean test split untouched. This
exposes the complementary axis the controlled benchmark otherwise removes:
robustness of a method when the validation signal it selects on is weak.

Flexible meta-learners (stacking, TPOT) overfit a small/noisy fit split and lose
test performance; parameter-light rules (averaging, fixed operators, parsimonious
OKAPI trees) degrade far less. This is precisely OKAPI's protective regime.
"""

from __future__ import annotations

import numpy as np

from fusionbench.types import Dataset


def _stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-stratified subsample of size ~n."""
    classes = np.unique(y)
    per = max(1, n // len(classes))
    idx = []
    for c in classes:
        pool = np.where(y == c)[0]
        take = min(per, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def degrade_fit(ds: Dataset, *, fit_n: int | None = None, label_noise: float = 0.0,
                seed: int = 0) -> Dataset:
    """Return a copy of ``ds`` with a degraded fit split (test left intact)."""
    rng = np.random.default_rng(seed)
    fp, fy = ds.fit_preds, ds.fit_y

    if fit_n is not None and fit_n < fy.shape[0]:
        idx = _stratified_subsample(fy, fit_n, rng)
        fp, fy = fp[idx], fy[idx]
    else:
        fy = fy.copy()

    if label_noise > 0.0:
        flip = rng.random(fy.shape[0]) < label_noise
        if ds.n_classes == 2:
            fy[flip] = 1 - fy[flip]
        else:
            shift = rng.integers(1, ds.n_classes, size=int(flip.sum()))
            fy[flip] = (fy[flip] + shift) % ds.n_classes

    return Dataset(
        name=ds.name,
        task=ds.task,
        n_classes=ds.n_classes,
        model_ids=ds.model_ids,
        fit_preds=fp,
        fit_y=fy,
        test_preds=ds.test_preds,
        test_y=ds.test_y,
        fit_oracle=None,
        test_oracle=ds.test_oracle,
        oracle_description=ds.oracle_description,
        expected_winners=ds.expected_winners,
        meta=ds.meta,
    )
