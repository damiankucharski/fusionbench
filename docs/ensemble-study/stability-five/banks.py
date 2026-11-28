"""Rebuild the five preassigned cohorts. Only bank.x and bank.y are fusion-fit inputs.

For digits, the original three-provider builder uses a stratified split of sklearn's
1,797 global image indices. The bank seed selects 898 base-training images; seed+1
splits the remainder into 449 fusion-fit and 450 audit images. The 100-row condition
uses seed+2 on the full 449 fit positions. Its audit and providers are identical.
Digits IDs on the Bank refer to the original sklearn rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from examples.digits_predictions import write_digits_predictions
from fusionbench import generate_dataset, load_dataset
from fusionbench.scenarios.stratified_pool import StratifiedPool
from fusionbench.transfer import load_preset


@dataclass
class Bank:
    context: dict
    x: np.ndarray
    y: np.ndarray
    audit_x: np.ndarray
    audit_y: np.ndarray | None = None
    mass: np.ndarray | None = None
    q: np.ndarray | None = None
    fit_ids: np.ndarray | None = None
    audit_ids: np.ndarray | None = None
    base_ids: np.ndarray | None = None


def _bank(round_no, cohort, family, seed, provider_ids, x, y, audit_x, *,
          audit_y=None, mass=None, q=None, fit_ids=None, audit_ids=None,
          base_ids=None, specialization=None):
    n_fit, providers, classes = x.shape
    assert y.shape == (n_fit,) and audit_x.shape[1:] == (providers, classes)
    assert len(provider_ids) == providers
    n_audit = len(audit_x)
    if audit_y is None:
        assert classes == 2 and mass.shape == q.shape == (n_audit,)
        assert np.all(mass >= 0) and np.isclose(mass.sum(), 1)
        assert np.all((q >= 0) & (q <= 1))
    else:
        assert audit_y.shape == (n_audit,) and q is None and mass is None
    context = {
        "round_no": round_no, "cohort": cohort, "family": family, "seed": seed,
        "specialization": specialization, "n_fit": n_fit, "n_audit": n_audit,
        "n_classes": classes, "n_providers": providers, "provider_ids": tuple(provider_ids),
        "class_ids": tuple(range(classes)), "fit_class_counts": tuple(np.bincount(y, minlength=classes).tolist()),
        "audit_class_counts": (None if audit_y is None else
                              tuple(np.bincount(audit_y, minlength=classes).tolist())),
    }
    for name, arr in (("fit_x", x), ("fit_y", y), ("audit_x", audit_x), ("audit_y", audit_y),
                      ("mass", mass), ("q", q), ("fit_ids", fit_ids), ("audit_ids", audit_ids),
                      ("base_ids", base_ids)):
        if arr is not None:
            context[name + "_sha256"] = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return Bank(context, x, y, audit_x, audit_y, mass, q, fit_ids, audit_ids, base_ids)


def _controlled(round_no, seeds):
    for family, d in (("equal", 0.0), ("weak", 0.25), ("strong", 0.75)):
        scenario = StratifiedPool(d)
        support, mass, q = scenario.support()
        for seed in seeds:
            x, y, latent = scenario.sample(128, np.random.default_rng(seed))
            strata = 2 * latent["high_amplitude"].astype(int) + y
            np.testing.assert_array_equal(np.bincount(strata), [32, 32, 32, 32])
            yield _bank(round_no, "controlled", family, seed, scenario.model_ids(),
                        x, y, support, mass=mass, q=q, specialization=d)


def _standard(round_no, seeds):
    families = load_preset("standard-v5")["scenarios"]
    if len(families) != 27:
        raise ValueError("This study requires the 27-family standard-v5 preset")
    for family in families:
        for seed in seeds:
            ds = generate_dataset(family, seed=seed, n_per_split=512)
            yield _bank(round_no, "standard", family, seed, ds.model_ids, ds.fit_preds,
                        ds.fit_y, ds.test_preds, audit_y=ds.test_y)


def _digits(round_no, seeds):
    labels = load_digits().target
    # Reconstruct original indices using the checked split helper from pattern-iterations.
    previous = Path(__file__).resolve().parents[1] / "pattern-iterations" / "digits.py"
    split_indices = run_path(str(previous))["split_indices"]
    for seed in seeds:
        with TemporaryDirectory() as tmp, threadpool_limits(limits=1):
            ds = load_dataset(write_digits_predictions(tmp, seed=seed))
        base, fit, test = split_indices(labels, seed)
        np.testing.assert_array_equal(ds.fit_y, labels[fit])
        np.testing.assert_array_equal(ds.test_y, labels[test])
        assert list(ds.model_ids) == ["lr", "neighbours", "forest"]
        subsets = [np.arange(len(fit))]
        if round_no == 5:
            small, _ = train_test_split(subsets[0], train_size=100, stratify=labels[fit],
                                        random_state=seed + 2)
            subsets.append(small)
        for kept in subsets:
            yield _bank(round_no, "digits", f"digits_{len(kept)}", seed, ds.model_ids,
                        ds.fit_preds[kept], ds.fit_y[kept], ds.test_preds, audit_y=ds.test_y,
                        fit_ids=fit[kept], audit_ids=test, base_ids=base)


def banks(round_no):
    """Yield 106 banks in each of rounds 1–4 and 220 in the independent fifth round."""
    if isinstance(round_no, (bool, np.bool_)) or not isinstance(round_no, (int, np.integer)) or not 1 <= round_no <= 5:
        raise ValueError("round_no must be an integer in 1..5")
    round_no = int(round_no)
    base = 270000 + 1000 * round_no
    yield from _controlled(round_no, range(base, base + (32 if round_no == 5 else 16)))
    yield from _standard(round_no, range(base + 100, base + (104 if round_no == 5 else 102)))
    yield from _digits(round_no, range(base + 200, base + (208 if round_no == 5 else 204)))
