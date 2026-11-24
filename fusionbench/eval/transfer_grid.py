"""Paired annotation noise and clean-label transfer conditions for fusion methods."""

from collections.abc import Mapping
from operator import index

import numpy as np
from scipy.special import softmax

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.blocks import SHIFTS, draw_block
from fusionbench.scenarios.registry import SCENARIOS

SOURCE_NOISE = (0.0, 0.15, 0.30)
SEVERITIES = (0.25, 0.50, 1.0)
INPUT_MODES = ("provider_swap", "log_jitter")
MODES = (*INPUT_MODES, "canonical_shift")
# Each draw starts a fresh default_rng([bank_seed, stream]). Requests may be
# reordered or restricted without changing any individual test condition.
STREAMS = {
    "development": 0,
    "clean_test": 1,
    "canonical_shift": 2,
    "source_validation": 3,
    "annotation_noise": 10,
    "row_threshold": 11,
    "provider_swap": 12,
    "log_jitter": 13,
    "validation_annotation_noise": 14,
}


def _constructor(name):
    if isinstance(name, str):
        return SCENARIOS[name]
    if isinstance(name, type) and issubclass(name, Scenario):
        return name
    raise TypeError("name must be a registered scenario key or a Scenario subclass")


def _indices(labels, n_classes):
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= n_classes)):
        raise ValueError("Supply a vector of integer class indices in range")
    return y


def noisy_labels(labels, n_classes, seed, q, *, stream=STREAMS["annotation_noise"]):
    """Flip each label independently with probability q, uniformly to another class.

    A row's threshold and wrong class are fixed across noise levels. Applying
    this once before discovery and validation splitting gives every method
    the same noisy development annotations. The optional stream separates the
    held-out source-validation flips from training flips. The input labels are untouched.
    """
    n_classes = index(n_classes)
    if n_classes < 2 or not 0.0 <= q <= 1.0:
        raise ValueError("Use at least two classes and a noise probability in [0, 1]")
    y = _indices(labels, n_classes)
    if q == 0.0:
        return y.copy()
    rng = np.random.default_rng([seed, stream])
    mask = rng.random(len(y)) < q
    offsets = rng.integers(1, n_classes, size=len(y))
    wrong = (y.astype(np.int64) + offsets) % n_classes
    return np.where(mask, wrong, y.astype(np.int64))


def development(name, seed, n, q=0.0, *, scenario_kwargs=None):
    """Draw source probabilities and possibly corrupted labels from stream 0.

    name is a registry key or Scenario subclass. scenario_kwargs are passed to
    its constructor. SOURCE_NOISE lists the levels used in the fixed experiment.
    """
    if n < 1:
        raise ValueError("Development requires at least one row")
    scenario = _constructor(name)(**(scenario_kwargs or {}))
    x, y, _ = draw_block(scenario, seed, STREAMS["development"], n)
    return x, noisy_labels(y, scenario.n_classes, seed, q)


def brier(predictions, labels):
    """Return the mean row sum of squared class probability errors (range 0 to 2)."""
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2 or len(p) != len(labels):
        raise ValueError("Expected one full-class probability vector per test label")
    _check_probabilities(p)
    y = _indices(labels, p.shape[1])
    return float(np.square(p - np.eye(p.shape[1])[y]).sum(1).mean())


def _check_probabilities(p):
    if not np.all((p >= 0) & (p <= 1)) or not np.allclose(
        p.sum(axis=-1), 1, rtol=0, atol=1e-6
    ):
        raise ValueError("Supply finite probabilities in [0, 1] summing to one")


def _grid(modes, severities, allowed):
    modes = (modes,) if isinstance(modes, str) else tuple(modes)
    levels = tuple(dict.fromkeys(float(s) for s in severities))
    if any(mode not in (*allowed, "clean") for mode in modes):
        raise ValueError(f"Unknown mode. Supported modes are clean and {allowed}")
    if any(not 0 <= s <= 1 for s in levels):
        raise ValueError("Severities must be finite numbers in [0, 1]")
    return tuple(dict.fromkeys(m for m in modes if m != "clean")), tuple(
        s for s in levels if s > 0
    )


def provider_swap_plan(n_rows, n_providers, seed):
    """Return (provider index, donor row indices, uniform row thresholds).

    The provider is uniform over columns. Donors are a uniform derangement,
    independent of labels. A swap at severity s uses rows with threshold < s.
    """
    if n_rows < 2 or n_providers < 1:
        raise ValueError("A swap requires at least two rows and one provider")
    rng = np.random.default_rng([seed, STREAMS["provider_swap"]])
    provider = int(rng.integers(n_providers))
    rows = np.arange(n_rows)
    donors = rng.permutation(n_rows)
    while np.any(donors == rows):
        donors = rng.permutation(n_rows)
    threshold = np.random.default_rng([seed, STREAMS["row_threshold"]]).random(n_rows)
    return provider, donors, threshold


def iter_input_stresses(predictions, seed, *, severities=SEVERITIES, modes=INPUT_MODES):
    """Yield (mode, positive severity, stressed probabilities) for supplied rows.

    Inputs have shape [row, provider, class]. provider_swap keeps one anonymous
    provider and its donor rows fixed across severity. log_jitter adds iid
    N(0,(1.5*s)^2) noise per centered log-probability coordinate, then softmax.
    The Gaussian direction is fixed across levels. Only relative coordinates
    affect the result. Exact zeros use the smallest normal float64 before log.
    """
    modes, levels = _grid(modes, severities, INPUT_MODES)
    p = np.asarray(predictions)
    if p.ndim != 3 or min(p.shape[:2]) < 1 or p.shape[2] < 2:
        raise ValueError("Expected nonempty row-by-provider-by-class probabilities")
    _check_probabilities(p)
    if not levels:
        return
    for mode in modes:
        if mode == "provider_swap":
            provider, donors, threshold = provider_swap_plan(len(p), p.shape[1], seed)
            for s in levels:
                mask = threshold < s
                swapped = p.copy()
                swapped[mask, provider] = p[donors[mask], provider]
                yield mode, s, swapped
        elif mode == "log_jitter":
            logs = np.log(np.maximum(p.astype(float, copy=False), np.finfo(float).tiny))
            centered = logs - logs.mean(axis=-1, keepdims=True)
            noise = np.random.default_rng([seed, STREAMS[mode]]).standard_normal(
                p.shape
            )
            for s in levels:
                yield mode, s, softmax(centered + (1.5 * s) * noise, axis=-1)


def iter_test_conditions(
    name,
    seed,
    n,
    *,
    scenario_kwargs=None,
    severities=SEVERITIES,
    modes=MODES,
    canonical_shift=None,
):
    """Yield (mode, severity, predictions, clean labels), starting with clean once.

    name is a registered key or Scenario subclass. Stream 1 supplies clean rows
    and stream 2 the canonical endpoint, matching the native-pattern benchmark.
    The built-in SHIFTS are looked up by registered name (or class). Supply a
    mapping as canonical_shift to replace the complete default shift definition.
    Source constructor options in scenario_kwargs carry over unless overridden
    by the shift. An unregistered scenario without an explicit shift yields
    only the common input stresses. Omit canonical_shift from modes to skip it.

    The same uniform thresholds govern provider swaps and domain mixtures.
    Severity is a selection probability, so its realized fraction fluctuates
    by bank. Each domain mixture takes an entire (prediction row, label) pair
    from one domain. Input-only disturbances preserve all original test labels.
    """
    if n < 1:
        raise ValueError("Test conditions require at least one row")
    modes, levels = _grid(modes, severities, MODES)
    ctor = _constructor(name)
    kwargs = dict(scenario_kwargs or {})
    scenario = ctor(**kwargs)
    clean, labels, _ = draw_block(scenario, seed, STREAMS["clean_test"], n)
    yield "clean", 0.0, clean, labels
    if not levels:
        return
    for mode in modes:
        if mode in INPUT_MODES:
            for _, s, x in iter_input_stresses(
                clean, seed, severities=levels, modes=(mode,)
            ):
                yield mode, s, x, labels
            continue

        registry_key = (
            name
            if isinstance(name, str)
            else next((k for k, v in SCENARIOS.items() if v is ctor), None)
        )
        shift = SHIFTS.get(registry_key) if canonical_shift is None else canonical_shift
        if shift is None:
            continue
        if not isinstance(shift, Mapping):
            raise TypeError("canonical_shift must be a mapping of constructor options")
        shifted = ctor(**{**kwargs, **shift})
        if (
            shifted.model_ids() != scenario.model_ids()
            or shifted.n_classes != scenario.n_classes
        ):
            raise ValueError("The shift changed the provider mapping or class count")
        other, other_y, _ = draw_block(shifted, seed, STREAMS[mode], n)
        threshold = np.random.default_rng([seed, STREAMS["row_threshold"]]).random(n)
        for s in levels:
            mask = threshold < s
            yield (
                mode,
                s,
                np.where(mask[:, None, None], other, clean),
                np.where(mask, other_y, labels),
            )
