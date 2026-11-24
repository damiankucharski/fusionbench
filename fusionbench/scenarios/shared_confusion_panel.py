"""Change which wrong class a provider favors while preserving its per-row loss."""

import hashlib
import json

import numpy as np

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.multiclass_provider_panel import (
    MulticlassProviderPanel,
    _finite,
)


def _destination_cdf(weights, shape, name):
    if weights is None:
        return np.broadcast_to(0.0, shape)
    weights = np.asarray(weights)
    if weights.shape != shape or weights.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real array of shape {shape}")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name} must contain finite nonnegative preference weights")
    if np.any(weights.diagonal(axis1=-2, axis2=-1)):
        raise ValueError(f"{name} must have zero on its true-class diagonal")
    # Scaling first avoids overflowing when summing several large finite weights.
    peak = weights.max(axis=-1, keepdims=True)
    normalized = np.divide(weights, peak, out=np.zeros(shape), where=peak > 0)
    cdf = np.cumsum(normalized, axis=-1)
    total = cdf[..., -1:]
    return np.divide(cdf, total, out=np.zeros_like(cdf), where=total > 0)


class SharedConfusionPanel(MulticlassProviderPanel):
    """Direct the largest wrong probability toward a sampled preferred class.

    shared_matrix[y, c] and provider_matrices[j, y, c] are destination weights.
    On each row and provider, strength is the probability of applying a swap.
    Given a swap, deviation_mix is the probability of using provider weights.
    A row of zero weights leaves the original prediction intact. Omitting
    provider_matrices reuses shared_matrix for all three providers.
    """

    def __init__(
        self,
        *,
        shared_matrix=None,
        provider_matrices=None,
        strength=0.0,
        deviation_mix=0.0,
        name=None,
        **panel_parameters,
    ):
        super().__init__(name=name, **panel_parameters)
        self.strength = _finite(strength, "strength", high=1)
        self.deviation_mix = _finite(deviation_mix, "deviation_mix", high=1)
        classes = self.n_classes
        self._shared_cdf = _destination_cdf(
            shared_matrix, (classes, classes), "shared_matrix"
        )
        if provider_matrices is None:
            self._provider_cdfs = np.broadcast_to(
                self._shared_cdf, (3, classes, classes)
            )
        else:
            self._provider_cdfs = _destination_cdf(
                provider_matrices, (3, classes, classes), "provider_matrices"
            )
        self.knobs.update(
            shared_matrix=None
            if shared_matrix is None
            else np.asarray(shared_matrix).tolist(),
            provider_matrices=None
            if provider_matrices is None
            else np.asarray(provider_matrices).tolist(),
            strength=self.strength,
            deviation_mix=self.deviation_mix,
        )
        if name is None:
            # Full matrices are recorded in metadata, and a short digest fits on disk.
            config = json.dumps(self.knobs, sort_keys=True).encode()
            self.name = (
                f"shared_confusion_panel_{hashlib.sha256(config).hexdigest()[:16]}"
            )

    def _sample_with_seeds(self, n, latent_seed, vote_seeds):
        x, y, latent = super()._sample_with_seeds(n, latent_seed, vote_seeds)
        if self.strength == 0:
            return x, y, latent
        # Independent of the latent/difficulty and finite-vote generators.
        # Draw all gates, destinations and tie choices before selecting a subset.
        rng = np.random.default_rng([int(latent_seed), 0x534350])
        gate, private, destination, tie = rng.random((n, 3, 4)).transpose(2, 0, 1)
        prefer_private = private < self.deviation_mix
        for label in np.unique(y):
            for provider in range(3):
                active = (y == label) & (gate[:, provider] < self.strength)
                for use_private, cdf in (
                    (False, self._shared_cdf[label]),
                    (True, self._provider_cdfs[provider, label]),
                ):
                    if cdf[-1] == 0:
                        continue
                    rows = np.flatnonzero(
                        active & (prefer_private[:, provider] == use_private)
                    )
                    if not len(rows):
                        continue
                    dest = np.searchsorted(
                        cdf, destination[rows, provider], side="right"
                    )
                    wrong = x[rows, provider]
                    wrong[:, label] = -np.inf
                    maxima = wrong == wrong.max(axis=1, keepdims=True)
                    tie_rank = tie[rows, provider] * maxima.sum(axis=1)
                    src = (maxima.cumsum(axis=1) > tie_rank[:, None]).argmax(axis=1)
                    x[rows, provider, src], x[rows, provider, dest] = (
                        x[rows, provider, dest],
                        x[rows, provider, src],
                    )
        return x, y, latent


class SimilarClasses38(Scenario):
    """Three ten-class providers share a persistent 3 ↔ 8 confusion."""

    task = "multiclass"
    primary_metric = "brier"

    def __init__(self, *, name=None):
        shared = np.zeros((10, 10))
        shared[3, 8] = shared[8, 3] = 1
        self.panel = SharedConfusionPanel(shared_matrix=shared, strength=1.0)
        super().__init__(name=name or "similar_classes_38", **self.panel.knobs)

    @property
    def n_classes(self):
        return self.panel.n_classes

    def model_ids(self):
        return self.panel.model_ids()

    def sample(self, n, rng):
        return self.panel.sample(n, rng)
