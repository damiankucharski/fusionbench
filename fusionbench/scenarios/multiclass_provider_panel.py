"""Three heterogeneous multiclass providers with shared difficulty and finite votes."""

from numbers import Integral, Real

import numpy as np
from scipy.special import softmax

from fusionbench.scenarios.base import Scenario


def _finite(value, name, *, low=0.0, high=np.inf):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return value


def _voted_probabilities(p, votes, seed):
    # A changed row's probabilities can change how many random numbers a
    # multinomial draw consumes. Separate row seeds keep all the other rows paired.
    seeds = np.random.default_rng(seed).integers(2**63, size=len(p), dtype=np.uint64)
    result = np.empty_like(p)
    for i, row_seed in enumerate(seeds):
        result[i] = np.random.default_rng(row_seed).multinomial(votes, p[i])
    return result / votes


class MulticlassProviderPanel(Scenario):
    """Sample labels, shared example difficulty, correlated provider logits and votes.

    Each provider receives unit-variance Gaussian logit noise with pairwise
    correlation shared_error within each class. The correct class receives a
    boost of separation times provider strength, scaled by hard_scale on hard
    rows. On those rows provider j also boosts class (y + 1 + j % (C-1)) % C
    by confusion_bias. Divide logits by the provider's temperature and apply
    softmax. Providers 1 and 2 optionally report independent multinomial vote
    frequencies. Provider indices start at zero. The clean posterior given
    the reported probabilities is unavailable.
    """

    task = "multiclass"
    primary_metric = "brier"

    def __init__(
        self,
        n_classes=10,
        separation=6.0,
        hard_fraction=0.15,
        hard_scale=0.55,
        shared_error=0.65,
        strengths=(1.0, 0.9, 0.8),
        temperatures=(1.5, 1.0, 0.75),
        votes=(31, 7),
        confusion_bias=0.0,
        *,
        name=None,
    ):
        if (
            isinstance(n_classes, (bool, np.bool_))
            or not isinstance(n_classes, Integral)
            or n_classes < 3
        ):
            raise ValueError("n_classes must be an integer of at least 3")
        self._classes = int(n_classes)
        self.separation = _finite(separation, "separation")
        self.hard_fraction = _finite(hard_fraction, "hard_fraction", high=1)
        self.hard_scale = _finite(hard_scale, "hard_scale", high=1)
        self.shared_error = _finite(shared_error, "shared_error", high=1)
        self.confusion_bias = _finite(confusion_bias, "confusion_bias")
        if np.shape(strengths) != (3,) or np.shape(temperatures) != (3,):
            raise ValueError("Provide three strengths and three temperatures")
        self.strengths = tuple(_finite(s, "strengths") for s in strengths)
        self.temperatures = tuple(_finite(t, "temperatures") for t in temperatures)
        if min(self.temperatures) <= 0:
            raise ValueError("temperatures must be positive")
        if np.shape(votes) != (2,) or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
            for v in votes
        ):
            raise ValueError("votes must be two nonnegative integer counts")
        self.votes = tuple(map(int, votes))
        super().__init__(
            name=name,
            n_classes=self._classes,
            separation=self.separation,
            hard_fraction=self.hard_fraction,
            hard_scale=self.hard_scale,
            shared_error=self.shared_error,
            strengths=self.strengths,
            temperatures=self.temperatures,
            votes=self.votes,
            confusion_bias=self.confusion_bias,
        )

    @property
    def n_classes(self):
        return self._classes

    def model_ids(self):
        return ["provider_0", "provider_1", "provider_2"]

    def sample(self, n, rng):
        # Consume a fixed number of caller draws for every configuration.
        # Toggling voting preserves latent examples on this and subsequent splits.
        latent_seed, *vote_seeds = rng.integers(2**63, size=3, dtype=np.uint64)
        latent = np.random.default_rng(latent_seed)
        y = latent.integers(self.n_classes, size=n)
        hard = latent.random(n) < self.hard_fraction
        shared = latent.standard_normal((n, 1, self.n_classes))
        private = latent.standard_normal((n, 3, self.n_classes))
        logits = (
            np.sqrt(self.shared_error) * shared
            + np.sqrt(1 - self.shared_error) * private
        )
        boost = self.separation * np.where(hard, self.hard_scale, 1.0)
        rows = np.arange(n)[:, None]
        providers = np.arange(3)
        logits[rows, providers, y[:, None]] += boost[:, None] * self.strengths
        wrong = (y[:, None] + 1 + providers % (self.n_classes - 1)) % self.n_classes
        logits[rows, providers, wrong] += hard[:, None] * self.confusion_bias
        probabilities = softmax(
            logits / np.asarray(self.temperatures)[None, :, None], axis=2
        )
        for j, (votes, seed) in enumerate(
            zip(self.votes, vote_seeds, strict=True), start=1
        ):
            if votes:
                probabilities[:, j] = _voted_probabilities(
                    probabilities[:, j], votes, seed
                )
        return probabilities, y, {"hard": hard}
