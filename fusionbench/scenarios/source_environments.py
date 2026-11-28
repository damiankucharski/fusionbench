"""Matched source environments and exact targets for the rare-shortcut scenario."""

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from fusionbench.scenarios.rare_shortcut_noise import RareShortcutNoise

SOURCE_ENVIRONMENT_PRESETS = MappingProxyType(
    {
        "homogeneous": ((0.5, 0.5), (0.5, 0.5)),
        "varied": ((0.5, 0.5), (0.0, 1.0)),
        "case_mix": ((0.15, 0.85), (0.5, 0.5)),
        "mixed": ((0.15, 0.85), (0.0, 1.0)),
    }
)


def _reports(x, order, original_scale):
    x = x[:, order, :]
    if original_scale:
        return x
    positive = np.where(x[:, :, 1] > 0.5, 0.8, 0.2)
    return np.stack((1 - positive, positive), axis=2)


@dataclass(frozen=True, eq=False)
class ExactTarget:
    """All 16 observed report patterns with their mass and positive-class probability."""

    x: np.ndarray
    mass: np.ndarray
    q: np.ndarray
    context: np.ndarray


@dataclass(frozen=True, eq=False)
class SourceEnvironmentBank:
    """Source data in anonymous provider order and a matched target-law constructor.

    Fit methods can use x, y, the source environment identifiers, and the binary
    context read from the observed cue. The structural role indices are for
    auditing recovery after fitting. Neither an environment ID nor an outcome
    is required to look up the context of a target example.
    """

    x: np.ndarray
    y: np.ndarray
    env: np.ndarray
    context: np.ndarray
    provider_order: tuple[int, ...]
    original_scale: bool

    @property
    def specialist_index(self) -> int:
        return self.provider_order.index(1)

    @property
    def shortcut_index(self) -> int:
        return self.provider_order.index(3)

    def target(self, region, shortcut) -> ExactTarget:
        """Enumerate a noise-free target law with the source's provider ordering."""
        law = RareShortcutNoise(
            region_test=region, shortcut_test=shortcut, provider_noise_test=0.0
        )
        x, mass, q = law.support("test")
        return ExactTarget(
            _reports(x, self.provider_order, self.original_scale),
            mass,
            q,
            x[:, 2, 1] > 0.5,
        )


def _row_counts(n_rows, n_env):
    if isinstance(n_rows, Integral) and not isinstance(n_rows, bool):
        total = int(n_rows)
        if total < n_env:
            raise ValueError("n_rows must provide at least one row per environment")
        base, rest = divmod(total, n_env)
        return [base + int(i < rest) for i in range(n_env)]
    try:
        counts = tuple(n_rows)
    except TypeError as exc:
        raise ValueError("n_rows must be a total or per-environment counts") from exc
    if len(counts) != n_env or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in counts
    ):
        raise ValueError("Provide one positive row count per environment")
    return [int(n) for n in counts]


def source_environment_bank(
    *, region, shortcut, n_rows, seed, original_scale=False, context_noise=0.0
) -> SourceEnvironmentBank:
    """Draw labelled environments from the public RareShortcutNoise scenario.

    region and shortcut specify parallel sequences of regional prevalence and
    shortcut activity. An integer n_rows is split as equally as possible,
    giving any remainder to the first environments. A sequence gives their
    exact row counts. The four provider roles have uniform report confidence
    .8 unless original_scale=True restores their published magnitudes. There is
    no added vote or label noise. context_noise is an independent flip
    probability on the supplied *source* context only. The cue in x stays
    observable. Matching seed and counts couple clean labels and random draws.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(original_scale, (bool, np.bool_)):
        raise TypeError("original_scale must be boolean")
    if (
        isinstance(context_noise, (bool, np.bool_))
        or not isinstance(context_noise, Real)
        or not 0 <= context_noise <= 1
    ):
        raise ValueError("context_noise must be a probability in [0, 1]")
    try:
        region, shortcut = tuple(region), tuple(shortcut)
    except TypeError as exc:
        raise ValueError(
            "region and shortcut must be per-environment sequences"
        ) from exc
    if not region or len(region) != len(shortcut):
        raise ValueError("region and shortcut need one value per environment")

    counts = _row_counts(n_rows, len(region))
    order = tuple(int(i) for i in np.random.default_rng([seed, 1]).permutation(4))
    xs, ys = [], []
    # Four balanced blocks permit a two-environment source to be subdivided
    # into four with exactly the same sampled individuals, not just the same laws.
    # A block never spans environments; its first row identifies an independent RNG.
    block_size = (sum(counts) + 3) // 4
    start = 0
    for pi, activity, count in zip(region, shortcut, counts, strict=True):
        law = RareShortcutNoise(region_fit=pi, shortcut_fit=activity)
        stop = start + count
        while start < stop:
            block = min(stop - start, block_size - start % block_size)
            x, y, _ = law.sample_split(
                block, np.random.default_rng([seed, 0, start]), "fit"
            )
            xs.append(x)
            ys.append(y)
            start += block
    take = np.random.default_rng([seed, 2]).permutation(sum(counts))
    x = np.concatenate(xs)[take]
    flip = np.random.default_rng([seed, 3]).random(len(x)) < context_noise
    return SourceEnvironmentBank(
        x=_reports(x, order, original_scale),
        y=np.concatenate(ys)[take],
        env=np.repeat(np.arange(len(region)), counts)[take],
        context=(x[:, 2, 1] > 0.5) ^ flip,
        provider_order=order,
        original_scale=bool(original_scale),
    )


def source_environment_preset(name, *, seed, n_rows=512, **options):
    """Draw one of the four documented source profiles with anonymous providers."""
    region, shortcut = SOURCE_ENVIRONMENT_PRESETS[name]
    return source_environment_bank(
        region=region, shortcut=shortcut, n_rows=n_rows, seed=seed, **options
    )
