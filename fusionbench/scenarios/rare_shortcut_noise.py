"""Regional expertise with independently shifted shortcut, reports and labels."""

from numbers import Integral, Real

import numpy as np

from fusionbench.scenarios.base import Scenario
from fusionbench.scenarios.calibrated import to_binary_proba
from fusionbench.scenarios.rare_specialist import RareSpecialist


class RareShortcutNoise(Scenario):
    """Pair clean latent labels and random draws across all parameter settings.

    Each provider-noise draw flips one generalist or specialist vote. An active
    shortcut matches the clean label with probability .9 and an inactive one
    emits an independent fair sign. The cue identifies the region exactly.
    Symmetric label noise is drawn after the reports and changes the observed
    outcome. Each fit/test parameter is a probability in [0, 1].
    """

    n_classes = 2
    primary_metric = "brier"
    evaluation_group_fields = ("region",)
    oracle_description = (
        "Posterior for the observed label given four reports and the split's known law"
    )

    def __init__(
        self,
        *,
        region_fit=0.05,
        region_test=0.5,
        shortcut_fit=1.0,
        shortcut_test=0.0,
        provider_noise_fit=0.0,
        provider_noise_test=0.20,
        label_noise_fit=0.0,
        label_noise_test=0.0,
        name=None,
    ):
        knobs = {
            "region_fit": region_fit,
            "region_test": region_test,
            "shortcut_fit": shortcut_fit,
            "shortcut_test": shortcut_test,
            "provider_noise_fit": provider_noise_fit,
            "provider_noise_test": provider_noise_test,
            "label_noise_fit": label_noise_fit,
            "label_noise_test": label_noise_test,
        }
        for key, value in knobs.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be a probability in [0, 1]")
        super().__init__(name=name, **{key: float(v) for key, v in knobs.items()})
        self._regional = {
            split: RareSpecialist(
                region_prevalence=self.knobs[f"region_{split}"],
                generalist_accuracy=0.75,
                specialist_gain=0.45,
                cue_accuracy=1.0,
            )
            for split in ("fit", "test")
        }

    def model_ids(self):
        return [*self._regional["fit"].model_ids(), "shortcut"]

    def _settings(self, split):
        if split not in ("fit", "test"):
            raise ValueError("split must be 'fit' or 'test'")
        return (
            self._regional[split],
            self.knobs[f"shortcut_{split}"],
            self.knobs[f"provider_noise_{split}"],
            self.knobs[f"label_noise_{split}"],
        )

    def sample(self, n, rng):
        return self.sample_split(n, rng, "fit")

    def sample_split(self, n, rng, split):
        regional, share, eta, label_noise = self._settings(split)
        reports, clean_y, latent = regional.sample(n, rng)
        scores = reports[:, :, 1].copy()
        flips = rng.random((n, 2)) < eta
        scores[:, :2] = np.where(flips, 1 - scores[:, :2], scores[:, :2])
        active = rng.random(n) < share
        shortcut_vote = np.where(
            active, clean_y ^ (rng.random(n) >= 0.9), rng.integers(2, size=n)
        )
        positive = np.column_stack((scores, np.where(shortcut_vote, 0.9, 1 - 0.9)))
        observed_y = clean_y ^ (rng.random(n) < label_noise)
        return (
            np.stack((1 - positive, positive), axis=-1),
            observed_y,
            {**latent, "split": split},
        )

    def support(self, split, *, region=None):
        """Return (preds[16,4,2], mass[16], observed-label posterior[16]).

        Rows are lexicographic vote signs 0000 through 1111 in model_ids()
        order. Rows of zero mass have posterior .5 by convention. Set region
        to 0 or 1 to hold the region fixed and normalize its distribution,
        including a counterfactual region absent from the original split.
        """
        regional, share, eta, label_noise = self._settings(split)
        if region is not None and (
            not isinstance(region, (Integral, np.bool_)) or region not in (0, 1)
        ):
            raise ValueError("region must be None, 0 or 1")
        prior = regional.region_prevalence if region is None else int(region)
        signs = np.indices((2, 2, 2, 2), dtype=np.int8).reshape(4, -1).T
        g, s, cue, h = signs.T
        region_mass = np.where(cue, prior, 1 - prior)
        generalist = eta + (1 - 2 * eta) * regional.generalist_accuracy
        specialist = eta + (1 - 2 * eta) * (0.5 + cue * regional.specialist_gain)
        shortcut = 0.5 + share * (0.9 - 0.5)
        joint = [
            0.5
            * region_mass
            * np.where(g == y, generalist, 1 - generalist)
            * np.where(s == y, specialist, 1 - specialist)
            * np.where(h == y, shortcut, 1 - shortcut)
            for y in (0, 1)
        ]
        mass = joint[0] + joint[1]
        q = np.divide(
            (1 - label_noise) * joint[1] + label_noise * joint[0],
            mass,
            out=np.full(16, 0.5),
            where=mass > 0,
        )
        confidence = np.array([0.75, 0.75, 0.75, 0.9])
        positive = np.where(signs, confidence, 1 - confidence)
        return np.stack((1 - positive, positive), axis=-1), mass, q

    def bayes_oracle(self, preds, latent):
        p = np.asarray(preds)
        if p.ndim != 3 or p.shape[1:] != (4, 2):
            raise ValueError("Expected binary predictions with shape (rows, 4, 2)")
        _, _, q = self.support(latent["split"])
        indices = (p[:, :, 1] > 0.5) @ np.array([8, 4, 2, 1])
        return to_binary_proba(q[indices])
