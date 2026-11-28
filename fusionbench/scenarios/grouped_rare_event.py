"""Clustered binary reports with rare events and controllable provider errors."""

from numbers import Integral, Real

import numpy as np
from scipy.special import expit

from fusionbench.scenarios.base import Scenario


class GroupedRareEvent(Scenario):
    """Generate independent scenes, each containing one or more dependent reports.

    A shared scene signal and private row signal determine a latent event probability.
    Providers observe noisy logits. Their noise can persist across a scene and be
    common across providers. One provider can approach the latent probability in a
    subset of scenes. The prediction tensor contains only the eight provider reports.
    """

    n_classes = 2
    primary_metric = "brier"
    evaluation_group_fields = ("scene_id", "specialist_region")
    oracle_description = "The latent event probability is available from sample() for conditional-risk diagnostics."
    scenario_id = "grouped_rare_event"
    n_providers = 8
    specialist_index = 3

    def __init__(
        self,
        scene_size=8,
        event_offset=2.0,
        heterogeneity=0.5,
        common_error_share=0.35,
        specialist_strength=0.75,
        specialist_rate=0.2,
        scene_error_share=0.85,
        bloc_size=8,
        name=None,
    ):
        if (
            isinstance(scene_size, (bool, np.bool_))
            or not isinstance(scene_size, Integral)
            or scene_size < 1
        ):
            raise ValueError("scene_size must be a positive integer")
        if (
            isinstance(event_offset, (bool, np.bool_))
            or not isinstance(event_offset, Real)
            or not np.isfinite(event_offset)
        ):
            raise ValueError("event_offset must be a finite real number")
        if (
            isinstance(bloc_size, (bool, np.bool_))
            or not isinstance(bloc_size, Integral)
            or not 0 <= bloc_size <= self.n_providers
        ):
            raise ValueError("bloc_size must be an integer from zero through eight")
        shares = {
            "heterogeneity": heterogeneity,
            "common_error_share": common_error_share,
            "specialist_strength": specialist_strength,
            "specialist_rate": specialist_rate,
            "scene_error_share": scene_error_share,
        }
        for key, value in shares.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be in [0, 1]")
        super().__init__(
            name=name or self.scenario_id,
            bloc_size=int(bloc_size),
            scene_size=int(scene_size),
            event_offset=float(event_offset),
            **{k: float(v) for k, v in shares.items()},
        )

    def model_ids(self):
        return [f"provider_{i}" for i in range(self.n_providers)]

    def sample(self, n, rng):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        k = self.knobs
        scene_id = np.arange(n) // k["scene_size"]
        n_scenes = int(scene_id[-1]) + 1
        signal = (
            0.9 * rng.standard_normal(n_scenes)[scene_id]
            + 0.8 * rng.standard_normal(n)
            - k["event_offset"]
        )
        true_logit = 1.8 * signal
        posterior = expit(true_logit)
        labels = (rng.random(n) < posterior).astype(np.int64)
        region = (rng.random(n_scenes) < k["specialist_rate"])[scene_id]

        # Unit-variance components. Draw all components even when a switch is zero
        # so fixed-seed interventions share the same signals and noise realizations.
        def residual(columns):
            across_scene = rng.standard_normal((n_scenes, columns))[scene_id]
            within_scene = rng.standard_normal((n, columns))
            share = k["scene_error_share"]
            return np.sqrt(share) * across_scene + np.sqrt(1 - share) * within_scene

        common = k["common_error_share"]
        shared = residual(1)
        private = residual(self.n_providers)
        noise = private.copy()
        bloc = slice(0, k["bloc_size"])
        noise[:, bloc] = (
            np.sqrt(common) * shared + np.sqrt(1 - common) * private[:, bloc]
        )
        relative_sd = np.exp(
            k["heterogeneity"] * np.linspace(-1.0, 1.0, self.n_providers)
        )
        relative_sd /= np.sqrt(np.mean(relative_sd**2))
        # Match the marginal residual variance of the earlier scene-pool control.
        sigma = np.sqrt(0.5**2 + 0.5**2 + 0.85**2 + 0.5**2)
        logits = 1.3 * signal[:, None] + sigma * relative_sd * noise
        specialist = self.specialist_index
        strength = k["specialist_strength"]
        logits[region, specialist] = (1 - strength) * logits[
            region, specialist
        ] + strength * true_logit[region]
        p = expit(logits)
        return (
            np.stack((1 - p, p), axis=-1),
            labels,
            {
                "scene_id": scene_id,
                "posterior": posterior,
                "specialist_region": region,
                "signal": signal,
            },
        )


class SharedErrorBloc(GroupedRareEvent):
    """Six equally accurate providers share a residual; two offer independent evidence."""

    scenario_id = "shared_error_bloc"

    def __init__(
        self,
        heterogeneity=0.0,
        common_error_share=0.85,
        specialist_strength=0.0,
        bloc_size=6,
        **kwargs,
    ):
        super().__init__(
            heterogeneity=heterogeneity,
            common_error_share=common_error_share,
            specialist_strength=specialist_strength,
            bloc_size=bloc_size,
            **kwargs,
        )


class RareSceneSpecialist(GroupedRareEvent):
    """One provider is conditionally exact on a minority of the independent scenes."""

    scenario_id = "rare_scene_specialist"

    def __init__(
        self,
        heterogeneity=0.0,
        common_error_share=0.0,
        specialist_strength=1.0,
        **kwargs,
    ):
        super().__init__(
            heterogeneity=heterogeneity,
            common_error_share=common_error_share,
            specialist_strength=specialist_strength,
            **kwargs,
        )
