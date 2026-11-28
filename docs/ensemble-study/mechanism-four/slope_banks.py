"""Fresh paired synthetic draws for the provider logit-slope stress experiment."""

from dataclasses import replace

import numpy as np
from banks_study import (
    LEGACY,
    Bank,
    _clustered,
    _digest,
    _factorial,
    _group_ids,
    _standard,
)

from fusionbench.scenarios.grouped_rare_event import GroupedRareEvent

SLOPES = (1.3, 1.8, 2.3)
_BASE = 496000
_KNOBS = (
    "heterogeneity",
    "common_error_share",
    "specialist_strength",
    "specialist_rate",
    "scene_error_share",
    "event_offset",
    "scene_size",
)


def _vary_slope(bank):
    """Regenerate provider reports on the same truths, regimes and scene random streams."""
    seed = bank.context["seed"]
    knobs = {key: bank.context[key] for key in _KNOBS}
    for slope in SLOPES:
        law = GroupedRareEvent(**knobs, provider_signal_slope=slope)
        x, y, fit = law.sample(len(bank.y), np.random.default_rng([seed, 0]))
        ax, ay, audit = law.sample(len(bank.audit_y), np.random.default_rng([seed, 1]))
        np.testing.assert_array_equal(y, bank.y)
        np.testing.assert_array_equal(ay, bank.audit_y)
        np.testing.assert_array_equal(_group_ids(fit["scene_id"], 0), bank.groups)
        np.testing.assert_array_equal(
            _group_ids(audit["scene_id"], 1), bank.audit_groups
        )
        np.testing.assert_array_equal(audit["specialist_region"], bank.audit_regime)
        np.testing.assert_array_equal(audit["posterior"], bank.q)
        if slope == 1.3:
            np.testing.assert_array_equal(x, bank.x)
            np.testing.assert_array_equal(ax, bank.audit_x)
        context = {
            **bank.context,
            "slope": slope,
            "fit_x_sha256": _digest(x),
            "audit_x_sha256": _digest(ax),
        }
        yield replace(bank, x=x, audit_x=ax, context=context)


def banks_slope():
    """Yield 288 paired factorial fits, 48 legacy controls, and 90 standard-suite fits."""
    for base_bank in _factorial(5, range(_BASE, _BASE + 12)):
        yield from _vary_slope(base_bank)
    for bank in _clustered(5, range(_BASE + 100, _BASE + 112)):
        yield replace(bank, context={**bank.context, "slope": None})
    for bank in LEGACY["_controlled"](5, range(_BASE + 200, _BASE + 212)):
        if bank.context["family"] in {"weak", "strong"}:
            bank.context["slope"] = None
            yield Bank(**vars(bank))
    for bank in _standard(5, range(_BASE + 400, _BASE + 403)):
        yield replace(bank, context={**bank.context, "slope": None})
