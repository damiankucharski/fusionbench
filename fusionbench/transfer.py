"""Run a paired transfer grid from a versioned preset or a plain YAML mapping."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata, resources
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from fusionbench import __version__
from fusionbench.eval.transfer_grid import (
    MODES,
    STREAMS,
    brier,
    development,
    iter_test_conditions,
    noisy_labels,
)
from fusionbench.methods.averaging import TopN
from fusionbench.methods.registry import FACTORIES
from fusionbench.scenarios.blocks import SHIFTS, default_target_shift, draw_block  # noqa: F401 (public historical shift catalog)
from fusionbench.scenarios.registry import SCENARIOS, _validated_defaults

PRESETS = ("smoke", "standard", "standard-v1", "standard-v2", "standard-v3", "standard-v4", "standard-v5")
CONFIG_KEYS = {
    "schema_version",
    "base_seed",
    "repeats",
    "fit_rows",
    "validation_rows",
    "test_rows",
    "dev_noise",
    "modes",
    "severities",
    "methods",
    "scenarios",
}


def load_preset(name="smoke"):
    """Read a version-1 configuration shipped inside the installed package."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    version = {"smoke": "smoke-v1", "standard": "standard-v5"}.get(name, name)
    path = resources.files("fusionbench").joinpath("presets", f"{version}.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _integer(value, minimum, field):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _list(value, field, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(
            f"{field} must be a {'possibly empty' if empty else 'nonempty'} list"
        )
    if any(isinstance(v, (dict, list)) for v in value) or len(set(value)) != len(value):
        raise ValueError(f"{field} must contain distinct scalar values")
    return list(value)


def _probabilities(values, field, *, zero):
    items = _list(values, field)
    if any(
        type(v) not in (int, float) or not (0 <= v <= 1) or (not zero and v == 0)
        for v in items
    ):
        raise ValueError(f"{field} values must be in {'[0, 1]' if zero else '(0, 1]'}")
    return list(map(float, items))


def _mapping(value, field):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{field} must be a mapping with string keys")
    return dict(value)


def _known(items, universe, field):
    missing = [k for k in items if k not in universe]
    if missing:
        raise ValueError(
            f"Unknown {field} {missing!r}. Available: {', '.join(universe)}"
        )


def _scenario_spec(name, spec):
    opts = _mapping(spec, f"scenarios.{name}")
    _known(opts, ("seed_slot", "knobs", "canonical_shift"), "scenario field")
    if "seed_slot" not in opts:
        raise ValueError(f"Missing stable seed_slot for {name}")
    slot = _integer(opts["seed_slot"], 0, f"{name}.seed_slot")
    knobs = _mapping(opts.get("knobs", {}), f"{name}.knobs")
    endpoint = opts.get("canonical_shift", default_target_shift(name))
    if endpoint is not None:
        endpoint = _mapping(endpoint, f"{name}.canonical_shift")
    _validated_defaults(name, {**knobs, **(endpoint or {})})
    return {"seed_slot": slot, "knobs": knobs, "canonical_shift": endpoint}


def resolve_config(config):
    """Validate and resolve a configuration against the current Python registries."""
    cfg = _mapping(config, "config")
    cfg.setdefault("validation_rows", 192)  # Earlier v1 configurations omit this field.
    _known(cfg, sorted(CONFIG_KEYS), "config field")
    if missing := CONFIG_KEYS - set(cfg):
        raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Supported transfer schema_version is 1")
    for field, lower in (
        ("base_seed", 0),
        ("repeats", 1),
        ("fit_rows", 2),
        ("validation_rows", 1),
        ("test_rows", 2),
    ):
        cfg[field] = _integer(cfg[field], lower, field)
    cfg["dev_noise"] = _probabilities(cfg["dev_noise"], "dev_noise", zero=True)
    cfg["severities"] = _probabilities(cfg["severities"], "severities", zero=False)
    cfg["modes"] = _list(cfg["modes"], "modes", empty=True)
    _known(cfg["modes"], MODES, "mode")
    methods = _list(cfg["methods"], "methods")
    _known(methods, FACTORIES, "method")
    if any(name.startswith("okapi") for name in methods):
        for control in ("simple_average", "top_n"):
            if control not in methods:
                methods.append(control)
        if not {"stacking_lr", "stacking_logit", "stacking_clr"}.intersection(methods):
            methods.append("stacking_lr")
    for name in methods:
        if not FACTORIES[name]().available():
            raise ValueError(
                f"Requested method {name!r} is unavailable. Install its optional "
                "dependencies or remove it from methods."
            )
    cfg["methods"] = methods
    scenarios = _mapping(cfg["scenarios"], "scenarios")
    if not scenarios:
        raise ValueError("Select at least one scenario")
    _known(scenarios, SCENARIOS, "scenario")
    cfg["scenarios"] = {
        name: _scenario_spec(name, spec) for name, spec in scenarios.items()
    }
    starts = sorted(_first_seed(cfg, s) for s in cfg["scenarios"].values())
    if any(b - a < cfg["repeats"] for a, b in pairwise(starts)):
        raise ValueError("Scenario seed ranges overlap; use separate seed_slot values")
    return cfg


def _first_seed(cfg, scenario):
    return cfg["base_seed"] + 1000 * scenario["seed_slot"]


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _software():
    versions = {"fusionbench": __version__, "python": platform.python_version()}
    for pkg in ("numpy", "scipy", "scikit-learn", "pandas", "PyYAML", "okapi", "torch"):
        try:
            dist = metadata.distribution(pkg)
        except metadata.PackageNotFoundError:
            continue
        versions[pkg] = dist.version
        origin = dist.read_text("direct_url.json")
        if origin:
            vcs = json.loads(origin).get("vcs_info", {})
            if "commit_id" in vcs:
                versions[pkg + "_commit"] = vcs["commit_id"]
    source = Path(__file__).parent
    checksum = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        checksum.update(path.relative_to(source).as_posix().encode() + b"\0")
        checksum.update(hashlib.sha256(path.read_bytes()).digest())
    versions["fusionbench_python_sha256"] = checksum.hexdigest()
    return versions


def _bank(name, spec, seed, repeat, config):
    knobs = spec["knobs"]
    scenario = SCENARIOS[name](**knobs)
    x, y = development(name, seed, config["fit_rows"], scenario_kwargs=knobs)
    x.setflags(write=False)
    vx, vy, _ = draw_block(
        scenario, seed, STREAMS["source_validation"], config["validation_rows"]
    )
    vx.setflags(write=False)
    models, fits, raw, selections = {}, [], [], []
    winners = {}
    bank = {
        "scenario": name,
        "replicate": repeat,
        "data_seed": seed,
        "providers": list(scenario.model_ids()),
        "classes": scenario.n_classes,
        "development_inputs_sha256": _digest(x),
        "annotations": {},
        "validation_inputs_sha256": _digest(vx),
        "validation_annotations": {},
        "test_conditions": [],
    }
    identifiers = {"scenario": name, "replicate": repeat, "data_seed": seed}
    for noise in config["dev_noise"]:
        labels = noisy_labels(y, scenario.n_classes, seed, noise)
        labels.setflags(write=False)
        bank["annotations"][str(noise)] = _digest(labels)
        validation_labels = noisy_labels(
            vy,
            scenario.n_classes,
            seed,
            noise,
            stream=STREAMS["validation_annotation_noise"],
        )
        validation_labels.setflags(write=False)
        bank["validation_annotations"][str(noise)] = _digest(validation_labels)
        validation_losses = {}
        for method_id in config["methods"]:
            model = FACTORIES[method_id]()
            method_seed = None
            if hasattr(model, "seed"):
                model.seed = method_seed = seed
            extra = {"primary_metric": "brier"} if isinstance(model, TopN) else {}
            start = perf_counter()
            try:
                model.fit(
                    x, labels, task=scenario.task, n_classes=scenario.n_classes, **extra
                )
                elapsed = perf_counter() - start
                fit_loss = brier(model.predict(x), labels)
            except Exception as exc:
                raise ValueError(
                    f"Fit failed for {name}/{method_id} at seed={seed}, "
                    f"dev_noise={noise}: {exc}"
                ) from exc
            try:
                validation_loss = brier(model.predict(vx), validation_labels)
            except Exception as exc:
                raise ValueError(
                    f"Validation failed for {name}/{method_id} at seed={seed}, "
                    f"dev_noise={noise}: {exc}"
                ) from exc
            validation_losses[method_id] = validation_loss
            models[noise, method_id] = model
            fits.append(
                {
                    **identifiers,
                    "dev_noise": noise,
                    "method": method_id,
                    "method_seed": method_seed,
                    "fit_rows": len(y),
                    "fit_brier": fit_loss,
                    "validation_rows": len(vy),
                    "validation_brier": validation_loss,
                    "fit_seconds": elapsed,
                    "n_selected": getattr(model, "n_selected_", None),
                }
            )
        # Strict empirical minimum; insertion order breaks exact ties.
        winner = min(validation_losses, key=validation_losses.__getitem__)
        winners[noise] = (winner, validation_losses[winner])
    clean_risk = {}
    modes = config["modes"]
    # A resolved null endpoint explicitly suppresses canonical shift for this scenario.
    if spec["canonical_shift"] is None:
        modes = [m for m in modes if m != "canonical_shift"]
    conditions = iter_test_conditions(
        name,
        seed,
        config["test_rows"],
        scenario_kwargs=knobs,
        severities=config["severities"],
        modes=modes,
        canonical_shift=spec["canonical_shift"],
    )
    for mode, severity, preds, target in conditions:
        preds.setflags(write=False)
        bank["test_conditions"].append(
            {
                "mode": mode,
                "severity": severity,
                "inputs_sha256": _digest(preds),
                "labels_sha256": _digest(target),
            }
        )
        risks = {noise: {} for noise in config["dev_noise"]}
        for key, model in models.items():
            noise, method_id = key
            try:
                risk = brier(model.predict(preds), target)
            except Exception as exc:
                raise ValueError(
                    f"Prediction failed for {name}/{method_id} at seed={seed}, "
                    f"dev_noise={noise}, {mode}={severity}: {exc}"
                ) from exc
            risks[noise][method_id] = risk
            if mode == "clean":
                clean_risk[key] = risk
            raw.append(
                {
                    **identifiers,
                    "dev_noise": noise,
                    "method": method_id,
                    "mode": mode,
                    "severity": severity,
                    "test_rows": len(target),
                    "brier": risk,
                    "delta_from_clean": risk - clean_risk[key],
                }
            )
        for noise, test_losses in risks.items():
            selected, validation_brier = winners[noise]
            test_best = min(test_losses, key=test_losses.__getitem__)
            excess = test_losses[selected] - test_losses[test_best]
            selections.append(
                {
                    **identifiers,
                    "dev_noise": noise,
                    "mode": mode,
                    "severity": severity,
                    "n_methods": len(test_losses),
                    "validation_best_method": selected,
                    "validation_best_brier": validation_brier,
                    "selected_test_brier": test_losses[selected],
                    "test_best_method": test_best,
                    "test_best_brier": test_losses[test_best],
                    "test_excess": excess,
                    "test_hit": excess <= 1e-12,
                }
            )
    return fits, raw, selections, bank


def summarize(raw):
    """Weight replicates within scenario, then weight represented scenarios equally."""
    group = ["method", "dev_noise", "mode", "severity"]
    per = raw.groupby(group + ["scenario"], as_index=False).agg(
        brier=("brier", "mean"),
        delta_from_clean=("delta_from_clean", "mean"),
        banks=("replicate", "size"),
    )
    scores = per.groupby(group, as_index=False).agg(
        brier=("brier", "mean"),
        delta_from_clean=("delta_from_clean", "mean"),
        scenarios=("scenario", "size"),
        banks=("banks", "sum"),
    )
    scenario_means = per.groupby(
        ["method", "dev_noise", "scenario"], as_index=False
    ).brier.mean()
    overall = scenario_means.groupby(["method", "dev_noise"], as_index=False).agg(
        brier=("brier", "mean"), scenarios=("scenario", "size")
    )
    return scores, overall


def run_transfer(config, out):
    """Fit registry methods once per noisy bank, then score paired deployment conditions."""
    cfg = resolve_config(config)
    fits, records, selections, banks = [], [], [], []
    for name, spec in cfg["scenarios"].items():
        for rep in range(cfg["repeats"]):
            seed = _first_seed(cfg, spec) + rep
            f, r, choices, bank = _bank(name, spec, seed, rep, cfg)
            fits.extend(f)
            records.extend(r)
            selections.extend(choices)
            banks.append(bank)
    raw = pd.DataFrame(records)
    summary, overall = summarize(raw)
    manifest = {
        "schema_version": 1,
        "config": cfg,
        "software": _software(),
        "seed_formula": "base_seed + 1000 * scenario.seed_slot + replicate",
        "streams": STREAMS,
        "banks": banks,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        for file, frame in (
            ("scores", raw),
            ("fits", pd.DataFrame(fits)),
            ("selection", pd.DataFrame(selections)),
            ("summary", summary),
            ("overall", overall),
        ):
            frame.to_csv(directory / (file + ".csv"), index=False)
        (directory / "resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for file in (
            "scores.csv",
            "fits.csv",
            "selection.csv",
            "summary.csv",
            "overall.csv",
            "resolved.yaml",
            "manifest.json",
        ):
            (directory / file).replace(out / file)
    return out
