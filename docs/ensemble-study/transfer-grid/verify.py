"""Check paired grid coverage, frozen selection diagnostics and replicate-level contrasts."""

import csv
import gzip
import json
import runpy
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
from scipy.stats import t

from fusionbench.eval.full_suite_point import SHIFTS
from fusionbench.scenarios.registry import SCENARIOS as AVAILABLE_SCENARIOS

FOLDER = Path(__file__).resolve().parent
# Preserve the independently frozen seed-to-scenario map if the registry grows.
SCENARIOS = {
    name: AVAILABLE_SCENARIOS[name]
    for name in (
        "committee",
        "deadweight",
        "needle",
        "saturated",
        "correlated",
        "calibrated_suppressor",
        "polarity_fault",
        "hardveto",
        "hardrescue",
        "weakest_link",
        "two_pathways",
        "parity",
        "compound",
        "gated_experts",
        "native_switch",
        "compositional_switch",
        "native_close_switch",
        "miscalibration",
        "multiclass_committee",
    )
}
NATIVE = runpy.run_path(str(FOLDER.parent / "native-pattern" / "verify.py"))[
    "native_prediction"
]
METHODS = (
    "deployed",
    "robust",
    "gp",
    "flat",
    "augmented",
    "half",
    "prob_lr",
    "logit",
    "mean",
    "top_n",
)
NOISE = (0.0, 0.15, 0.30)
CANDIDATES = ("flat", "augmented", "gp", "half")


def rows(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(FOLDER / name, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def near(actual, expected, tol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tol)


def interval(sample):
    average = fmean(sample)
    width = stdev(sample) * float(t.ppf(0.975, len(sample) - 1)) / sqrt(len(sample))
    return average, average - width, average + width


SEVERITIES = (0.25, 0.50, 1.0)


def conditions(name):
    modes = ["provider_swap", "log_jitter"]
    if name in SHIFTS:
        modes.append("canonical_shift")
    return [("clean", 0.0)] + [(mode, dose) for mode in modes for dose in SEVERITIES]


def data(name, seed, stream, nrows, shifted=False):
    generator = SCENARIOS[name](**(SHIFTS[name] if shifted else {}))
    x, y, _ = generator._generate(nrows, np.random.default_rng([seed, stream]))
    return x, np.asarray(y).ravel()


def replay_test(name, seed, nrows):
    """Recreate paired clean, provider-corrupted, jittered and mixed-domain examples."""
    clean, y = data(name, seed, 1, nrows)
    yield "clean", 0.0, clean, y
    u = np.random.default_rng([seed, 11]).random(nrows)
    rng = np.random.default_rng([seed, 12])
    column = rng.integers(clean.shape[1])
    ix = np.arange(nrows)
    donors = rng.permutation(ix)
    while (ix == donors).any():
        donors = rng.permutation(ix)
    for s in SEVERITIES:
        mask = u < s
        altered = clean.copy()
        altered[mask, column] = clean[donors[mask], column]
        yield "provider_swap", s, altered, y
    g = np.random.default_rng([seed, 13]).standard_normal(clean.shape)
    log = np.log(np.clip(clean, np.finfo(float).tiny, None))
    log -= log.mean(axis=2, keepdims=True)
    for s in SEVERITIES:
        logits = log + (1.5 * s) * g
        shifted = np.exp(logits - logits.max(axis=2, keepdims=True))
        yield "log_jitter", s, shifted / shifted.sum(axis=2, keepdims=True), y
    if name in SHIFTS:
        other, labels = data(name, seed, 2, nrows, shifted=True)
        for s in SEVERITIES:
            mask = u < s
            yield (
                "canonical_shift",
                s,
                np.where(mask[:, None, None], other, clean),
                np.where(mask, labels, y),
            )


def cell(row):
    return (
        row["scenario"],
        int(row["replicate"]),
        float(row["noise"]),
        row["mode"],
        float(row["severity"]),
    )


def mean_loss(p, y):
    assert p.ndim == 2 and p.shape[0] == len(y)
    near(p.sum(1), np.ones(len(p)))
    assert np.isfinite(p).all() and p.min() >= -1e-12
    return float(np.mean(np.square(p - np.eye(p.shape[1])[y]).sum(1)))


def macro_bank_means(values):
    """Fixed-family averages within replicates, followed by a paired t interval."""
    runs = sorted({rep for _, rep in values})
    families = sorted({name for name, _ in values})
    assert runs == list(range(8)) and len(values) == len(families) * 8
    return interval([fmean(values[s, r] for s in families) for r in runs])


def compare_half(programs, width):
    modified = json.loads(json.dumps(programs["augmented"]))
    head = modified["children"][-1]
    k = "coefficients" if head["type"] == "BinaryAffineNode" else "weights"
    head[k][width:] = [0.5 * w for w in head[k][width:]]
    assert modified == programs["half"]


def validation_rows(name, seed, q):
    x, y = data(name, seed, 0, 640)
    rng = np.random.default_rng([seed, 10])
    corrupt = rng.random(640) < q
    shifts = rng.integers(1, x.shape[-1], size=640)
    noisy = np.where(corrupt, (y + shifts) % x.shape[-1], y)
    ix = np.random.default_rng([seed, 51]).permutation(640)[448:]
    return x[ix], noisy[ix]


def select_robust(scores):
    threshold = max(s[0] for s in scores.values()) - 0.01
    return max(
        (k for k in CANDIDATES if scores[k][0] >= threshold),
        key=lambda k: (min(scores[k]), scores[k][0]),
    )


def check_raw(scores, diagnostics):
    expected_scores, expected_diag = set(), set()
    transfers = {}
    choices = {}
    for idx, name in enumerate(SCENARIOS):
        for rep in range(8):
            seed = 1700000 + 1000 * idx + rep
            for q in NOISE:
                for mode, dose in conditions(name):
                    k = name, rep, q, mode, dose
                    expected_diag.add(k)
                    diag = diagnostics[k]
                    assert int(diag["data_seed"]) == seed
                    for method in METHODS:
                        expected_scores.add(k + (method,))
                        r = scores[k + (method,)]
                        assert (
                            int(r["data_seed"]) == seed and 0 <= float(r["brier"]) <= 2
                        )
                    old = diag["old_choice"]
                    robust = diag["robust_choice"]
                    v = {m: float(diag["val_" + m]) for m in CANDIDATES}
                    z = {m: float(diag["test_" + m]) for m in CANDIDATES}
                    assert all(0 <= v[m] <= 2 and 0 <= z[m] <= 2 for m in CANDIDATES)
                    assert old in CANDIDATES[:3] and robust in CANDIDATES
                    assert max(CANDIDATES[:3], key=lambda m: 1.0 - 0.5 * v[m]) == old
                    for policy, selected in [("deployed", old), ("robust", robust)]:
                        near(
                            float(scores[k + (policy,)]["brier"]),
                            float(scores[k + (selected,)]["brier"]),
                        )
                    if (name, rep, q) in choices:
                        assert choices[name, rep, q] == (old, robust, v)
                    choices[name, rep, q] = old, robust, v
                    winner = min(CANDIDATES, key=v.get)
                    regret = z[winner] - min(z.values())
                    # This diagnostic compares four identical pre-refit models across domains.
                    transfers[k] = (
                        winner,
                        regret,
                        float(regret <= 1e-12),
                        1 + sum(val < z[winner] - 1e-12 for val in z.values()),
                    )
                    # The test labels and the untrained mean are the same across label-noise levels.
                    near(
                        float(scores[k + ("mean",)]["brier"]),
                        float(scores[name, rep, 0.0, mode, dose, "mean"]["brier"]),
                    )
    assert set(scores) == expected_scores and len(scores) == 38400
    assert set(diagnostics) == expected_diag and len(diagnostics) == 3840
    return choices, transfers


def primary_numeric(loss):
    disturbed = [
        (mode, s) for mode in ("provider_swap", "log_jitter") for s in SEVERITIES
    ]
    primary = {}
    clean = {}
    for name in SCENARIOS:
        for rep in range(8):

            def diff(q, mode, s, name=name, rep=rep):
                tag = name, rep, q, mode, s
                return loss[tag + ("robust",)] - loss[tag + ("deployed",)]

            primary[name, rep] = fmean(
                diff(q, mode, s) for q in NOISE for mode, s in disturbed
            )
            clean[name, rep] = diff(0.0, "clean", 0.0)
    return macro_bank_means(primary), macro_bank_means(clean)


def replay_validation(x, seed):
    rng = np.random.default_rng([seed, 841])
    order = rng.permutation(len(x))
    other = np.empty(len(x), dtype=int)
    other[order] = np.r_[order[-1], order[:-1]]
    providers = rng.integers(x.shape[1], size=len(x))
    bad = np.flatnonzero(rng.random(len(x)) < 0.5)
    swapped = x.copy()
    swapped[bad, providers[bad]] = x[other[bad], providers[bad]]
    power = rng.choice([0.5, 2.0], size=(1, x.shape[1], 1))
    temp = x**power
    return x, swapped, temp / temp.sum(axis=2, keepdims=True)


def replay_samples(raw, diag, costs):
    with gzip.open(FOLDER / "sample_trees.jsonl.gz", "rt") as stream:
        manifests = [json.loads(line) for line in stream]
    indexed = {(r["scenario"], r["replicate"], r["noise"]): r for r in manifests}
    assert len(indexed) == len(manifests) == 57
    replays = 0
    checks = 0
    for idx, name in enumerate(SCENARIOS):
        seed = 1700000 + 1000 * idx
        # Training label noise never changes the target bank or its perturbation.
        blocks = list(replay_test(name, seed, 2048))
        for q in NOISE:
            tag = name, 0, q
            r = indexed[tag]
            assert r["data_seed"] == seed
            x, y = validation_rows(name, seed, q)
            for program_set in ("discovery_trees", "trees"):
                assert set(r[program_set]) == set(CANDIDATES)
                compare_half(r[program_set], x.shape[1])
            assert r["discovery_trees"]["gp"] == r["trees"]["gp"]
            probes = replay_validation(x, seed)
            probe_scores = {
                m: [
                    1.0 - 0.5 * mean_loss(NATIVE(r["discovery_trees"][m], z), y)
                    for z in probes
                ]
                for m in CANDIDATES
            }
            chosen = select_robust(probe_scores)
            assert chosen == r["robust_choice"] and r["old_choice"] == max(
                CANDIDATES[:3], key=lambda m: probe_scores[m][0]
            )
            near(
                [probe_scores[m][0] for m in CANDIDATES],
                [
                    1.0 - 0.5 * float(diag[tag + ("clean", 0.0)]["val_" + m])
                    for m in CANDIDATES
                ],
            )
            stored = json.loads(costs[tag]["probe_scores"])
            near(
                [probe_scores[m] for m in CANDIDATES],
                [stored[m] for m in CANDIDATES],
            )
            checks += len(CANDIDATES) * len(probes)
            for mode, s, z, test_y in blocks:
                k = tag + (mode, s)
                for m in CANDIDATES:
                    v = mean_loss(NATIVE(r["discovery_trees"][m], z), test_y)
                    near(v, float(diag[k]["test_" + m]))
                    loss = mean_loss(NATIVE(r["trees"][m], z), test_y)
                    near(loss, float(raw[k + (m,)]["brier"]))
                    replays += 2
                near(mean_loss(z.mean(1), test_y), float(raw[k + ("mean",)]["brier"]))
    return len(manifests), checks, replays


def audit_costs(choices):
    costs = rows("costs.csv")
    indexed = {
        (r["scenario"], int(r["replicate"]), float(r["noise"])): r for r in costs
    }
    assert len(indexed) == len(costs) == len(choices) == 456 and set(choices) == set(
        indexed
    )
    total_calls = 0
    for (name, rep, q), r in indexed.items():
        idx = list(SCENARIOS).index(name)
        seed = 1700000 + idx * 1000 + rep
        width = len(SCENARIOS[name]().model_ids())
        c = SCENARIOS[name]().n_classes
        assert (
            int(r["data_seed"]) == seed
            and int(r["input_providers"]) == width
            and int(r["classes"]) == c
        )
        assert (
            int(r["development_rows"]) == 640
            and int(r["test_rows_per_condition"]) == 2048
        )
        assert (
            int(r["top_n_prefixes"]) == width and 1 <= int(r["top_n_selected"]) <= width
        )
        assert int(r["gp_discovery_calls"]) == 600 and int(r["conditions"]) == len(
            conditions(name)
        )
        wrong = np.count_nonzero(np.random.default_rng([seed, 10]).random(640) < q)
        assert int(r["training_rows_corrupted"]) == wrong
        old, robust, v = choices[name, rep, q]
        assert (r["old_choice"], r["robust_choice"]) == (old, robust)
        base = json.loads(r["base_counts"])
        transfer = json.loads(r["transfer_counts"])
        assert (
            base["discovery_rows"],
            base["validation_rows"],
            base["refit_rows"],
        ) == (448, 192, 640)
        assert (
            transfer["validation_rows"] == 192
            and transfer["additional_score_calls"] == 9
        )
        assert (
            sum(base["score_calls"].values())
            == transfer["base_score_calls"]
            == int(r["base_score_calls"])
        )
        assert transfer["total_score_calls"] == transfer["base_score_calls"] + 9
        total_calls += transfer["total_score_calls"]
        all_views = json.loads(r["probe_scores"])
        assert robust == select_robust(all_views)
        near(
            [all_views[m][0] for m in CANDIDATES],
            [1.0 - 0.5 * v[m] for m in CANDIDATES],
        )
    return indexed, total_calls


def transfer_summaries(transfers):
    out = {}
    for q in NOISE:
        for mode, dose in conditions("compositional_switch"):
            names = SHIFTS if mode == "canonical_shift" else SCENARIOS
            for metric, idx in [("regret", 1), ("hit", 2), ("rank", 3)]:
                observations = {
                    (name, r): transfers[name, r, q, mode, dose][idx]
                    for name in names
                    for r in range(8)
                }
                out[q, mode, dose, metric] = macro_bank_means(observations)
    return out


def audit_summaries(loss, transfers):
    metrics = (*METHODS, "regret", "validation_hit")
    refs = ("deployed", "prob_lr", "logit", "top_n", "mean")
    per_condition = defaultdict(list)
    per_family = defaultdict(list)
    per_panel = defaultdict(lambda: defaultdict(list))
    for key, (_, regret, hit, _) in transfers.items():
        name, rep, noise, mode, severity = key
        values = {m: loss[key + (m,)] for m in METHODS} | {
            "regret": regret,
            "validation_hit": hit,
        }
        per_condition[mode, noise, severity].append(values)
        if mode == "clean":
            panels = ["all clean tests"] + (
                ["clean source and test"] if noise == 0.0 else []
            )
        elif mode == "canonical_shift":
            panels = ["scenario-specific shift"]
        else:
            panels = ["generic disturbances", mode]
            per_family[name].append(values)
        for panel in panels:
            per_panel[panel][name, rep].append(values)

    def verify_grouped(filename, grouped, identifier):
        output = rows(filename)
        assert len(output) == len(grouped) and set(grouped) == {
            identifier(r) for r in output
        }
        for row in output:
            observed = grouped[identifier(row)]
            for m in metrics:
                near(fmean(v[m] for v in observed), float(row[m]))

    verify_grouped(
        "conditions.csv",
        per_condition,
        lambda r: (r["mode"], float(r["noise"]), float(r["severity"])),
    )
    verify_grouped("families.csv", per_family, lambda r: r["scenario"])
    panel_banks = {
        k: {
            bank: {m: fmean(v[m] for v in group) for m in metrics}
            for bank, group in d.items()
        }
        for k, d in per_panel.items()
    }
    output = rows("panels.csv")
    assert len(output) == len(panel_banks) and set(panel_banks) == {
        r["panel"] for r in output
    }
    for row in output:
        d = panel_banks[row["panel"]]
        assert len({name for name, rep in d}) == int(row["scenarios"])
        for m in metrics:
            near(
                macro_bank_means({bank: v[m] for bank, v in d.items()})[0],
                float(row[m]),
            )
    contrasts = rows("contrasts.csv")
    assert len(contrasts) == len(panel_banks) * len(refs) and {
        (r["panel"], r["reference"]) for r in contrasts
    } == {(k, ref) for k in panel_banks for ref in refs}
    for row in contrasts:
        d = panel_banks[row["panel"]]
        ref = row["reference"]
        triple = macro_bank_means({bank: v["robust"] - v[ref] for bank, v in d.items()})
        near(triple, [float(row[k]) for k in ("difference", "low", "high")])
    return len(contrasts)


def audit_reference_selection(loss):
    public_names = {
        "simple_average": "mean",
        "top_n": "top_n",
        "stacking_lr": "prob_lr",
        "stacking_logit": "logit",
    }
    validation = defaultdict(dict)
    fits = rows("reference-validation.csv.gz")
    selections = rows("reference-selection.csv.gz")
    assert len(fits) == 1824 and len(selections) == 3840
    for row in fits:
        key = (row["scenario"], int(row["replicate"]), float(row["dev_noise"]))
        validation[key][row["method"]] = float(row["validation_brier"])
    assert len(validation) == 456
    seen = set()
    for row in selections:
        bank = (row["scenario"], int(row["replicate"]), float(row["dev_noise"]))
        condition = (*bank, row["mode"], float(row["severity"]))
        assert condition not in seen
        seen.add(condition)
        risks = validation[bank]
        picked = min(public_names, key=risks.__getitem__)
        actual = {m: loss[(*condition, alias)] for m, alias in public_names.items()}
        best = min(actual.values())
        assert row["validation_best_method"] == picked and int(row["n_methods"]) == 4
        near(float(row["validation_best_brier"]), risks[picked])
        near(float(row["test_best_brier"]), best)
        near(float(row["selected_test_brier"]), actual[picked])
        near(float(row["test_excess"]), actual[picked] - best)
        assert (row["test_hit"] == "True") == (actual[picked] - best <= 1e-12)


def main():
    raw = rows("results.csv.gz")
    diagnostics = rows("diagnostics.csv.gz")
    scores = {cell(r) + (r["method"],): r for r in raw}
    diag = {cell(r): r for r in diagnostics}
    assert len(scores) == len(raw) and len(diag) == len(diagnostics)
    choices, transfers = check_raw(scores, diag)
    costs, total_calls = audit_costs(choices)
    nfits, checks, replays = replay_samples(scores, diag, costs)
    loss = {k: float(r["brier"]) for k, r in scores.items()}
    disturbed, clean = primary_numeric(loss)
    direct = transfer_summaries(transfers)
    verified = audit_summaries(loss, transfers)
    audit_reference_selection(loss)
    print(
        f"{verified} paired summary contrasts, all condition and family tables reproduced."
    )
    print(
        f"Paired robust minus deployed Brier, six disturbances: {disturbed}; clean at q=0: {clean}."
    )
    print(
        f"{len(choices)} fits, {total_calls} reported native score calls, {len(diag)} paired conditions, {len(scores)} method rows."
    )
    print(
        f"{len(direct)} diagnostic summaries, {nfits} archived fits, {checks} validation view and {replays} independent native test scores."
    )


if __name__ == "__main__":
    main()
