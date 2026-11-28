"""Verify scene sampling, saved results and paired contrasts from the frozen studies."""

import hashlib
import json
from itertools import combinations, product
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fusionbench.scenarios.checkpoint_bank import (
    generate_checkpoint_bank,
    generate_scene_pool,
)
from fusionbench.transfer import load_preset

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BANKS = range(286100, 286132)
LAWS = ("equal", "signal")
ARMS = ((24, 8), (192, 1), (192, 8), (1536, 1))
METHODS = ("mean", "top_n", "prob_lr", "regularized", "seeded")
KEYS = ["seed", "law", "n_scenes", "rows_per_scene", "method"]
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, identity):
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=TOL, err_msg=str(identity)
    )


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def indexed(filename, keys, expected):
    frame = pd.read_csv(OUT / filename, float_precision="round_trip").set_index(keys)
    require(
        frame.index.is_unique and set(frame.index) == set(expected),
        filename + ": missing, extra or duplicated cell",
    )
    return frame.sort_index()


def pooled_f1(p, y):
    positive = p > 0.5
    denominator = (y == 1).sum() + positive.sum()
    return 2 * ((y == 1) & positive).sum() / denominator if denominator else 0.0


def binary_metrics(y, p):
    return {"brier": float(2 * np.square(y - p).mean()), "f1": float(pooled_f1(p, y))}


def oracle_excess(q, p):
    # Integrate both possible outcomes for the frozen rule, then subtract the
    # Bayes risk. This is independent of the runner's squared-distance formula.
    risk = 2 * (q * (1 - p) ** 2 + (1 - q) * p**2).mean()
    value = risk - (2 * q * (1 - q)).mean()
    same(value, 2 * np.square(p - q).mean(), "conditional Brier decomposition")
    return float(value)


def audit_inputs(seed, pools, chosen_signal):
    expected_shapes = {"A": (1, 24), "F": (4, 1536), "I": (3, 512)}
    group_ids = []
    for domain, (channel, count) in expected_shapes.items():
        equal, signal = [p[domain] for p in pools]
        group = np.arange(count)
        if domain == "F":
            group = 24 + 2 * group
        if domain == "I":
            group = 25 + 2 * group
        group_ids.append(set(group))
        np.testing.assert_array_equal(equal.scene, np.repeat(group, 8))
        scene_latent = np.random.default_rng([seed, channel, 0]).standard_normal(
            (count, 1)
        )
        row_latent = np.random.default_rng([seed, channel, 1]).standard_normal(
            (count, 8)
        )
        q = 1 / (1 + np.exp(-1.8 * (0.9 * scene_latent + 0.8 * row_latent - 1.1)))
        y = np.random.default_rng([seed, channel, 2]).random(q.shape) < q
        for block in (equal, signal):
            require(block.P.shape == (count * 8, 8, 8), f"{domain} tensor dimensions")
            np.testing.assert_array_equal(q.ravel(), block.posterior)
            np.testing.assert_array_equal(y.ravel(), block.y)
            np.testing.assert_array_equal(equal.scene, block.scene)
        others = np.arange(8) != chosen_signal
        np.testing.assert_array_equal(equal.P[:, others], signal.P[:, others])
        np.testing.assert_array_equal(
            signal.P[:, chosen_signal], np.repeat(q.reshape(-1, 1), 8, axis=1)
        )
        require(
            not np.array_equal(equal.P[:, chosen_signal], signal.P[:, chosen_signal]),
            "no planted signal",
        )
    require(
        all(not (a & b) for a, b in combinations(group_ids, 2)), "A, F and I overlap"
    )


def selections(block):
    return [
        int(np.argmax([pooled_f1(block.P[:, j, k], block.y) for k in range(8)]))
        for j in range(8)
    ]


def compose(block, slots):
    p = np.column_stack([block.P[:, j, slot] for j, slot in enumerate(slots)])
    return np.stack((1 - p, p), axis=2)


def fit_rows(s, r):
    # Decode the two coordinates from the maximal pool row IDs. This is
    # independent of the generator's prefix-index constructor.
    absolute = np.arange(1536 * 8, dtype=np.int64)
    return absolute[(absolute // 8 < s) & (absolute % 8 < r)]


def compare_controls(scores, key, x, y, ax, ay, q):
    fit, audit = x[:, :, 1], ax[:, :, 1]
    rank = sorted(range(8), key=lambda j: np.square(fit[:, j] - y).mean())
    count = min(
        range(1, 9), key=lambda k: np.square(fit[:, rank[:k]].mean(axis=1) - y).mean()
    )
    comparisons = []
    for method, indices in (("mean", list(range(8))), ("top_n", rank[:count])):
        p = fit[:, indices].mean(axis=1)
        ap = audit[:, indices].mean(axis=1)
        actual = scores.loc[(*key, method)]
        fields = {f"fit_{k}": v for k, v in binary_metrics(y, p).items()}
        fields.update({f"audit_{k}": v for k, v in binary_metrics(ay, ap).items()})
        fields["audit_excess"] = oracle_excess(q, ap)
        fields["n_providers"] = len(indices)
        if method == "top_n":
            fields["n_selected"] = count
        for field, expect in fields.items():
            same(actual[field], expect, (key, method, field))
            comparisons.append(abs(float(actual[field]) - expect))
    return max(comparisons)


def native_intervention(x, y, ax, ay, q, sid, seed, archived, score):
    fit = run_path(str(HERE.parent / "seeded-checkpoint/native_seed.py"))["run_seeded"]
    options = {"seed": seed + 1000000, "arm": "seeded", "signal_provider": sid}
    original, before = fit(x, y, ax, ay, audit_q=q, **options)
    replaced, after = fit(x, y, ax, 1 - ay, audit_q=np.full_like(q, 0.5), **options)
    for key in ("trajectory_sha256", "selected_program_sha256"):
        require(
            original[key] == replaced[key] == archived[key],
            f"{key} changed with audit treatment",
        )
    for split in ("fit_selected", "audit_selected"):
        np.testing.assert_array_equal(before[split], after[split])
    audit = before["audit_selected"]
    same(
        score.fit_brier,
        binary_metrics(y, before["fit_selected"])["brier"],
        "fit native Brier",
    )
    same(score.audit_excess, oracle_excess(q, audit), "native conditional Brier")
    same(score.audit_brier, binary_metrics(ay, audit)["brier"], "audit native Brier")
    same(score.audit_f1, pooled_f1(audit, ay), "audit native F1")
    same(
        replaced["audit_selected"],
        binary_metrics(1 - ay, audit)["brier"],
        "inverted native Brier",
    )
    same(
        replaced["audit_selected_excess"],
        oracle_excess(np.full_like(q, 0.5), audit),
        "replacement oracle excess",
    )
    require(
        abs(original["audit_selected"] - replaced["audit_selected"]) > TOL,
        "label replacement did not change audit loss",
    )
    require(
        abs(original["audit_selected_excess"] - replaced["audit_selected_excess"])
        > TOL,
        "oracle replacement did not change excess",
    )
    return {
        "seed": seed,
        "law": "signal",
        "n_scenes": 24,
        "rows_per_scene": 8,
        "trajectory_sha256": original["trajectory_sha256"],
        "selected_program_sha256": original["selected_program_sha256"],
        "fit_predictions_sha256": sha(before["fit_selected"]),
        "audit_predictions_sha256": sha(audit),
        "original_observed_brier": original["audit_selected"],
        "label_inverted_brier": replaced["audit_selected"],
        "original_excess": original["audit_selected_excess"],
        "constant_oracle_excess": replaced["audit_selected_excess"],
    }


def main_grid():
    cells = [
        (seed, law, s, r, method)
        for seed, law, (s, r), method in product(BANKS, LAWS, ARMS, METHODS)
    ]
    scores = indexed("scores.csv", KEYS, cells)
    stages = indexed(
        "trajectories.csv.gz",
        [*KEYS, "generation"],
        [(*c, g) for c in cells if c[-1] == "seeded" for g in range(17)],
    )
    np.testing.assert_array_equal(
        scores.n_rows, [s * r for _, _, s, r, _ in scores.index]
    )
    raw = stages.xs(16, level="generation")
    for a, b in [
        ("audit_selected", "audit_brier"),
        ("audit_f1", "audit_f1"),
        ("audit_selected_excess", "audit_excess"),
        ("fit_selected", "fit_brier"),
    ]:
        same(scores.loc[raw.index, b], raw[a], f"terminal {a}")
    infos = json.loads((OUT / "run.json").read_text())
    require(
        infos["protocol_sha256"]
        == hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest(),
        "protocol SHA mismatch",
    )
    require(
        sorted(r["seed"] for r in infos["banks"]) == list(BANKS), "manifest coverage"
    )
    return scores, len(stages), {r["seed"]: r for r in infos["banks"]}


def fixed_variance(law, seed, x, y, ax, q, provider, rows, components, decisions):
    # Var(cluster mean) versus within-cluster variance for this fixed pair of
    # predictions. Compute deviations by explicit sums of squares at both levels.
    margin = 2 * (
        np.square(x[:, provider, 1] - y) - np.square(x[:, :, 1].mean(axis=1) - y)
    )
    groups = margin.reshape(1536, 8)
    centers = groups.mean(axis=1)
    within = np.sum((groups - centers[:, None]) ** 2) / (1536 * 7)
    between = np.sum((centers - centers.mean()) ** 2) / 1535
    audit = oracle_excess(q, ax[:, provider, 1]) - oracle_excess(
        q, ax[:, :, 1].mean(axis=1)
    )
    same(
        [within, between, audit],
        components.loc[
            (seed, law), ["within", "between_means", "audit_conditional_difference"]
        ],
        (seed, law, "fixed variance"),
    )
    for (n, r), indices in rows.items():
        same(
            [margin[indices].mean(), audit],
            decisions.loc[
                (seed, law, n, r), ["fit_difference", "audit_conditional_difference"]
            ],
            (seed, law, n, r, "fixed margin"),
        )
    return within, between


def verify_banks(scores, infos):
    error = 0.0
    replay = None
    overlaps = {}
    components = indexed(
        "fixed-variance-components.csv", ["seed", "law"], product(BANKS, LAWS)
    )
    decisions = indexed(
        "fixed-decisions.csv",
        ["seed", "law", "n_scenes", "rows_per_scene"],
        [(seed, law, n, r) for seed, law, (n, r) in product(BANKS, LAWS, ARMS)],
    )
    moments = {law: [] for law in LAWS}
    rows = {arm: fit_rows(*arm) for arm in ARMS}
    for (a, idx), (b, other) in combinations(rows.items(), 2):
        size = np.intersect1d(idx, other).size
        require(size == min(a[0], b[0]) * min(a[1], b[1]), f"fit row overlap {a}, {b}")
        overlaps[f"{a[0]}x{a[1]} + {b[0]}x{b[1]}"] = int(size)
    for seed in BANKS:
        pools = [generate_scene_pool(seed, law=law) for law in LAWS]
        sid = int(np.random.default_rng([seed, 0, 1]).integers(8))
        audit_inputs(seed, pools, sid)
        for law, bank in zip(LAWS, pools, strict=True):
            manifest = infos[seed]["laws"][law]
            require(infos[seed]["search_seed"] == seed + 1000000, "search seed")
            slot = selections(bank["A"])
            require(manifest["slots"] == slot, f"{seed}/{law} checkpoint selection")
            require(
                manifest["signal_provider"] == (sid if law == "signal" else None),
                f"{seed}/{law} provider role",
            )
            if law == "signal":
                require(slot[sid] == 0, "identical slots should tie at zero")
            for dom, block in bank.items():
                for name, expected in manifest["blocks"][dom].items():
                    require(
                        sha(getattr(block, name)) == expected,
                        f"{seed}/{law}/{dom}/{name}",
                    )
            F, I = bank["F"], bank["I"]
            x, ax = compose(F, slot), compose(I, slot)
            moments[law].append(
                fixed_variance(
                    law, seed, x, F.y, ax, I.posterior, sid, rows, components, decisions
                )
            )
            require(sha(ax) == manifest["audit_x_sha256"], "audit tensor")
            archived = {
                (a["n_scenes"], a["rows_per_scene"]): a for a in manifest["allocations"]
            }
            require(
                len(manifest["allocations"]) == 4 and set(archived) == set(ARMS),
                "manifest arms",
            )
            for (s, r), indices in rows.items():
                cell = seed, law, s, r
                meta = archived[s, r]
                xfit, yfit = x[indices], F.y[indices]
                observed, counts = np.unique(F.scene[indices], return_counts=True)
                require(
                    len(observed) == s
                    and np.all(counts == r)
                    and len(yfit) == s * r == meta["n_rows"],
                    f"sample counts {cell}",
                )
                for name, data in [
                    ("indices", indices),
                    ("fit_x", xfit),
                    ("fit_y", yfit),
                    ("scene_ids", F.scene[indices]),
                ]:
                    require(sha(data) == meta[name + "_sha256"], f"{cell}: {name}")
                error = max(
                    error,
                    compare_controls(scores, cell, xfit, yfit, ax, I.y, I.posterior),
                )
                if (seed, law, s, r) == (BANKS.start, "signal", 24, 8):
                    replay = native_intervention(
                        xfit,
                        yfit,
                        ax,
                        I.y,
                        I.posterior,
                        sid,
                        seed,
                        meta,
                        scores.loc[(*cell, "seeded")],
                    )
            if seed == BANKS.start:
                # Rebuilding shorter F/I blocks retains all overlapping rows,
                # including posterior, independent of the requested size.
                short = generate_scene_pool(
                    seed, law=law, n_fit_scenes=192, n_audit_scenes=23
                )
                old = generate_checkpoint_bank(seed, law=law)
                for dom, fields in [
                    (d, ("scene", "y", "posterior", "P")) for d in bank
                ]:
                    for field in fields:
                        prefix = getattr(short[dom], field)
                        np.testing.assert_array_equal(
                            prefix, getattr(bank[dom], field)[: len(prefix)]
                        )
                for dom in ("A", "I"):
                    for field in ("y", "posterior", "P"):
                        prefix = getattr(old[dom], field)
                        np.testing.assert_array_equal(
                            prefix, getattr(bank[dom], field)[: len(prefix)]
                        )
    summary = pd.read_csv(OUT / "fixed-variance-summary.csv").set_index("law")
    variance_ratios = {}
    draws = np.random.default_rng(97823).integers(len(BANKS), size=(10_000, len(BANKS)))
    for law, pairs in moments.items():
        w, b = np.array(pairs).T

        def ratio(within, between):
            return 8 * between / (between + 7 * within / 8)

        estimate = ratio(w.mean(), b.mean())
        bootstrap = ratio(w[draws].mean(axis=1), b[draws].mean(axis=1))
        numbers = [estimate, *np.quantile(bootstrap, [0.025, 0.975])]
        same(summary.loc[law, ["variance_ratio", "ci_low", "ci_high"]], numbers, law)
        variance_ratios[law] = dict(
            zip(("variance_ratio", "ci_low", "ci_high"), numbers, strict=True)
        )
    return {
        "fixed_pair_variance_ratios": variance_ratios,
        "regenerated_banks": len(BANKS),
        "coupled_laws": list(LAWS),
        "verified_mean_top_n_scores": len(BANKS) * len(LAWS) * len(ARMS) * 2,
        "max_abs_control_error": error,
        "verified_fit_row_intersections": overlaps,
        "audit_replacement": replay,
    }


def interval(values, draws):
    x = np.asarray(values)
    return [
        float(x.mean()),
        *map(float, np.quantile(x[draws].mean(axis=1), [0.025, 0.975])),
    ]


def compare_interval(row, expected, identity):
    same(
        row[["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float), expected, identity
    )


def verify_contrasts(scores):
    table = pd.read_csv(OUT / "design-contrasts.csv").set_index(
        ["law", "method", "metric", "contrast"]
    )
    pairs = {
        "clustered_minus_independent_192": (ARMS[0], ARMS[1]),
        "clustered_minus_independent_1536": (ARMS[2], ARMS[3]),
        "large_minus_small_clustered": (ARMS[2], ARMS[0]),
        "large_minus_small_independent": (ARMS[3], ARMS[1]),
        "extra_rows_same_192_scenes": (ARMS[2], ARMS[1]),
    }
    draws = np.random.default_rng(97823).integers(len(BANKS), size=(10_000, len(BANKS)))
    primary = None
    for law, method, (name, (a, b)) in product(LAWS, METHODS, pairs.items()):
        left = [scores.loc[(seed, law, *a, method), "audit_excess"] for seed in BANKS]
        right = [scores.loc[(seed, law, *b, method), "audit_excess"] for seed in BANKS]
        delta = np.subtract(left, right)
        estimate = interval(delta, draws)
        compare_interval(
            table.loc[(law, method, "audit_excess", name)],
            estimate,
            (law, method, name),
        )
        if (
            law == "signal"
            and method == "seeded"
            and name == "clustered_minus_independent_192"
        ):
            raw = pd.read_csv(OUT / "primary-by-bank.csv").sort_values("seed")
            np.testing.assert_array_equal(raw.seed, list(BANKS))
            same(
                raw[["clustered_excess", "independent_excess", "difference"]],
                np.column_stack((left, right, delta)),
                "per-bank primary",
            )
            primary = dict(
                zip(("estimate", "ci_low", "ci_high"), estimate, strict=True)
            )
    return {
        "verified_excess_contrasts": len(pairs) * len(LAWS) * len(METHODS),
        "primary": primary,
    }


def brier(y, predictions):
    return np.square(predictions - np.eye(predictions.shape[1])[y]).sum(axis=1).mean()


def compare_bank(frame, fit_x, fit_y, test_x, test_y):
    require(
        (frame.n_fit == len(fit_y)).all() and (frame.n_test == len(test_y)).all(),
        "row counts",
    )
    require(
        (frame.pool_size == fit_x.shape[1]).all()
        and (frame.n_classes == fit_x.shape[2]).all(),
        "provider or class dimensions",
    )
    for split, x, y in (("fit", fit_x, fit_y), ("test", test_x, test_y)):
        require((frame[split + "_inputs_sha256"] == sha(x)).all(), f"{split} inputs")
        require((frame[split + "_labels_sha256"] == sha(y)).all(), f"{split} labels")
        same(
            brier(y, x.mean(axis=1)),
            frame.loc["mean", split + "_brier"],
            f"{split} mean Brier",
        )


def verify_suite():
    from fusionbench import generate_dataset

    suites = list(load_preset("standard-v4")["scenarios"])
    older = list(load_preset("standard-v3")["scenarios"])
    require(
        len(suites) == 25 and suites == [*older, "cloned_shortcut"],
        "changed historical standard-suite order",
    )
    seeds = (286201, 286202)
    idx = ["scenario_id", "data_seed"]
    scores = indexed(
        "panel-scores.csv", [*idx, "method"], product(suites, seeds, METHODS)
    )
    stages = indexed(
        "panel-trajectories.csv",
        [*idx, "generation"],
        product(suites, seeds, range(17)),
    )
    for scenario, seed in product(suites, seeds):
        ds = generate_dataset(scenario, n_per_split=512, seed=seed)
        compare_bank(
            scores.loc[(scenario, seed)],
            ds.fit_preds,
            ds.fit_y,
            ds.test_preds,
            ds.test_y,
        )
    comparisons = pd.read_csv(OUT / "panel-contrasts.csv").set_index(
        ["suite", "method", "reference"]
    )
    results = {}
    for suite, ids in (("standard-v4", suites), ("historical-24", older)):
        draws = np.random.default_rng(97823).integers(2, size=(10_000, len(ids), 2))
        for reference in set(METHODS) - {"seeded"}:
            delta = np.array(
                [
                    [
                        scores.loc[(s, seed, "seeded"), "test_brier"]
                        - scores.loc[(s, seed, reference), "test_brier"]
                        for seed in seeds
                    ]
                    for s in ids
                ]
            )
            values = delta[np.arange(len(ids))[None, :, None], draws]
            estimate = [
                delta.mean(),
                *np.quantile(values.mean(axis=(1, 2)), [0.025, 0.975]),
            ]
            compare_interval(
                comparisons.loc[(suite, "seeded", reference)],
                estimate,
                (suite, reference),
            )
            if reference == "mean":
                results[suite] = dict(
                    zip(("estimate", "ci_low", "ci_high"), estimate, strict=True)
                )
    return {
        "scores": len(scores),
        "stages": len(stages),
        "regenerated_scenario_banks": len(suites) * len(seeds),
        "seeded_minus_mean": results,
    }


def verify_crossed():
    from fusionbench import get_scenario

    seeds = range(286300, 286324)
    conditions = {
        (0, 0): "neither",
        (1, 0): "redundancy_only",
        (0, 1): "shift_only",
        (1, 1): "both",
    }
    scores = indexed(
        "factorial-scores.csv",
        ["data_seed", "condition", "method"],
        product(seeds, conditions.values(), METHODS),
    )
    stages = indexed(
        "factorial-trajectories.csv",
        ["data_seed", "clone_probability", "generation"],
        product(seeds, (0, 1), range(17)),
    )
    for seed in seeds:
        baseline = {}
        for (c, r), name in conditions.items():
            scenario = get_scenario(
                "cloned_shortcut", clone_probability=c, test_reversal_probability=r
            )
            rng = np.random.default_rng(seed)
            data = {}
            for split, n in (("fit", 1024), ("test", 4096)):
                x, y, meta = scenario.sample_split(n, rng, split)
                if name == "neither":
                    baseline[split] = x, y
                control, labels = baseline[split]
                np.testing.assert_array_equal(y, labels)
                np.testing.assert_array_equal(x[:, :2], control[:, :2])
                signs = 2 * (x[:, :, 1] > 0.5).astype(float) - 1
                expected = control[:, 2 : 3 if c else None, 1] > 0.5
                if split == "test" and r:
                    expected = ~expected
                np.testing.assert_array_equal(
                    x[:, 2:, 1] > 0.5, np.broadcast_to(expected, x[:, 2:, 1].shape)
                )
                shortcut_sign = signs[:, 2] if c else signs[:, 2:].sum(axis=1)
                reversal = r if split == "test" else 0
                log_odds = (
                    np.log(3) * signs[:, :2].sum(axis=1)
                    + (1 - 2 * reversal) * np.log(4) * shortcut_sign
                )
                same(
                    scenario.bayes_oracle(x, meta)[:, 1],
                    1 / (1 + np.exp(-log_odds)),
                    (seed, name, split, "observable oracle"),
                )
                data[split] = x, y
            compare_bank(scores.loc[(seed, name)], *data["fit"], *data["test"])
        for aligned, shifted in (
            ("neither", "shift_only"),
            ("redundancy_only", "both"),
        ):
            for field in ("fit_inputs_sha256", "fit_labels_sha256", "fit_brier"):
                np.testing.assert_array_equal(
                    scores.loc[(seed, aligned)][field],
                    scores.loc[(seed, shifted)][field],
                )
    effects = pd.read_csv(OUT / "factorial-effects.csv").set_index(["method", "effect"])
    raw = pd.read_csv(OUT / "factorial-bank-effects.csv").set_index(
        ["data_seed", "method", "effect"]
    )
    draws = np.random.default_rng(97823).integers(len(seeds), size=(10_000, len(seeds)))
    interaction = {}
    for method in METHODS:
        none, clone, flip, both = [
            np.array([scores.loc[(seed, n, method), "test_brier"] for seed in seeds])
            for n in ("neither", "redundancy_only", "shift_only", "both")
        ]
        for name, delta in (
            ("interaction", (both - clone) - (flip - none)),
            ("redundancy_main", ((both - flip) + (clone - none)) / 2),
            ("reversal_main", ((both - clone) + (flip - none)) / 2),
        ):
            same(
                [raw.loc[(seed, method, name), "value"] for seed in seeds],
                delta,
                (method, name),
            )
            estimate = interval(delta, draws)
            compare_interval(effects.loc[(method, name)], estimate, (method, name))
            if name == "interaction":
                interaction[method] = dict(
                    zip(("estimate", "ci_low", "ci_high"), estimate, strict=True)
                )
    return {
        "scores": len(scores),
        "unique_fit_trajectories": len(stages) // 17,
        "trajectory_rows": len(stages),
        "regenerated_conditions": len(seeds) * len(conditions),
        "interaction": interaction,
    }


def verify():
    scores, n_stages, info = main_grid()
    return {
        "score_rows": len(scores),
        "trajectory_rows": n_stages,
        "main": verify_banks(scores, info),
        "paired_32_bank_bootstrap": verify_contrasts(scores),
        "standard_v4": verify_suite(),
        "crossed_controls": verify_crossed(),
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        result = verify()
    target = OUT / "verification.json"
    target.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(f"Verified the main, standard-v4 and crossed panels; wrote {target}")
