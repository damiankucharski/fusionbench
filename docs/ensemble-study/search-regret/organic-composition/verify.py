"""Independently replay predictions, costs and contrasts on the compositional panel."""

import argparse
import hashlib
import json
import runpy
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t

REFERENCE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "matched-graft" / "verify.py")
)
ARMS = ("stock", "type_only", "provider_only", "both")
LAWS = ("composition", "logit")
CPS = ("calls_600", "calls_1800", "generation_100")
DOMAINS = {
    "fit": (512, 0.5),
    "validation": (256, 0.5),
    "iid": (8192, 0.5),
    "interior": (8192, 0),
}
FIXED = ("average", "top_n", "stack_logit", "achievable_composition", "known_law")
COST_INDEX = ["data_seed", "fit_law", "gp_seed", "arm"]


def targets(ticks, roles):
    a, b, c, d = ticks[:, roles].T
    # Compute each two-way minimum and the final maximum by integer arithmetic.
    ab = (a + b - abs(a - b)) // 2
    cd = (c + d - abs(c - d)) // 2
    return {
        "composition": ((ab + cd + abs(ab - cd)) // 2) / 64,
        "logit": expit(0.2 + 0.7 * logit(a / 64) + 0.7 * logit(b / 64)),
    }


def draw(seed):
    roles = tuple(np.random.default_rng([seed, 229]).permutation(6)[:4])
    result = {}
    for i, (name, (n, p)) in enumerate(DOMAINS.items()):
        rng = np.random.default_rng([seed, i])
        mask = rng.random(n) < p
        bits = rng.integers(0, 2, (n, 6))
        ticks = rng.integers(1, 64, (n, 6))
        ticks[mask] = 1 + 62 * bits[mask]
        u = rng.random(n)
        q = targets(ticks, roles)
        result[name] = ticks / 64, q, {k: (u < v).astype(int) for k, v in q.items()}
    return result, roles


def certificates():
    bits = np.asarray(list(product((0, 1), repeat=6)), dtype=int)
    y = (bits[:, 0] & bits[:, 1]) | (bits[:, 2] & bits[:, 3])
    lo = hi = proj = 64
    for k in range(1, 7):
        for subset in combinations(range(6), k):
            vals = bits[:, subset]
            lo = min(lo, int(np.sum(vals.min(1) != y)))
            hi = min(hi, int(np.sum(vals.max(1) != y)))
            if k == 1:
                proj = min(proj, int(np.sum(vals[:, 0] != y)))
    assert (lo, hi, proj) == (12, 20, 20)
    # The full threshold family restricts to MIN, MAX or its first operand on
    # two endpoints. Any native convex weighted mean is a point of the
    # six-simplex. Verify the optimality conditions for equal weights on the
    # four signals using the exact integer gradient proportional to X'(Xw-y).
    r = bits[:, :4].sum(1) - 4 * y
    gradient = bits.T @ r
    assert (gradient[:4] == gradient[0]).all() and (gradient[4:] >= gradient[0]).all()
    mse = Fraction(int(r @ r), 64 * 16)
    corner_bound = min(Fraction(min(lo, hi, proj), 64), mse) * Fraction(62, 64) ** 2
    assert corner_bound == Fraction(961, 8192)
    # Two disjoint pairs of negative states have the same coordinate sums
    # as the positive pair. The smaller positive logit is <= the common pair
    # mean; the larger negative logit of EACH pair is >= that same mean.
    # Monotonicity transfers both inequalities to the predicted probabilities.
    h, l = Fraction(63, 64), Fraction(1, 64)
    for e, f in product((0, 1), repeat=2):
        positive = np.array([[1, 1, 0, 0, e, f], [0, 0, 1, 1, e, f]])
        negatives = [
            np.array([[1, 0, 1, 0, e, f], [0, 1, 0, 1, e, f]]),
            np.array([[1, 0, 0, 1, e, f], [0, 1, 1, 0, e, f]]),
        ]
        assert len(set(map(tuple, np.vstack([positive, *negatives])))) == 6
        assert (
            targets(1 + 62 * positive, (0, 1, 2, 3))["composition"] == float(h)
        ).all()
        for pair in negatives:
            np.testing.assert_array_equal(positive.sum(0), pair.sum(0))
            assert (
                targets(1 + 62 * pair, (0, 1, 2, 3))["composition"] == float(l)
            ).all()
    # A fixed positive prediction u <= both negative predictions imposes
    # squared error >= (h-u)^2 + 2 max(0,u-l)^2. If u<l this already costs
    # >= (h-l)^2. On [l,1], verify equality of all coefficients of this
    # quadratic with 3(u-m)^2 + (2/3)(h-l)^2, using exact rational numbers.
    m = (h + 2 * l) / 3
    minimum = Fraction(2, 3) * (h - l) ** 2
    assert l <= m <= 1 and (h - l) ** 2 >= minimum
    assert (3, -2 * (h + 2 * l), h * h + 2 * l * l) == (3, -6 * m, 3 * m * m + minimum)
    linear_mixed = minimum / 16 / 2
    assert linear_mixed == Fraction(961, 49152)
    assert linear_mixed > Fraction(961, 131072)
    return (1 + 62 * bits) / 64, corner_bound, linear_mixed


def verify_costs(events, costs, gp):
    assert (
        len(costs) == 12 * 2 * 2 * 4 * 3
        and not costs.duplicated([*COST_INDEX, "checkpoint"]).any()
    )
    assert not events.duplicated([*COST_INDEX, "generation"]).any()
    e_groups = {k: v for k, v in events.groupby(COST_INDEX)}
    u_groups = {
        k: v
        for k, v in gp.groupby(
            ["data_seed", "fit_law", "gp_seed", "method", "checkpoint"]
        )
    }
    for key, rows in costs.groupby(COST_INDEX):
        seed, law, start, arm = key
        assert set(rows.checkpoint) == set(CPS)
        events_run = e_groups[key].sort_values("generation")
        assert np.array_equal(
            events_run.generation.values, np.arange(1, len(events_run) + 1)
        )
        opportunity = events_run[events_run.offspring_index.notna()]
        assert (events_run.eligible_children > 0).equals(
            events_run.offspring_index.notna()
        )
        assert (opportunity.offspring_index < opportunity.offspring_count).all()
        assert (
            (opportunity.before_type != opportunity.after_type)
            == (arm in ("both", "type_only"))
        ).all()
        assert (
            (opportunity.before_provider != opportunity.after_provider)
            == (arm in ("both", "provider_only"))
        ).all()
        assert (opportunity.proposed_type != opportunity.before_type).all()
        assert (opportunity.proposed_provider != opportunity.before_provider).all()
        use_type = arm in ("both", "type_only")
        use_provider = arm in ("both", "provider_only")
        assert np.array_equal(
            opportunity.after_type,
            opportunity.proposed_type if use_type else opportunity.before_type,
        )
        assert np.array_equal(
            opportunity.after_provider,
            opportunity.proposed_provider
            if use_provider
            else opportunity.before_provider,
        )
        assert np.array_equal(
            opportunity.changed, opportunity.old_sha256 != opportunity.new_sha256
        )
        assert (opportunity.changed == (arm != "stock")).all()
        assert (
            (opportunity.origin == "crossover")
            == (opportunity.offspring_index < opportunity.crossover_children)
        ).all()
        seed0 = np.random.SeedSequence([seed, start, 419]).generate_state(1)[0]
        expected_seed = [
            int(np.random.SeedSequence([seed0, int(j)]).generate_state(1)[0])
            for j in events_run.generation
        ]
        np.testing.assert_array_equal(events_run.event_seed, expected_seed)
        for cp in rows.itertuples():
            selected = u_groups[seed, law, start, arm, cp.checkpoint]
            assert all(selected.calls == cp.calls) and all(
                selected.generation == cp.generation
            )
            assert selected.path_digest.nunique() == 1 and all(
                selected.path_digest.str.len() == 64
            )
            assert cp.fit_calls == cp.calls
            assert (
                all(selected.archive_trees == cp.validation_calls)
                and cp.validation_calls <= cp.fit_calls
            )
            if cp.checkpoint.startswith("calls_"):
                assert cp.calls == int(cp.checkpoint.split("_")[1])
            else:
                assert cp.generations_completed == cp.generation == 100
            assert cp.generation in (
                cp.generations_completed,
                cp.generations_completed + 1,
            )
            before = opportunity[opportunity.generation <= cp.generation]
            finished = before[before.generation <= cp.generations_completed]
            assert (
                len(before) == cp.opportunities
                and cp.generation - len(before) == cp.no_eligible
            )
            assert cp.changed == before.changed.sum()
            assert cp.type_changed == sum(before.before_type != before.after_type)
            assert cp.provider_changed == sum(
                before.before_provider != before.after_provider
            )
            assert cp.passed_dedup == sum(before.passed_dedup == True)
            assert cp.filtered_duplicate == sum(before.passed_dedup == False)
            assert cp.evaluated == sum(before.evaluation_call <= cp.calls)
            assert cp.survived == sum(finished.survived == True)
            assert cp.equivalent_survived == sum(finished.equivalent_survived == True)
            assert cp.selection_pending == len(before) - len(finished)


def statistical_replay(runs, path, bound):
    keys = ["fit_law", "split", "test_law", "decision", "checkpoint"]
    gp = runs[runs.gp_seed.notna()]
    table = pd.read_csv(path / "paired.csv", float_precision="round_trip").set_index(
        [*keys, "metric", "contrast"]
    )
    assert table.index.is_unique
    checked = set()
    for combination, rows in gp.groupby(keys):
        for metric in ("conditional_excess", "observed_brier"):
            # Average the two independent GP starts within bank and arm first.
            bank = rows.groupby(["data_seed", "method"])[metric].mean().unstack()
            a, b, c, d = (bank[k] for k in ARMS)
            comparisons = {
                "type_only_minus_stock": b - a,
                "provider_only_minus_stock": c - a,
                "both_minus_stock": d - a,
                "type_minus_provider": b - c,
                "interaction": (d - b) - (c - a),
            }
            if (
                combination[:3] == ("composition", "iid", "composition")
                and metric == "conditional_excess"
            ):
                comparisons.update(
                    {f"{k}_minus_single_update_bound": bank[k] - bound for k in ARMS}
                )
            for name, delta in comparisons.items():
                idx = (*combination, metric, name)
                actual = table.loc[idx]
                checked.add(idx)
                assert actual.banks == len(delta) == 12 and actual.starts == 24
                mean = delta.mean()
                half = t.ppf(0.975, 11) * delta.std() / np.sqrt(12)
                np.testing.assert_allclose(
                    [actual.difference, actual.low_95, actual.high_95],
                    [mean, mean - half, mean + half],
                    rtol=0,
                    atol=1e-14,
                )
    assert checked == set(table.index) and len(checked) == 744
    report = pd.read_csv(path / "means.csv", float_precision="round_trip").set_index(
        [*keys, "method"]
    )
    assert report.index.is_unique
    fields = ["observed_brier", "conditional_excess", "nodes", "calls"]
    expected = (
        runs.groupby([*keys, "method", "data_seed"])[fields]
        .mean()
        .groupby([*keys, "method"])
        .mean()
    )
    assert set(expected.index) == set(report.index) and len(expected) == 348
    np.testing.assert_allclose(
        report.loc[expected.index, fields],
        expected,
        rtol=1e-14,
        atol=2e-15,
        equal_nan=True,
    )
    return len(table), len(report)


def verify(path):
    runs = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    events = pd.read_csv(path / "events.csv.gz", float_precision="round_trip")
    costs = pd.read_csv(path / "costs.csv.gz", float_precision="round_trip")
    cube, single, linear = certificates()
    assert set(runs.data_seed) == set(range(4200, 4212)) and set(runs.fit_law) == set(
        LAWS
    )
    assert len(runs) == 12 * 2 * (5 + 2 * 4 * 3 * 2) * 6
    assert not runs.duplicated(
        [
            "data_seed",
            "fit_law",
            "gp_seed",
            "method",
            "decision",
            "checkpoint",
            "split",
            "test_law",
        ]
    ).any()
    gp = runs[runs.gp_seed.notna()]
    assert (
        set(gp.method) == set(ARMS)
        and set(gp.gp_seed) == {331, 337}
        and set(gp.checkpoint) == set(CPS)
    )
    assert set(gp.decision) == {"fit", "validation"}
    assert set(runs[runs.gp_seed.isna()].method) == set(FIXED)
    max_error = 0.0
    programs = witnesses = single_checks = baseline_rows = 0
    for seed, data in runs.groupby("data_seed"):
        draws, roles = draw(seed)
        corner_q = targets((cube * 64).astype(int), roles)["composition"]
        for law, rows in data.groupby("fit_law"):
            matrices = [draws[s][0] for s in DOMAINS]
            baselines, (n, rank, prefixes) = REFERENCE["make_baselines"](
                matrices[0], draws["fit"][2][law], [*matrices, cube]
            )
            assert (
                np.mean((baselines["stack_logit"][-1] - corner_q) ** 2)
                >= float(2 * linear) - 1e-14
            )
            cache = {}
            checked_corner = set()
            for row in rows.itertuples():
                split = row.split
                x, qs, ys = draws[split]
                assert row.test_law in (
                    LAWS if split in ("iid", "interior") else (law,)
                )
                if row.method in baselines:
                    pred = baselines[row.method][list(DOMAINS).index(split)]
                    baseline_rows += 1
                    if row.method == "top_n":
                        assert row.n_selected == n and np.array_equal(
                            rank, json.loads(row.ranking)
                        )
                        np.testing.assert_allclose(
                            prefixes, json.loads(row.prefix_scores), rtol=0, atol=3e-15
                        )
                elif row.method == "known_law":
                    pred = qs[law]
                else:
                    key = row.program, split
                    if key not in cache:
                        program = json.loads(row.program)
                        assert REFERENCE["count_nodes"](program) == row.nodes
                        cache[key] = REFERENCE["evaluate"](program, x)
                        if row.program not in checked_corner:
                            corner_pred = REFERENCE["evaluate"](program, cube)
                            if row.method == "achievable_composition":
                                np.testing.assert_array_equal(corner_pred, corner_q)
                            elif row.operators <= 1:
                                assert (
                                    np.mean((corner_pred - corner_q) ** 2)
                                    >= float(single) - 1e-14
                                )
                                single_checks += 1
                            checked_corner.add(row.program)
                    pred = cache[key]
                    programs += 1
                    if row.method == "achievable_composition":
                        assert row.nodes == 7
                        np.testing.assert_array_equal(pred, qs["composition"])
                        witnesses += 1
                digest = hashlib.sha256(
                    np.asarray(pred, dtype="<f8").tobytes()
                ).hexdigest()
                assert digest == row.prediction_sha256, (
                    seed,
                    law,
                    row.method,
                    split,
                    "vector",
                )
                err = max(
                    abs(row.observed_brier - np.mean((pred - ys[row.test_law]) ** 2)),
                    abs(
                        row.conditional_excess - np.mean((pred - qs[row.test_law]) ** 2)
                    ),
                )
                assert err < 1e-13, (seed, law, row.method, split, err)
                max_error = max(err, max_error)
    verify_costs(events, costs, gp)
    contrasts, means = statistical_replay(runs, path, float(single / 2))
    verification = {
        "predicted_vectors": len(runs),
        "losses": 2 * len(runs),
        "native_program_split_rows": programs,
        "three_baselines_split_rows": baseline_rows,
        "witness_vectors_equal_to_composition": witnesses,
        "data_banks": 12,
        "runs": len(costs) // 3,
        "cost_checkpoints": len(costs),
        "generational_events": len(events),
        "checked_single_update_winners_on_cube": single_checks,
        "one_update_corner_excess": str(single),
        "one_update_mixed_lower_bound": str(single / 2),
        "affine_logit_mixed_lower_bound": str(linear),
        "contrasts": contrasts,
        "method_means": means,
        "max_abs_loss_error": max_error,
        "source_sha256": {
            file: hashlib.sha256((path / file).read_bytes()).hexdigest()
            for file in (
                "runs.csv.gz",
                "costs.csv.gz",
                "events.csv.gz",
                "paired.csv",
                "means.csv",
            )
        },
    }
    (path / "verified.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
