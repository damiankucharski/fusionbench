"""Independently reconstruct the complete-support protocol and its archived event traces."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from fractions import Fraction
from itertools import product
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import rel_entr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SUPPORT = np.asarray(list(product((1 / 8, 7 / 8), repeat=3)))
FULL = np.stack([1 - SUPPORT, SUPPORT], axis=-1)
POLICIES = ("ordinary", "open_raw", "open_kl", "strict_kl")


def close(actual, expected, atol=4e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def problem(bank):
    seed = 4782000 + bank
    a, b, c = np.random.default_rng([seed, 17]).permutation(3)
    q = {
        "null": SUPPORT.mean(1),
        "signal": np.minimum(SUPPORT[:, c], np.maximum(SUPPORT[:, a], SUPPORT[:, b])),
    }

    def stream(i, n):
        rng = np.random.default_rng([seed, i])
        bits = rng.integers(2, size=(n, 3))
        cell = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
        u = rng.random(n)
        return cell, {law: (u < target[cell]).astype(int) for law, target in q.items()}

    return q, stream


def metrics(p, q, ids, y):
    brier_zero = 2 * p**2
    brier_one = 2 * (1 - p) ** 2
    return {
        "exact_excess": float(np.mean(2 * (p - q) ** 2)),
        "exact_brier": float(np.mean((1 - q) * brier_zero + q * brier_one)),
        "test_brier": float(np.mean(np.where(y, brier_one[ids], brier_zero[ids]))),
        "exact_accuracy": float(np.mean(np.where(p > 0.5, q, 1 - q))),
        "test_accuracy": float(np.mean((p[ids] > 0.5) == y)),
    }


def gate(inc, chall, y, slots, rule):
    # Calculate the losses for both admissible labels directly. Positive
    # and negative margins can differ in size even for the same pair.
    h0 = 2 * (inc - chall) * (inc + chall)
    h1 = 2 * (chall - inc) * (2 - inc - chall)
    lower, upper = np.minimum(h0, h1), np.maximum(h0, h1)
    obs = np.where(y, h1, h0)
    gain = obs.mean()
    if rule == "raw":
        return gain, gain
    width = upper - lower
    active = width > 0
    if not active.any():
        return gain, lower.mean()
    scale = width.max()
    v = float(np.clip(np.mean((obs - lower)[active]) / scale, 0, 1))
    k = log(slots / (0.05 / 6)) / active.sum()
    if v <= 0.0:
        bound = 0.0
    elif v >= 1.0:
        bound = exp(-k)
    else:
        # Invert binary divergence with a SciPy root solver independent of
        # the production bisection and explicit logs.
        lo = np.finfo(float).tiny

        def equation(u):
            return rel_entr(v, u) + rel_entr(1 - v, 1 - u) - k

        if equation(lo) < 0.0:
            bound = 0.0
        else:
            bound = brentq(equation, lo, v, xtol=2e-15, rtol=1e-15)
    return gain, min(gain, lower.mean() + scale * active.mean() * bound)


def baseline_rows(fit, law):
    xx = FULL[fit[0]]
    yy = fit[1][law]
    ranking = np.argsort(
        np.mean(2 * (xx[:, :, 1] - yy[:, None]) ** 2, axis=0), kind="stable"
    )
    prefix = [
        float(np.mean(2 * (xx[:, ranking[:r], 1].mean(1) - yy) ** 2))
        for r in range(1, 4)
    ]
    best = int(np.argmin(prefix)) + 1
    scaler = StandardScaler().fit(xx.reshape(len(yy), -1))
    lr = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit(
        scaler.transform(xx.reshape(len(yy), -1)), yy
    )
    p = lr.predict_proba(scaler.transform(FULL.reshape(8, -1)))[:, 1]
    return (
        {
            "mean": SUPPORT.mean(1),
            "top_n": SUPPORT[:, ranking[:best]].mean(1),
            "prob_lr": p,
        },
        ranking,
        prefix,
        best,
    )


def check_references(out):
    baselines = pd.read_csv(out / "baselines.csv.gz", keep_default_na=False).set_index(
        ["bank", "law", "split", "method"], verify_integrity=True
    )
    power = pd.read_csv(out / "direct-witness.csv", keep_default_na=False).set_index(
        ["bank", "law", "gate", "slots"], verify_integrity=True
    )
    assert len(baselines) == 576 and len(power) == 384
    accepted = Counter()
    for bank in range(48):
        q, stream = problem(bank)
        ids, labels = stream(41, 2048)
        test_w, y_w = stream(51, 128)
        blocks = [stream(i, 128) for i in (21, *range(31, 37))]
        merged = (
            np.concatenate([row[0] for row in blocks]),
            {law: np.concatenate([row[1][law] for row in blocks]) for law in q},
        )
        for law, target in q.items():
            for split, fit in [("discovery", blocks[0]), ("all_development", merged)]:
                predictions, ranking, sc, n = baseline_rows(fit, law)
                for method, p in predictions.items():
                    record = baselines.loc[bank, law, split, method]
                    assert record.n_labels == len(fit[0])
                    close(json.loads(record.p1), p)
                    confirmed = metrics(p, target, ids, labels[law])
                    for key, value in confirmed.items():
                        close(record[key], value)
                    if method == "top_n":
                        np.testing.assert_array_equal(
                            json.loads(record.ranking), ranking
                        )
                        close(json.loads(record.prefix_brier), sc)
                        assert float(record.top_n) == n
            for slots in (1, 12):
                inc = q["null"][test_w]
                challenger = q["signal"][test_w]
                for rule in ("raw", "kl"):
                    gain, bound = gate(inc, challenger, y_w[law], slots, rule)
                    record = power.loc[bank, law, rule, slots]
                    close(
                        [record.empirical_gain, record.lower_bound, record.true_gain],
                        [gain, bound, 9 / 64 if law == "signal" else -9 / 64],
                    )
                    assert bool(record["accept"]) == (bound > 0)
                    accepted[f"{law}_{rule}_{slots}"] += int(bound > 0)
    return {
        "retrained_or_rebuilt_baselines": len(baselines),
        "direct_witness_gate_replays": len(power),
        "direct_witness_acceptance": dict(accepted),
    }


def cell_power():
    """Condition on observed stream-51 provider vectors, before observing labels.

    Five corner types have width 1 and lower gain -.75, one has width 2 and
    lower gain -1.25. The other two have zero width. Under the signal
    an aligned label occurs independently with probability 7/8 on each
    informative row. Enumerate the two independent binomial counts.
    """
    from scipy.stats import binom

    ans = {}
    for slots in (1, 12):
        probs, max_bounds = [], []
        for bank in range(48):
            q, stream = problem(bank)
            cell, _ = stream(51, 128)
            inc, cand = q["null"][cell], q["signal"][cell]
            delta = cand - inc
            n1, n2 = (
                np.count_nonzero(abs(delta) == 0.25),
                np.count_nonzero(abs(delta) == 0.5),
            )
            m = n1 + n2
            scale = 2 if n2 else 1
            k1, k2 = np.indices((n1 + 1, n2 + 1))
            v = (k1 + 2 * k2) / (m * scale)
            v0 = (0.75 * n1 + 1.25 * n2) / (m * scale)
            threshold = log(6 * slots / 0.05)
            passed = (v > v0) & (
                m * (rel_entr(v, v0) + rel_entr(1 - v, 1 - v0)) > threshold
            )
            probability = np.sum(
                passed * binom.pmf(k1, n1, 0.875) * binom.pmf(k2, n2, 0.875)
            )
            probs.append(float(probability))
            _, h = gate(inc, cand, delta > 0, slots, "kl")
            assert (probability > 0) == (h > 0)
            max_bounds.append(h)
        ans[str(slots)] = {
            "banks": len(probs),
            "blocks_unable_to_accept_for_any_labels": sum(p == 0 for p in probs),
            "conditional_probability_min": min(probs),
            "conditional_probability_mean": sum(probs) / len(probs),
            "conditional_probability_max": max(probs),
            "expected_number_accepting": sum(probs),
            "probability_all_48_reject_conditional_on_inputs": float(
                np.prod(1 - np.array(probs))
            ),
            "max_possible_bound_over_banks": float(max(max_bounds)),
        }
    return ans


def evaluate(tree, support=SUPPORT):
    j = int(tree["id"][1:-4])
    ids, n_ops, total, value = {j}, 0, 1, support[:, j]
    for op in tree["children"]:
        items = [value]
        n_ops += 1
        total += 1
        for child in op["children"]:
            p, other, nested, size = evaluate(child, support)
            items.append(p)
            ids.update(other)
            n_ops += nested
            total += size
        ps = np.stack(items)
        kind = op["type"]
        if kind == "MeanNode":
            value = ps.mean(axis=0)
        elif kind == "WeightedMeanNode":
            value = (ps * np.array(op["weights"])[:, None]).sum(axis=0)
        elif kind in ("CloseThresholdNode", "FarThresholdNode"):
            distances = abs(ps - op["threshold"])
            f = np.argmin if kind == "CloseThresholdNode" else np.argmax
            value = ps[f(distances, axis=0), np.arange(8)]
        elif kind in ("MinNode", "MaxNode"):
            value = (np.min if kind == "MinNode" else np.max)(ps, axis=0)
        else:
            raise ValueError(kind)
    return value, ids, n_ops, total


def archived(out, name):
    from itertools import groupby

    def key(r):
        return r["bank"], r["law"], r["policy"]

    with gzip.open(out / name, "rt") as f:
        for keys, records in groupby((json.loads(line) for line in f), key=key):
            yield keys, list(records)


def check_search(out):
    from statistics import mean, stdev

    from scipy.stats import t

    published = pd.read_csv(out / "search.csv", keep_default_na=False).set_index(
        ["bank", "law", "policy"], verify_integrity=True
    )
    groups = pd.read_csv(out / "paired.csv", keep_default_na=False).set_index(
        ["law", "endpoint", "comparison"]
    )
    nt = Counter()
    data = {}
    all_keys = set()
    exact_ties = []
    rational_support = np.array(
        [[Fraction(v) for v in row] for row in SUPPORT], dtype=object
    )
    for (key, records), (gkey, stages) in zip(
        archived(out, "births.jsonl.gz"),
        archived(out, "generations.jsonl.gz"),
        strict=True,
    ):
        assert key == gkey and key not in all_keys
        bank, law, policy = key
        all_keys.add(key)
        nt["births"] += len(records)
        nt["generations"] += len(stages)
        assert len(stages) == 6 and len(records) == 76
        r = published.loc[key]
        q, stream = problem(bank)
        target = q[law]
        fit, fy = stream(21, 128)
        test, ty = stream(41, 2048)
        a, b, c = np.random.default_rng([4782000 + bank, 17]).permutation(3)
        components = {
            name: p
            for name, p in [
                ("min_ac", np.minimum(SUPPORT[:, a], SUPPORT[:, c])),
                ("min_bc", np.minimum(SUPPORT[:, b], SUPPORT[:, c])),
                ("max_ab", np.maximum(SUPPORT[:, a], SUPPORT[:, b])),
            ]
        }
        risk, programs, ancestry, sizes = {}, {}, {}, {}
        for j, row in enumerate(records):
            assert row["id"] == j
            p, ids, n, size = evaluate(row["function"])
            programs[j] = np.array(row["p1"])
            close(p, programs[j])
            close(row["fit_brier"], np.mean(2 * (p[fit] - fy[law]) ** 2))
            sizes[j] = size
            assert sorted(ids) == row["provider_ids"] and n == row["n_ops"]
            risk[j] = float(np.mean(2 * (p - target) ** 2))
            close(risk[j], row["exact_excess"])
            flag = np.max(abs(p - target)) < 1e-10
            assert flag == row["hits_oracle"]
            partial = {
                name for name, v in components.items() if np.max(abs(p - v)) < 1e-10
            }
            assert partial == set(row["partial"])
            assert all(k < j for k in row["parents"])
            lineage = set(row["parents"]).union(*(ancestry[k] for k in row["parents"]))
            ancestry[j] = lineage
            assert set(row["partial_ancestors"]) == set().union(
                *(records[i]["partial"] for i in lineage)
            )
            assert bool(row["unconfirmed_ancestor"]) == any(
                records[i]["confirmed"] is False for i in lineage
            )
        current = stages[0]["before"]
        assert risk[current] == (0 if law == "null" else 9 / 64)
        initialize = set(range(4))
        survivors = initialize
        adverse = improved = 0
        offers = []
        for idx, step in enumerate(stages, start=1):
            assert step["gen"] == idx and step["before"] == current
            proposals = step["proposals"]
            offers.extend(proposals)
            assert len(proposals) == 12
            assert all(
                set(records[i]["parents"]) <= survivors and records[i]["gen"] == idx
                for i in proposals
            )
            if policy != "ordinary":
                cell, y = stream(30 + idx, 128)
                inc = programs[current][cell]
                scored = [
                    gate(
                        inc,
                        programs[i][cell],
                        y[law],
                        12,
                        "raw" if policy == "open_raw" else "kl",
                    )
                    for i in proposals
                ]
                gains, bounds = zip(*scored, strict=True)
                close(step["gains"], gains)
                close(step["lower_bounds"], bounds)
                nt["replayed_pairwise_confirmation_bounds"] += len(bounds)
                eligible = {
                    i for i, v in zip(proposals, bounds, strict=True) if v > 4e-12
                }
                borderline = {
                    i for i, v in zip(proposals, bounds, strict=True) if abs(v) <= 4e-12
                }
                recorded = set(step["eligible"])
                assert eligible <= recorded <= eligible | borderline
                for i in proposals:
                    assert records[i]["confirmed"] == (i in recorded)
                if any(v > 4e-12 for v in bounds):
                    assert (
                        step["after"] in proposals
                        and abs(bounds[proposals.index(step["after"])] - max(bounds))
                        < 4e-12
                    )
                elif step["after"] != current:
                    assert step["after"] in recorded
                    nt["promotion_within_4e-12_of_zero"] += 1
                    assert policy == "open_raw"
                    previous = evaluate(records[current]["function"], rational_support)[
                        0
                    ]
                    following = evaluate(
                        records[step["after"]]["function"], rational_support
                    )[0]
                    assert all(isinstance(v, Fraction) for v in [*previous, *following])
                    exact = sum(
                        2 * ((p - yval) ** 2 - (qhat - yval) ** 2)
                        for p, qhat, yval in zip(
                            previous[cell],
                            following[cell],
                            map(int, y[law]),
                            strict=True,
                        )
                    ) / len(cell)
                    exact_ties.append(
                        {
                            "bank": bank,
                            "generation": idx,
                            "law": law,
                            "reported_gain": max(step["gains"]),
                            "exact_rational_gate_gain": str(exact),
                            "population_excess_change": risk[step["after"]]
                            - risk[current],
                        }
                    )
                if policy == "strict_kl":
                    assert set(step["survivors"]) <= survivors | recorded
            else:
                assert (
                    not step["gains"]
                    and not step["eligible"]
                    and not step["lower_bounds"]
                )
                choice = min(
                    step["survivors"], key=lambda i: (records[i]["fit_brier"], sizes[i])
                )
                assert choice == step["after"]
            assert (
                set(step["survivors"]) <= survivors | set(proposals)
                and len(step["survivors"]) <= 8
            )
            survivors = set(step["survivors"])
            change = risk[step["after"]] - risk[current]
            adverse += int(change > 1e-12)
            improved += int(change < -1e-12)
            assert (step["promoted"] is not None) == (
                records[current]["function"] != records[step["after"]]["function"]
            )
            current = step["after"]
        assert (
            len(set(offers)) == 72
            and r.deploy_id == current
            and r.n_adverse_moves == adverse
            and r.n_true_improving_moves == improved
        )
        assert bool(r.ever_harmful_move) == (adverse > 0)
        assert r.n_final_ancestors == len(ancestry[current])
        final = metrics(programs[current], target, test, ty[law])
        for col, v in final.items():
            close(r[col], v)
        assert r.final_oracle == records[current]["hits_oracle"]
        assert r.initial_oracle == any(records[i]["hits_oracle"] for i in initialize)
        assert r.offer_oracle == any(records[i]["hits_oracle"] for i in offers)
        assert r.ever_oracle == any(row["hits_oracle"] for row in records)
        assert r.final_better_than_mean == (
            risk[current] < risk[stages[0]["before"]] - 1e-12
        )
        assert r.offered_better_than_mean == (
            min(risk[i] for i in offers) < risk[stages[0]["before"]] - 1e-12
        )
        assert r.deploy_ops == records[current]["n_ops"] and r.n_providers == len(
            records[current]["provider_ids"]
        )
        close(r.best_offered_excess, min(risk[i] for i in offers))
        assert r.unique_genotypes == len(
            {json.dumps(i["function"], sort_keys=True) for i in records}
        )
        assert r.unique_functions == len(
            {tuple(np.round(p, 12)) for p in programs.values()}
        )
        data[key] = {**final, "best_offered_excess": min(risk[i] for i in offers)}
    assert len(all_keys) == 48 * 2 * 4 == len(published)

    for (law, endpoint, pair), record in groups.iterrows():
        right, left = pair.split(" minus ")
        v = [
            data[k, law, right][endpoint] - data[k, law, left][endpoint]
            for k in range(48)
        ]
        m = mean(v)
        radius = stdev(v) / np.sqrt(len(v)) * t.isf(0.025, 47)
        close([record["mean"], record.lower, record.upper], [m, m - radius, m + radius])
        assert record.lower_risk == sum(
            x < -1e-12 for x in v
        ) and record.higher_risk == sum(x > 1e-12 for x in v)
    nt["paired_intervals"] = len(groups)
    nt["searches"] = len(published)
    return {
        **dict(nt),
        "near_zero_raw_promotions": exact_ties,
        "immutable_input_digests": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                out / "births.jsonl.gz",
                out / "generations.jsonl.gz",
                out / "search.csv",
            ]
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    out = parser.parse_args().results
    with threadpool_limits(limits=1):
        result = {
            "references": check_references(out),
            "search": check_search(out),
            "post_hoc_exact_conditional_witness_power_on_stream_51": cell_power(),
        }
    print(json.dumps(result, indent=2))
