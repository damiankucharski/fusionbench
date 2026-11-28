"""Independent audit of archived replacement runs. Use --native-replay with the matching OKAPI checkout."""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import kl_div
from scipy.stats import sem, t

HERE = Path(__file__).resolve().parent
POLICIES = ("ordinary", "open_raw", "open_kl", "strict_kl")


def jsonl(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def check_close(observed, expected):
    np.testing.assert_allclose(observed, expected, atol=5e-12, rtol=0)


def kl_lower(rate, cost):
    if rate == 0:
        return 0.0
    if rate == 1:
        return np.exp(-cost)

    def equation(p):
        return kl_div(rate, p) + kl_div(1 - rate, 1 - p) - cost

    return brentq(equation, np.finfo(float).tiny, rate, xtol=1e-15)


def binary_gates(inc, candidates, y, *, slots=12):
    """Enumerate both realized-label losses, then invert KL independently with SciPy."""
    delta0 = 2 * (inc**2 - candidates**2)
    delta1 = 2 * ((1 - inc) ** 2 - (1 - candidates) ** 2)
    observed = np.where(y, delta1, delta0)
    floor = np.minimum(delta0, delta1)
    width = np.abs(delta1 - delta0)
    active = (width > 0).sum(axis=1)
    totals = (observed - floor).sum(axis=1)
    bounds = floor.mean(axis=1)
    for i in np.flatnonzero(active):
        span = width[i].max()
        rate = np.clip(totals[i] / (span * active[i]), 0, 1)
        bounds[i] += (
            span * active[i] / y.size * kl_lower(rate, np.log(120 * slots) / active[i])
        )
    return observed.mean(axis=1), bounds


def stage_decision(row, policy):
    offers, survivors = row["proposals"], row["survivors"]
    assert len(offers) == len(set(offers)) == 12 and 1 <= len(survivors) <= 8
    bounds = np.asarray(row["lower_bounds"])
    if not len(bounds):
        assert policy == "ordinary" and not row["gains"] and not row["eligible"]
        assert row["after"] in survivors
        return
    eligible = [i for i, b in zip(offers, bounds, strict=True) if b > 0]
    assert eligible == row["eligible"]
    if policy != "ordinary":
        pick = offers[int(np.argmax(bounds))] if eligible else row["before"]
        assert pick == row["after"]
    else:
        assert row["after"] in survivors
    if policy == "strict_kl":
        assert not (set(offers) & set(survivors)) - set(eligible)


def controlled():
    folder = HERE / "controlled/results"
    scores = pd.read_csv(folder / "search.csv", keep_default_na=False).set_index(
        ["bank", "law", "policy"]
    )
    births, stages = defaultdict(list), defaultdict(list)
    for dest, filename in ((births, "births"), (stages, "generations")):
        for record in jsonl(folder / f"{filename}.jsonl.gz"):
            dest[record["bank"], record["law"], record["policy"]].append(record)
    assert len(scores) == len(births) == len(stages) == 384 and scores.index.is_unique
    table = (1 + 6 * np.array(list(product(range(2), repeat=3)))) / 8
    harm = Counter()
    finals = defaultdict(list)
    direct = pd.read_csv(folder / "direct-witness.csv", keep_default_na=False)
    assert len(direct) == 48 * 2 * 2 * 2
    for bank in range(48):
        seed = 4782000 + bank
        a, b, c = np.random.default_rng([seed, 17]).permutation(3)
        null = table.mean(axis=1)
        signal = np.minimum(table[:, c], np.maximum(table[:, a], table[:, b]))

        def draw(stream, n, truth, seed=seed):
            rng = np.random.default_rng([seed, stream])
            indices = rng.integers(0, 2, (n, 3)) @ [4, 2, 1]
            return indices, rng.random(n) < truth[indices]

        for law, truth in (("null", null), ("signal", signal)):
            ix, y = draw(51, 128, truth)
            for slots in (1, 12):
                raw, bound = binary_gates(null[ix], signal[ix][None], y, slots=slots)
                for gate, number in (("raw", raw[0]), ("kl", bound[0])):
                    entry = direct.query(
                        "bank==@bank and law==@law and slots==@slots and gate==@gate"
                    ).iloc[0]
                    check_close(entry.lower_bound, number)
                    assert entry.accept == (number > 0)

            paths = []
            for policy in POLICIES:
                key = bank, law, policy
                records, looks, score = births[key], stages[key], scores.loc[key]
                assert len(records) == score.discovery_evaluations == 76
                assert all(r["id"] == i for i, r in enumerate(records))
                assert score.proposal_count == 72 and len(looks) == 6
                p = np.array([r["p1"] for r in records])
                excess = np.mean(2 * (p - truth) ** 2, axis=1)
                check_close(excess, [r["exact_excess"] for r in records])
                ix, y = draw(21, 128, truth)
                check_close(
                    np.mean(2 * (p[:, ix] - y) ** 2, axis=1),
                    [r["fit_brier"] for r in records],
                )
                live = set(range(4))
                old = looks[0]["before"]
                check_close(p[old], null)
                for gen, row in enumerate(looks, 1):
                    assert row["gen"] == gen and row["before"] == old
                    ids = row["proposals"]
                    assert all(set(records[i]["parents"]) <= live for i in ids)
                    ix, y = draw(30 + gen, 128, truth)
                    if policy == "ordinary":
                        assert not row["gains"] and not row["lower_bounds"]
                    else:
                        gains, bounds = binary_gates(p[old, ix], p[ids][:, ix], y)
                        check_close(gains, row["gains"])
                        check_close(
                            bounds if policy.endswith("kl") else gains,
                            row["lower_bounds"],
                        )
                    stage_decision(row, policy)
                    live = set(row["survivors"])
                    old = row["after"]
                assert old == score.deploy_id
                ix, y = draw(41, 2048, truth)
                check_close(np.mean(2 * (p[old, ix] - y) ** 2), score.test_brier)
                check_close(excess[old], score.exact_excess)
                if law == "null":
                    harm[policy] += any(excess[r["after"]] > 1e-12 for r in looks)
                else:
                    finals[policy].append(excess[old])
                if policy != "strict_kl":
                    paths.append(score.path_digest)
            assert len(set(paths)) == 1
    diffs = np.subtract(finals["strict_kl"], finals["open_kl"])
    ci = t.interval(0.95, len(diffs) - 1, loc=diffs.mean(), scale=sem(diffs))
    reported = (
        pd.read_csv(folder / "paired.csv")
        .query(
            "law=='signal' and endpoint=='exact_excess' and comparison=='strict_kl minus open_kl'"
        )
        .iloc[0]
    )
    check_close([diffs.mean(), *ci], reported[["mean", "lower", "upper"]].astype(float))
    print(
        "Controlled: strict minus open",
        diffs.mean(),
        ci,
        "ever left null optimum",
        dict(harm),
    )


def barrier():
    folder = HERE / "barrier"
    rows = pd.read_csv(folder / "supplied.csv")
    live = pd.read_csv(folder / "live.csv")
    assert (
        not rows.duplicated(["policy", "bank", "generation"]).any()
        and len(rows) == 32 * 6 * 2
    )
    assert not live.duplicated(["method", "bank"]).any() and len(live) == 32 * 10
    corners = np.array(list(product(range(2), repeat=2)))
    a, b = corners.T
    high = 0.125 + 0.75 * (a != b)
    functions = {
        "mean": np.full(4, 0.5),
        "xor": high,
        "u": 0.125 + 0.75 * (a * (1 - b)),
        "v": 0.125 + 0.75 * ((1 - a) * b),
    }
    excess = {name: 2 * np.mean((p - high) ** 2) for name, p in functions.items()}
    assert excess == {"mean": 9 / 32, "xor": 0, "u": 9 / 32, "v": 9 / 32}
    rejected = []
    for (policy, bank), trajectory in rows.groupby(["policy", "bank"]):
        old = "mean"
        allowed = False
        for r in trajectory.itertuples(index=False):
            rng = np.random.default_rng([5910000, bank, r.generation])
            ix = rng.integers(4, size=128)
            y = rng.random(128) < high[ix]
            candidates = [old] * 12
            if r.generation == 1:
                candidates[:2] = ["u", "v"]
            if r.generation == 2 and allowed:
                candidates[0] = "xor"
            assert (r.offered_1, r.offered_2, r.previous) == (*candidates[:2], old)
            _, bound = binary_gates(
                functions[old][ix], np.array([functions[k][ix] for k in candidates]), y
            )
            check_close(bound[0], r.lower_1)
            assert (r.eligible_1, r.eligible_2) == tuple(bound[:2] > 0)
            if r.generation == 1:
                allowed = policy == "open_kl" or np.all(bound[:2] > 0)
            if bound.max() > 0:
                old = candidates[np.argmax(bound)]
            assert old == r.champion and excess[old] == r.excess
            if policy == "open_kl" and r.generation == 2 and old != "xor":
                rejected.append((bank, int(np.sum((high[ix] > 0.5) == y)), bound[0]))
    end = rows.query("generation==6").groupby("policy").champion.value_counts()
    paired = live.query("method in @POLICIES").pivot(
        index="bank", columns="method", values="exact_excess"
    )
    print(
        "Barrier: supplied",
        end.to_dict(),
        "two refusals",
        rejected,
        "live open minus strict",
        (paired.open_kl - paired.strict_kl).mean(),
    )


def panel(replay):
    from fusionbench.scenarios.registry import STANDARD_V3_SCENARIOS

    folder = HERE / "panel/results"
    scores = pd.read_csv(folder / "scores.csv").set_index(["scenario", "rep", "method"])
    banks = json.loads((folder / "banks.json").read_text())
    assert len(scores) == 144 * 14 and scores.index.is_unique and len(banks) == 144
    seen = set()
    keys = set()
    for r in banks:
        k = r["scenario"], r["rep"]
        assert k not in keys
        keys.add(k)
        offset = list(STANDARD_V3_SCENARIOS).index(k[0]) * 1000 + k[1]
        assert k[1] in range(6) and (r["data_seed"], r["search_seed"]) == (
            5230000 + offset,
            7230000 + offset,
        )
        assert (r["n_source"], r["n_test"], r["gate_looks"], r["gate_rows"]) == (
            640,
            1024,
            6,
            64,
        )
        for policy in (*POLICIES, "ordinary_full"):
            score = scores.loc[(*k, policy)]
            assert score.proposal_count == 72
            cnt = 0 if policy == "ordinary_full" else 78
            assert (
                score.gate_evaluations == cnt
                and score.scored_gate_row_instances == 64 * cnt
            )
            assert score.discovery_evaluations == 72 + min(8, r["n_providers"])
            assert (
                score.scored_discovery_row_instances
                == (r["n_providers"] + score.discovery_evaluations)
                * score.discovery_rows
            )
    for g in jsonl(folder / "generations.jsonl.gz"):
        k = g["scenario"], g["rep"], g["method"], g["generation"]
        assert k not in seen and 1 <= k[-1] <= 6
        seen.add(k)
        if k[2] != "ordinary_full":
            stage_decision(
                {**g, "before": g["incumbent_before"], "after": g["incumbent_after"]},
                k[2],
            )
        else:
            assert not g["gains"] and not g["eligible"] and not g["gate_rows"]
        if g["generation"] == 6:
            check_close(g["test_brier_after"], scores.loc[k[:3]].brier)
    assert len(seen) == 4320
    wide = scores.brier.unstack("method")
    paths = scores.path_digest.unstack("method")
    assert (paths.open_kl == paths.open_raw).all() and (
        paths.open_raw == paths.ordinary
    ).all()
    report = pd.read_csv(folder / "contrasts.csv")
    draws = np.random.default_rng(5230999).integers(6, size=(10000, 24, 6))
    for r in report.itertuples(index=False):
        a, b = r.contrast.split("_minus_")
        diff = (wide[a] - wide[b]).unstack("rep").sort_index().to_numpy()
        means = diff[np.arange(24)[None, :, None], draws].mean(axis=(1, 2))
        lo, hi = np.quantile(means, [0.025, 0.975])
        check_close([diff.mean(), lo, hi], [r.mean, r.ci025, r.ci975])
        assert (
            np.sum(diff < -1e-12),
            np.sum(np.abs(diff) <= 1e-12),
            np.sum(diff > 1e-12),
        ) == (r.wins, r.ties, r.losses)
        if a == "open_kl" and b in ("open_raw", "strict_kl"):
            print("Panel", r.contrast, diff.mean(), lo, hi, (r.wins, r.ties, r.losses))
    if replay:
        replay_native(scores)


def replay_native(scores):
    from loguru import logger

    from fusionbench import get_scenario
    from fusionbench.eval.replacement_search import run_replacement_search
    from fusionbench.eval.suite_point_gp import predict_tree
    from fusionbench.scenarios.blocks import draw_block

    logger.disable("okapi")
    key = "multiclass_committee", 0
    law = get_scenario(key[0])
    ref = scores.loc[(*key, "open_kl")]
    x, y, _ = draw_block(law, int(ref.data_seed), 0, 640, split="fit")
    test, labels, _ = draw_block(law, int(ref.data_seed), 1, 1024, split="test")
    fresh, new_y, _ = draw_block(law, int(ref.data_seed), 901, 1024, split="test")
    blocks = [(x[i : i + 64], y[i : i + 64]) for i in range(256, 640, 64)]
    for policy in POLICIES:
        args = {
            "seed": int(ref.search_seed),
            "policy": policy,
            "incumbent": "best_single",
        }
        out = run_replacement_search(x[:256], y[:256], blocks, **args)
        refrow = scores.loc[(*key, policy)]
        assert out.path_digest == refrow.path_digest
        assert out.records[out.champion_id].genotype == refrow.champion_sha256
        p = predict_tree(out.tree, test)
        assert (
            hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
            == refrow.predictions_sha256
        )
        loss = np.mean(np.sum((p - np.eye(p.shape[1])[labels]) ** 2, axis=1))
        check_close(loss, refrow.brier)
        new_p = predict_tree(out.tree, fresh)
        new_loss = np.mean(np.sum((new_p - np.eye(p.shape[1])[new_y]) ** 2, axis=1))
        check_close(p, predict_tree(out.tree, test))
        print(
            "Native replay",
            key,
            policy,
            "original and fresh test Brier",
            loss,
            new_loss,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-replay", action="store_true")
    args = parser.parse_args()
    panel(args.native_replay)
    barrier()
    controlled()
