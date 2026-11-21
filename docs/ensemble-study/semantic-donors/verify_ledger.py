"""Rebuild the scorer chronology, empirical winners and program families from traces."""

import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def old_reference():
    f = (
        Path(__file__).resolve().parents[1]
        / "selection-precision"
        / "verify_lattice.py"
    )
    sp = importlib.util.spec_from_file_location("finite_density_reference", f)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def verify(folder):
    ref = old_reference()
    tables = ref.signatures()
    lookup = {v.tobytes(): i for i, v in enumerate(tables)}
    static = json.loads((folder / "chain-reference.json").read_text())
    chains = set(static["chain_rule_bits"])
    canal = set(static["canalizing_rule_bits"])
    corners = np.stack(
        [
            np.array([0.90 if x & (1 << j) else 0.10 for j in range(4)])
            for x in range(16)
        ]
    )
    atom_pos = {tuple(row): i for i, row in enumerate(ref.X)}
    ix = np.array([atom_pos[tuple(z)] for z in corners])
    masks = [sum(1 << k for k in range(16) if row[ix[k]] > 0.5) for row in tables]
    full = pd.read_csv(folder / "runs.csv.gz", float_precision="round_trip")
    gen_log = []
    summary = []
    vectors = metrics = programs = rounds = objective_calls = 0
    largest = 0
    with gzip.open(folder / "traces.jsonl.gz", "rt") as fp:
        for ln in fp:
            r = json.loads(ln)
            name, seed, start = r["law"], r["data_seed"], r["gp_seed"]
            q, sets = ref.data(name, seed)
            fit_idx, fy = sets[0]
            hard = lambda v, split, sets=sets: np.mean(
                (v[sets[split][0]] - sets[split][1].astype(float)) ** 2
            )
            costs = np.sum(
                np.rint((20 * tables[:, fit_idx] - 20 * fy) ** 2).astype("int64"),
                axis=1,
            )
            vals = np.sum(
                np.rint((20 * tables[:, sets[1][0]] - 20 * sets[1][1]) ** 2).astype(
                    "int64"
                ),
                axis=1,
            )
            risk = np.mean((tables - q) ** 2, axis=1)
            truth = r["goals"]["population_rule"]
            assert risk[truth] <= risk.min() + 1e-14
            assert r["goals"]["erm_rules"] == list(np.flatnonzero(costs == costs.min()))
            native = r["programs"]
            ids = [lookup[ref.native_values(p).tobytes()] for p in native]
            assert ids == r["rule_ids"]
            programs += len(ids)
            raw = list(r["initial_scores"])
            past = {ids[i] for i in raw}
            previous = None
            first_nonchain = None
            for g in r["generations"]:
                gen = g["generation"]
                assert len(raw) == g["calls_before"] and sorted(past) == g["past_rules"]
                parents = g["parents"]
                new = g["crossover"]
                mutated = g["mutation"]
                if previous is not None:
                    assert parents == previous
                for pos, fpval in zip(parents, g["parent_fitness"], strict=True):
                    assert (
                        abs(round(1 - costs[ids[pos]] / (400 * len(fy)), 3) - fpval)
                        < 2e-12
                    )
                produced = parents + new + mutated
                ordered = list(dict.fromkeys(produced))
                scored = g["scored"]
                assert bool(parents)
                assert scored == ordered[: len(scored)]
                assert (
                    len(scored) == len(ordered)
                    if g["complete"]
                    else len(scored) <= len(ordered)
                )
                if g["survivors"] is not None:
                    assert set(g["survivors"]) <= set(scored)
                previous = g["survivors"]
                live_masks = {masks[ids[i]] for i in parents}
                all_seen = {masks[h] for h in past}
                live_new = len(live_masks - chains)
                if first_nonchain is None and live_new:
                    first_nonchain = gen
                assert g["noncanalizing_live"] == sum(
                    masks[ids[i]] not in canal for i in parents
                )
                assert g["noncanalizing_past"] == len(all_seen - canal)
                productive = {}
                for label, tg in (
                    ("population", {truth}),
                    ("erm", set(r["goals"]["erm_rules"])),
                ):
                    target = {masks[k] for k in tg}

                    def join_count(raw_masks, target=target):
                        pairs = list(itertools.combinations(raw_masks - target, 2))
                        return {
                            "union": sum((a | b) in target for a, b in pairs),
                            "intersection": sum((a & b) in target for a, b in pairs),
                        }

                    assert g["joins"][label]["live"] == join_count(live_masks)
                    assert g["joins"][label]["past"] == join_count(all_seen)
                    productive[label] = sum(
                        ids[parents[left]] not in tg
                        and ids[parents[right]] not in tg
                        and any(ids[p] in tg for p in new[2 * j : 2 * j + 2])
                        for j, (left, right) in enumerate(
                            g.get("sampled_parent_indices", [])
                        )
                    )
                gen_log.append(
                    {
                        "law": name,
                        "data_seed": seed,
                        "gp_seed": start,
                        "generation": gen,
                        "calls_before": len(raw),
                        "distinct_live": len(live_masks),
                        "nonchain_live": live_new,
                        "noncanalizing_live": len(live_masks - canal),
                        "distinct_past": len(all_seen),
                        "nonchain_past": len(all_seen - chains),
                        "noncanalizing_past": len(all_seen - canal),
                        "productive_population_crossovers": productive["population"],
                        "productive_erm_crossovers": productive["erm"],
                    }
                )
                raw.extend(scored)
                past.update(ids[i] for i in scored)
                rounds += 1
            assert raw[:400] == r["archive"] and len(raw) == r["real_calls"]
            objective_calls += len(r["archive"])
            observed = list(dict.fromkeys(ids[i] for i in r["archive"]))
            selected = {
                "fit": min(observed, key=lambda i: costs[i]),
                "validation": min(observed, key=lambda i: (vals[i], costs[i])),
            }
            index = full[(full.law == name) & (full.data_seed == seed)]
            base, rank, prefix = ref.fit_baselines(fit_idx, fy)
            # Validate each ordinary baseline once per bank, and both choices per trajectory.
            sub = index[
                index.gp_seed.eq(start) | (index.gp_seed.isna() & (start == 809))
            ]
            for v in sub.itertuples():
                if v.method == "gp":
                    assert v.rule == selected[v.selector]
                    pred = tables[int(v.rule)]
                elif v.method == "exact_erm":
                    pred = tables[costs.argmin()]
                else:
                    pred = base[v.method]
                if v.method == "top_n":
                    assert json.loads(v.ranking) == list(
                        rank
                    ) and v.n_selected == 1 + np.argmin(prefix)
                    np.testing.assert_allclose(
                        json.loads(v.prefix_scores), prefix, atol=1e-14, rtol=0
                    )
                assert (
                    hashlib.sha256(np.asarray(pred, dtype="<f8").tobytes()).hexdigest()
                    == v.prediction_sha256
                )
                statistical = lambda weights, pred=pred, q=q: np.sum(
                    weights * ((pred - q) ** 2 + q * (1 - q))
                )
                target_err = np.sum((tables - q) ** 2 * ref.UNIFORM, axis=1).min()
                expected = {
                    "fit_brier": hard(pred, 0),
                    "validation_brier": hard(pred, 1),
                    "iid_test_brier": hard(pred, 2),
                    "shift_test_brier": hard(pred, 3),
                    "iid_risk": statistical(ref.UNIFORM),
                    "shift_risk": statistical(ref.SHIFT),
                    "fit_minus_family_erm": hard(pred, 0)
                    - costs.min() / (400 * len(fy)),
                    "iid_minus_family": np.sum((pred - q) ** 2 * ref.UNIFORM)
                    - target_err,
                }
                err = max(abs(val - getattr(v, k)) for k, val in expected.items())
                assert err < 2e-13, (name, seed, v.method, err)
                largest = max(largest, err)
                vectors += 1
                metrics += len(expected)
            summary.append(
                {
                    "law": name,
                    "data_seed": seed,
                    "gp_seed": start,
                    "first_live_nonchain": first_nonchain,
                    "n_seen": len(observed),
                    "n_nonchain": len({masks[i] for i in observed} - chains),
                    "n_noncanalizing": len({masks[i] for i in observed} - canal),
                }
            )
    pd.DataFrame(gen_log).to_csv(folder / "shape-generations.csv.gz", index=False)
    pd.DataFrame(summary).to_csv(folder / "shape-runs.csv", index=False)
    result = {
        "trajectories": len(summary),
        "archived_objective_calls": objective_calls,
        "programs_reinterpreted": programs,
        "generation_ledgers": rounds,
        "checked_probability_vectors": vectors,
        "scalar_metrics": metrics,
        "max_error": largest,
    }
    assert vectors == len(full) and len(summary) == 64 and objective_calls == 25600
    (folder / "ledger-verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    verify(p.parse_args().output)
