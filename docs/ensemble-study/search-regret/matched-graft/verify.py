"""Independently reconstruct the matched-graft panel, interventions and recorded losses."""

import argparse
import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the independent NumPy interpreter and brute-force enumerator, not the GP
# evaluator, the graft constructor or the fitted one-gate selection routine.
previous = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "native-switch" / "verify.py")
)
evaluate = previous["evaluate"]
literal_optimum = previous["literal_optimum"]

REGIMES = ("far", "close", "linear")
SPLITS = ("fit", "validation", "far_test", "close_test", "linear_test")
OFFERS = ("fit_pair_fit_cut", "fit_pair_half", "duplicate_fit_cut", "duplicate_half")
ARMS = (*OFFERS, "reorder", "archive_only")
PLAIN = ("plain_tail", "plain_forward_tail", "plain_candidate_tail")


def make_draws(seed, splits=((0, 512), (1, 256), (2, 8192))):
    roles = np.random.default_rng([seed, 15]).permutation(6)[:2]
    draws = []
    for split, size in splits:
        rng = np.random.default_rng([seed, split])
        ticks = rng.integers(1, 64, size=(size, 6))
        x = ticks / 64
        u = rng.random(size)
        a, b = ticks[:, roles].T
        logistic = expit(0.2 + 0.7 * logit(a / 64) + 0.7 * logit(b / 64))
        np.testing.assert_allclose(
            logistic,
            expit(0.2 + 0.7 * np.log(a / (64 - a)) + 0.7 * np.log(b / (64 - b))),
            atol=2e-16,
            rtol=1e-15,
        )
        q = {
            "far": np.where(abs(a - 27) >= abs(b - 27), a, b) / 64,
            "close": np.where(abs(a - 37) <= abs(b - 37), a, b) / 64,
            "linear": logistic,
        }
        draws.append((x, q, {k: (u < v).astype(int) for k, v in q.items()}))
    return draws, roles


def make_baselines(fit, y, matrices):
    rank = np.argsort(np.mean((fit - y[:, None]) ** 2, axis=0), kind="stable")
    prefixes = [
        2 * np.mean((fit[:, rank[:k]].mean(1) - y) ** 2)
        for k in range(1, fit.shape[1] + 1)
    ]
    n = int(np.argmin(prefixes)) + 1

    def features(x):
        p = np.stack([1 - x, x], axis=2).reshape(len(x), -1)
        return np.log(p / (1 - p))

    stack = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000, solver="lbfgs")
    )
    stack.fit(features(fit), y)

    def positive_normalized(p):
        q = np.clip(p, 1e-12, None)
        return q[:, 1] / q.sum(axis=1)

    return {
        "average": [x.mean(1) for x in matrices],
        "top_n": [
            positive_normalized(
                np.stack((1 - x[:, rank[:n]], x[:, rank[:n]]), axis=-1).mean(axis=1)
            )
            for x in matrices
        ],
        "stack_logit": [
            positive_normalized(stack.predict_proba(features(x))) for x in matrices
        ],
    }, (n, rank, prefixes)


def subtree(program, path):
    node = program
    for index in path:
        node = node["children"][index]
    return node


def count_nodes(program):
    return 1 + sum(count_nodes(child) for child in program["children"])


def verify_graft_layout(raw, offers, original):
    first = offers[OFFERS[0]]
    if first == original:
        assert raw.variant_forwards == 0 and not raw.admitted
        assert all(edit == original for edit in offers.values())
        return
    assert raw.variant_forwards == 3
    site = json.loads(raw.site_path)
    anchor = subtree(original, site)
    assert anchor["id"] == raw.anchor and not anchor["children"]
    assert (
        raw.chosen_partner in json.loads(raw.operand_order)
        and raw.chosen_partner != raw.anchor
    )
    for name, actual in offers.items():
        expected = deepcopy(original)
        terminal = subtree(expected, site)
        close = raw.direction == "close"
        assert raw.direction in ("close", "far")
        terminal["children"].append(
            {
                "type": "CloseThresholdNode" if close else "FarThresholdNode",
                "threshold": 0.5 if name.endswith("half") else raw.fitted_threshold,
                "close": close,
                "children": [
                    {
                        "type": "ValueNode",
                        "id": raw.anchor
                        if name.startswith("duplicate")
                        else raw.chosen_partner,
                        "children": [],
                    }
                ],
            }
        )
        assert expected == actual, (name, raw.data_seed, raw.gp_seed, raw.fit_regime)


def verify_parent_slots(tails, admission, order, offers):
    fields = (
        "state_parent_keys",
        "state_parent_fitness",
        "state_parent_mutation_chances",
    )
    ledgers = {
        name: tuple(zip(*(json.loads(row[field]) for field in fields), strict=True))
        for name, row in tails.iterrows()
    }
    original = ledgers["archive_only"]
    assert len(original) == 6
    for name, entries in ledgers.items():
        assert len(entries) == 6
        assert all(len(key) == 64 for key, _, _ in entries)
        assert tails.loc[name, "state_duplicate_structures"] == 6 - len(
            {e[0] for e in entries}
        )
        assert all(
            np.isfinite(fitness) and 0 <= chance <= 1 for _, fitness, chance in entries
        )
        if not admission.admitted or name == "archive_only":
            assert entries == original
            continue
        slot = int(admission.slot)
        for i, parent in enumerate(order):
            if i != slot:
                assert entries[i] == original[parent]
        if name == "reorder":
            assert entries[slot] == original[int(admission.displaced)]
        assert entries[slot][1] == tails.loc[name, "state_slot_fitness"]
    if admission.admitted:
        slot = int(admission.slot)
        assert len({ledgers[k][slot][2] for k in OFFERS}) == 1
        for a in OFFERS:
            for b in OFFERS:
                assert (offers[a] == offers[b]) == (
                    ledgers[a][slot][0] == ledgers[b][slot][0]
                )
        assert ledgers[OFFERS[0]][slot][0] not in {e[0] for e in original}


def verify_occurrences(row):
    records = json.loads(row.winner_occurrences)
    program = json.loads(row.program)
    paths = {tuple(o["path"]) for o in records}
    assert len(paths) == len(records)

    def gate_at(path):
        assert path
        node = subtree(program, path)
        assert node["type"] in ("FarThresholdNode", "CloseThresholdNode")
        return node

    for o in records:
        path = o["path"]
        node = gate_at(path)
        assert o["threshold"] == node["threshold"]
        assert o["direction"] == ("close" if node["close"] else "far")
        assert o["operand_ids"] == [
            subtree(program, path[:-1])["id"],
            *(c["id"] for c in node["children"]),
        ]
    if row.method in ARMS and row.decision == "fit":
        ever_paths = json.loads(row.lineage_winner_ever_paths)
        for path in ever_paths:
            gate_at(path)
        assert bool(ever_paths) == row.lineage_winner_ever_marked
        assert bool(records) == row.lineage_winner_first_marked
        exact = any(
            r["threshold"] == row.fitted_threshold and r["direction"] == row.direction
            for r in records
        )
        assert exact == row.lineage_winner_first_exact


def verify_restart(group, matrices):
    fits = group[(group.split == "fit") & (group.decision == "fit")].set_index("method")
    assert (
        len(fits) == 1 + len(OFFERS) + len(ARMS) + len(PLAIN) and fits.index.is_unique
    )
    for method, records in group.groupby("method"):
        expected_decisions = (
            {"fit"} if method.startswith("immediate_") else {"fit", "validation"}
        )
        assert set(records.decision) == expected_decisions
        for _, selected in records.groupby("decision"):
            assert (
                set(selected.split) == set(SPLITS) and selected.program.nunique() == 1
            )
    p_fields = [field for field in group if field.startswith(("state_", "lineage_"))]
    p_fields += ["winner_occurrences"]
    assert (
        (group.groupby(["method", "decision"])[p_fields].nunique(dropna=False) == 1)
        .all()
        .all()
    )
    for record in group[group.split == "fit"].itertuples():
        verify_occurrences(record)
    raw = fits.loc["gp_initial"]
    base = int(raw.base_calls)
    assert base >= 1200 and raw.calls == base and raw.graft_forwards == 41
    for name, offset in zip(
        PLAIN,
        (
            300,
            300 + raw.graft_forwards + raw.variant_forwards,
            300 + raw.graft_forwards + raw.variant_forwards + raw.threshold_candidates,
        ),
        strict=True,
    ):
        assert fits.loc[name, "calls"] == base + offset
    for arm in ARMS:
        assert fits.loc[arm, "calls"] == base + 300
    offers = {k: json.loads(fits.loc["immediate_" + k, "program"]) for k in OFFERS}
    original = json.loads(raw.program)
    verify_graft_layout(raw, offers, original)
    for name in OFFERS:
        assert fits.loc["immediate_" + name, "calls"] == base
        assert (
            fits.loc["immediate_" + name, "free_admission"]
            == fits.loc[name, "free_admission"]
        )
    for x in matrices:
        initial = evaluate(original, x)
        for name in OFFERS[2:]:
            assert np.array_equal(initial, evaluate(offers[name], x))

    assert fits.loc[OFFERS[0], "free_admission"] == raw.status
    assert (raw.status == "admitted") == bool(raw.admitted)
    order = json.loads(raw.parent_order)
    assert len(order) == 6
    tails = fits.loc[list(ARMS)]
    verify_parent_slots(tails, raw, order, offers)
    assert (tails.state_rng_before == raw.rng_before).all()
    assert tails.loc["archive_only", "state_rng_after"] == raw.rng_before
    assert (
        fits.loc["plain_tail", "path_sha256"]
        == tails.loc["archive_only", "path_sha256"]
    )
    assert (tails.state_rng_final.str.len() == 64).all()
    assert (tails.state_population_digest.str.len() == 64).all()
    assert (tails.state_rng_after.str.len() == 64).all()
    if raw.admitted:
        slot, displaced = int(raw.slot), int(raw.displaced)
        assert order[slot] == -1 and sorted(order) == sorted(
            [i for i in range(6) if i != displaced] + [-1]
        )
        assert (
            tails.drop("archive_only").state_rng_after == raw.rng_after_admission
        ).all()
        for arm in OFFERS:
            score = fits.loc["immediate_" + arm, "observed_brier"]
            assert tails.loc[arm, "state_slot_fitness"] == np.round(1 - score, 3)
        assert pd.isna(tails.loc["archive_only", "state_slot_fitness"])
    else:
        assert pd.isna(raw.slot) and pd.isna(raw.displaced) and order == list(range(6))
        assert raw.rng_before == raw.rng_after_admission
        assert tails.state_slot_fitness.isna().all()
        assert tails.state_rng_after.nunique() == tails.state_rng_final.nunique() == 1
        assert (
            tails.state_population_digest.nunique() == tails.path_sha256.nunique() == 1
        )
        for _, rows in group[group.method.isin(ARMS)].groupby(["decision", "split"]):
            assert len(rows) == 6 and rows.program.nunique() == 1
            assert (
                rows.observed_brier.nunique() == rows.conditional_excess.nunique() == 1
            )
    for row in tails.itertuples():
        assert 0 <= row.lineage_marked_unique <= row.lineage_marked_evaluations <= 300
        assert row.lineage_marked_occurrences >= row.lineage_marked_evaluations
        if not raw.admitted or row.Index in ("archive_only", "reorder"):
            assert row.lineage_marked_evaluations == 0
            assert (
                not row.lineage_winner_ever_marked
                and not row.lineage_winner_first_marked
            )
    return int(raw.admitted)


def verify(path):
    frame = pd.read_csv(path / "runs.csv.gz", float_precision="round_trip")
    assert set(frame.data_seed) == set(range(4100, 4116))
    assert set(frame.fit_regime) == set(REGIMES) and set(frame.split) == set(SPLITS)
    assert not frame.duplicated(
        ["data_seed", "fit_regime", "gp_seed", "decision", "method", "split"]
    ).any()
    assert len(frame) == 16 * 3 * (5 + 2 * 24) * 5
    n_losses = n_programs = certificates = restarts = admitted = 0
    max_error = 0.0
    for seed, bank in frame.groupby("data_seed"):
        draws, roles = make_draws(seed)
        for regime, group in bank.groupby("fit_regime"):
            matrices = [draws[i][0] for i in (0, 1, 2, 2, 2)]
            qs = [
                draws[i][1][r]
                for i, r in zip(
                    (0, 1, 2, 2, 2), (regime, regime, *REGIMES), strict=True
                )
            ]
            ys = [
                draws[i][2][r]
                for i, r in zip(
                    (0, 1, 2, 2, 2), (regime, regime, *REGIMES), strict=True
                )
            ]
            base, (n, rank, prefixes) = make_baselines(matrices[0], ys[0], matrices)
            base["known_q"] = qs
            gate = json.loads(group[group.method == "one_gate"].iloc[0].program)
            loss, ties, winner = literal_optimum(matrices[0], ys[0], False)
            assert (
                gate["parent"],
                gate["child"],
                gate["direction"],
                gate["threshold"],
            ) == winner
            assert (
                gate["loss_numerator"],
                gate["loss_denominator"],
                gate["minimizers_on_grid"],
            ) == (loss, 512 * 64**2, ties)
            certificates += 1
            checked = {}
            for row in group.itertuples():
                index = SPLITS.index(row.split)
                if row.method == "top_n":
                    assert row.n_selected == n and np.array_equal(
                        json.loads(row.ranking), rank
                    )
                    np.testing.assert_allclose(
                        json.loads(row.prefix_scores), prefixes, atol=3e-15, rtol=0
                    )
                if row.method in base:
                    p = base[row.method][index]
                else:
                    assert isinstance(row.program, str)
                    idx = min(index, 2)
                    cache_key = row.program, idx
                    if cache_key not in checked:
                        x = matrices[idx]
                        tree = json.loads(row.program)
                        if row.method == "one_gate":
                            assert tree == gate and row.nodes == 3
                            a, b = x[:, tree["parent"]], x[:, tree["child"]]
                            da, db = (
                                abs(a - tree["threshold"]),
                                abs(b - tree["threshold"]),
                            )
                            checked[cache_key] = np.where(
                                da <= db if tree["direction"] == "close" else da >= db,
                                a,
                                b,
                            )
                        else:
                            assert count_nodes(tree) == row.nodes
                            checked[cache_key] = evaluate(tree, x)
                    p = checked[cache_key]
                    n_programs += 1
                digest = hashlib.sha256(
                    np.ascontiguousarray(p, dtype="<f8").tobytes()
                ).hexdigest()
                assert digest == row.prediction_sha256, (
                    seed,
                    regime,
                    row.method,
                    row.split,
                    "prediction digest",
                )
                error = np.abs(
                    np.asarray(
                        [np.mean((p - ys[index]) ** 2), np.mean((p - qs[index]) ** 2)]
                    )
                    - [row.observed_brier, row.conditional_excess]
                )
                assert error.max() < 1e-13, (seed, regime, row.method, row.split, error)
                max_error = max(max_error, float(error.max()))
                n_losses += 2
            restarted = {
                int(k): g for k, g in group[group.gp_seed.notna()].groupby("gp_seed")
            }
            assert set(restarted) == {311, 313}
            for samples in restarted.values():
                selected = samples[samples.chosen_partner.notna()]
                for row in (
                    selected[["signal_pair", "anchor", "chosen_partner"]]
                    .drop_duplicates()
                    .itertuples()
                ):
                    assert row.signal_pair == (
                        {row.anchor, row.chosen_partner} == {f"p{i}.npy" for i in roles}
                    )
                admitted += verify_restart(samples, matrices)
                restarts += 1
    verification = {
        "okapi_commit": "4c784f6a217675d1305d378eaeb525738c37761d",
        "protocol_commit": "8df5db69e8b92964fbf69daa1eeee21dc48d3c7a",
        "input_sha256": hashlib.sha256((path / "runs.csv.gz").read_bytes()).hexdigest(),
        "banks": int(frame.data_seed.nunique()),
        "prediction_vectors": len(frame),
        "losses": n_losses,
        "decoded_program_split_rows": n_programs,
        "exact_single_gate_certificates": certificates,
        "matched_restarts": restarts,
        "admissions": admitted,
        "nonadmissions": restarts - admitted,
        "duplicate_vs_original_split_comparisons": restarts * 2 * len(SPLITS),
        "verified_parent_slots": restarts * 6 * len(ARMS),
        "max_absolute_loss_difference": max_error,
    }
    (path / "verified.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        f"Verified {n_losses} losses, including {n_programs} decoded program/split rows, "
        f"{certificates} exact gate winners, {restarts} four-cell interventions and admission ledgers "
        f"({admitted} admitted, {restarts - admitted} not admitted). Largest loss error {max_error:.3g}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
