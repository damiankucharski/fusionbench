"""Fit and score all banks for one registered round; audit inputs enter only after choices."""

import argparse
import importlib
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from banks import banks
from threadpoolctl import threadpool_limits

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REFS = ("mean", "top_n", "prob_lr", "regularized")
CANDIDATES = {
    1: ("gp_top8",),
    2: ("simplex",),
    3: ("bootstrap_medoid",),
    4: ("cf_gain", "cf_margin"),
}


def train(model, x, y):
    c = x.shape[-1]
    return model.fit(x, y, task="binary" if c == 2 else "multiclass", n_classes=c)


def simplex(x, y, regularized=True):
    from fusionbench.methods.simplex_tree import SimplexTree

    return train(SimplexTree(regularized=regularized), x, y)


def variants(r, x, y):
    if r == 2:
        arms = {"simplex": simplex(x, y), "simplex_free": simplex(x, y, False)}
        return arms, {
            k: {
                "weights": v.weights_.tolist()
                if hasattr(v.weights_, "tolist")
                else list(v.weights_),
                "info": v.fit_info_,
            }
            for k, v in arms.items()
        }
    if r == 5:
        selected = json.loads((HERE / "selection.json").read_text())
        k, winning = selected["round"], selected["candidate"]
        if k == 2:
            return {"simplex": simplex(x, y)}, {"candidate": "simplex"}
        models, info = variants(k, x, y)
        finalists = {
            m: models[m]
            for m in dict.fromkeys((winning, "simplex", "regularized"))
            if m in models
        }
        if "simplex" not in finalists:
            finalists["simplex"] = simplex(x, y)
        return finalists, info
    return importlib.import_module(f"r{r}").fit_variants(x, y)


def valid(p, n, c):
    p = np.asarray(p, dtype=float)
    if (
        p.shape != (n, c)
        or not np.isfinite(p).all()
        or np.min(p) < -1e-7
        or not np.allclose(p.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError(f"Invalid forecast of shape {p.shape}")
    return p


def sample_brier(p, y):
    target = np.eye(p.shape[1])[np.asarray(y).reshape(-1)]
    return np.square(p - target).sum(axis=1).mean()


def risk(bank, p):
    if bank.mass is None:
        return float(sample_brier(p, bank.audit_y)), None
    q = np.asarray(bank.q)
    w = np.asarray(bank.mass)
    excess = 2 * np.dot(w, np.square(p[:, 1] - q))
    irreducible = 2 * np.dot(w, q * (1 - q))
    return float(excess + irreducible), float(excess)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def measure(r, bank):
    x, y = bank.x, bank.y
    start = perf_counter()
    models, info = variants(r, x, y)
    variants_seconds = perf_counter() - start
    ref_models = {
        "mean": SimpleAverage(),
        "top_n": TopN(metric="brier"),
        "prob_lr": StackingLR(c_value=1.0),
        "regularized": RegularizedNativePatternOKAPI(seed=0, split_seed=0),
    }
    for name, model in ref_models.items():
        if name not in models:
            models[name] = train(model, x, y)
    # All learned decisions are now frozen.
    rows = []
    for name, m in models.items():
        pred = valid(m.predict(bank.audit_x), len(bank.audit_x), x.shape[-1])
        fit_pred = valid(m.predict(x), len(y), x.shape[-1])
        score, excess = risk(bank, pred)
        row = {
            **bank.context,
            "method": name,
            "test_brier": score,
            "excess": excess,
            "fit_brier": float(sample_brier(fit_pred, y)),
            "test_prediction_hash": sha256(
                np.ascontiguousarray(pred).tobytes()
            ).hexdigest(),
        }
        tree = getattr(m, "tree_", None)
        if tree is None and hasattr(m, "trees_") and hasattr(m, "chosen_"):
            tree = m.trees_[m.chosen_]
        if tree is not None:
            row.update(
                nodes=tree.nodes_count, providers_used=len(tree.unique_value_node_ids)
            )
        elif hasattr(m, "n_selected_"):
            row["providers_used"] = m.n_selected_
        if hasattr(m, "validation_predict"):
            raw = valid(
                m.validation_predict(bank.audit_x), len(bank.audit_x), x.shape[-1]
            )
            row["pre_refit_test"] = risk(bank, raw)[0]
            row["validation_brier"] = m.validation_brier_
        rows.append(row)
    return rows, {
        **bank.context,
        "variants_fit_seconds": variants_seconds,
        "total_seconds": perf_counter() - start,
        "variants": info,
    }


def summarize(tab, r):
    name = "round" + str(r)
    levels = []
    contrasts = []
    families = []
    family_names = sorted(tab.loc[tab.cohort == "standard", "family"].unique())
    assert len(family_names) == 27
    for cohort in ("controlled", "standard", "digits"):
        subset = tab[tab.cohort == cohort]
        subsets = (
            {"standard": subset}
            if cohort == "standard"
            else dict(tuple(subset.groupby("family")))
        )
        for group, part in subsets.items():
            cells = part.pivot(
                index=["family", "seed"], columns="method", values="test_brier"
            ).sort_index()
            assert np.isfinite(cells).all().all()
            nf = cells.index.get_level_values(0).nunique()
            ns = len(cells) // nf
            draws = (
                np.random.default_rng(89000 + r).integers(ns, size=(10000, nf, ns))
                if cohort != "digits"
                else None
            )
            for m in cells.columns:
                levels.append(
                    {
                        "group": group,
                        "method": m,
                        "brier": cells[m].mean(),
                        "families": nf,
                        "seeds_per_family": ns,
                    }
                )
                for ref in REFS:
                    if ref == m:
                        continue
                    v = (cells[m] - cells[ref]).to_numpy().reshape(nf, ns)
                    lo = hi = None
                    if draws is not None:
                        lo, hi = np.quantile(
                            v[np.arange(nf)[None, :, None], draws].mean(axis=(1, 2)),
                            [0.025, 0.975],
                        )
                    contrasts.append(
                        {
                            "group": group,
                            "method": m,
                            "reference": ref,
                            "delta": v.mean(),
                            "lo": lo,
                            "hi": hi,
                            "better_banks": int((v < -1e-12).sum()),
                            "worse_banks": int((v > 1e-12).sum()),
                            "total": v.size,
                        }
                    )
                if cohort == "standard":
                    g = cells[m].groupby("family").mean()
                    d = (cells[m] - cells.regularized).groupby("family").mean()
                    families.extend(
                        {
                            "family": f,
                            "method": m,
                            "brier": g.loc[f],
                            "delta_reg": d.loc[f],
                        }
                        for f in family_names
                    )
    for suffix, table in (
        ("levels", levels),
        ("contrasts", contrasts),
        ("families", families),
    ):
        pd.DataFrame(table).to_csv(OUT / f"{name}-{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, choices=range(1, 6))
    r = parser.parse_args().round
    OUT.mkdir(parents=True, exist_ok=True)
    clock = perf_counter()
    rows = []
    records = []
    with threadpool_limits(limits=1):
        for idx, b in enumerate(banks(r), 1):
            values, info = measure(r, b)
            rows.extend(values)
            records.append(info)
            if idx % 20 == 0:
                print(f"Round {r}: {idx} banks", flush=True)
    assert idx == (220 if r == 5 else 106), idx
    tab = pd.DataFrame(rows)
    assert not tab.duplicated(["cohort", "family", "seed", "method"]).any()
    tab.to_csv(OUT / f"round{r}-scores.csv", index=False)
    with (OUT / f"round{r}-fits.jsonl").open("w") as file:
        for item in records:
            file.write(json.dumps(item, default=json_default, allow_nan=False) + "\n")
    summarize(tab, r)
    (OUT / f"round{r}-run.json").write_text(
        json.dumps(
            {
                "round": r,
                "banks": idx,
                "seconds": perf_counter() - clock,
                "numerical_threads": 1,
                "fit_records": len(tab),
                "bootstrap_seed": 89000 + r,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Round {r} complete", flush=True)


if __name__ == "__main__":
    main()
