"""Characterization report: method x scenario matrix, winners, regret, cost."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.metrics import HIGHER_BETTER


def add_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``test_primary`` column using each scenario's primary metric."""
    df = df.copy()
    df["test_primary"] = [
        row[f"test_{row['primary_metric']}"] if row["error"] == "" else np.nan
        for _, row in df.iterrows()
    ]
    return df


def regret_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario x method matrix of regret-to-oracle (lower is better)."""
    comp = df[(df["method"] != "oracle") & (df["error"] == "")]
    if comp.empty:
        return pd.DataFrame()
    return comp.pivot_table(index="scenario", columns="method", values="regret")


def winners(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario best competing method and whether it was expected."""
    comp = df[(df["method"] != "oracle") & (df["error"] == "")]
    comp = add_primary(comp).dropna(subset=["test_primary"])
    rows = []
    for scenario, g in comp.groupby("scenario", sort=False):
        primary = g["primary_metric"].iloc[0]
        best_idx = g["test_primary"].idxmax() if HIGHER_BETTER[primary] else g["test_primary"].idxmin()
        best = g.loc[best_idx, "method"]
        expected = sorted(g[g["expected_winner"]]["method"].tolist())
        rows.append({
            "scenario": scenario,
            "primary_metric": primary,
            "best_test_score": g.loc[best_idx, "test_primary"],
            "best_method": best,
            "best_regret": float(g.loc[best_idx, "regret"]),
            "expected_winners": ", ".join(expected),
            "as_expected": best in expected if expected else None,
        })
    return pd.DataFrame(rows, columns=[
        "scenario", "primary_metric", "best_test_score", "best_method",
        "best_regret", "expected_winners", "as_expected",
    ])


def save_report(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = add_primary(df)
    df.to_csv(out / "results.csv", index=False)

    rmat = regret_matrix(df)
    rmat.to_csv(out / "regret_matrix.csv")
    win = winners(df)
    win.to_csv(out / "winners.csv", index=False)

    lines = ["# fusionbench characterization report", ""]
    if not rmat.empty:
        lines += ["## Regret to oracle (lower is better)", "", rmat.round(4).to_markdown(), ""]
    comp = df[(df["method"] != "oracle") & (df["error"] == "")]
    for (scenario, metric), runs in comp.groupby(["scenario", "primary_metric"]):
        order = runs.sort_values("test_primary", ascending=not HIGHER_BETTER[metric])
        lines += [f"## {scenario}", "", order[["method", "test_primary"]].rename(
            columns={"test_primary": metric}).round(4).to_markdown(index=False), ""]
    failed = df.loc[df["error"] != "", ["scenario", "method", "error"]]
    if not failed.empty:
        lines += ["## Failed methods", "", failed.to_markdown(index=False), ""]
    lines += ["## Per-scenario winners", "", win.round(4).to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(lines))
    return out
