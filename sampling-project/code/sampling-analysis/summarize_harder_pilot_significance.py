#!/usr/bin/env python3
"""
Recompute harder-task pilot significance summaries from paired deltas.

Inputs:
  - paired_deltas_running.csv with columns: seed,budget,dq,dplanq,dtime,dcost

Outputs (same directory by default):
  - significance_snapshot.csv
  - per_budget_snapshot.csv
  - evidence_snapshot.md
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd


METRICS = [
    ("dq", "higher"),
    ("dplanq", "higher"),
    ("dtime", "lower"),
    ("dcost", "lower"),
]


def two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    k = int(min(wins, losses))
    cdf = sum(comb(n, i) for i in range(0, k + 1)) / float(2**n)
    return min(1.0, 2.0 * cdf)


def _wins_losses_ties(series: pd.Series, direction: str) -> tuple[int, int, int]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if direction == "higher":
        wins = int((vals > 0).sum())
        losses = int((vals < 0).sum())
    else:
        wins = int((vals < 0).sum())
        losses = int((vals > 0).sum())
    ties = int((vals == 0).sum())
    return wins, losses, ties


def summarize_significance(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric, direction in METRICS:
        if metric not in paired.columns:
            continue
        vals = pd.to_numeric(paired[metric], errors="coerce")
        wins, losses, ties = _wins_losses_ties(vals, direction)
        pval = two_sided_sign_test_pvalue(wins, losses)
        rows.append(
            {
                "metric": metric,
                "direction_better_for_stratified": direction,
                "n_pairs": int(vals.notna().sum()),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_delta": float(vals.mean()) if vals.notna().any() else 0.0,
                "median_delta": float(vals.median()) if vals.notna().any() else 0.0,
                "p_value_sign_test_two_sided": float(pval),
                "significant_p_lt_0_05": bool(pval < 0.05),
            }
        )
    return pd.DataFrame(rows)


def summarize_per_budget(paired: pd.DataFrame) -> pd.DataFrame:
    if "budget" not in paired.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for budget, grp in paired.groupby("budget", dropna=False):
        vals_dq = pd.to_numeric(grp.get("dq"), errors="coerce")
        vals_dtime = pd.to_numeric(grp.get("dtime"), errors="coerce")
        vals_dcost = pd.to_numeric(grp.get("dcost"), errors="coerce")
        rows.append(
            {
                "budget": int(budget),
                "dq_mean": float(vals_dq.mean()) if vals_dq.notna().any() else 0.0,
                "dtime_mean_s": float(vals_dtime.mean()) if vals_dtime.notna().any() else 0.0,
                "dcost_mean": float(vals_dcost.mean()) if vals_dcost.notna().any() else 0.0,
                "time_wins": int((vals_dtime < 0).sum()),
                "time_losses": int((vals_dtime > 0).sum()),
                "cost_wins": int((vals_dcost < 0).sum()),
                "cost_losses": int((vals_dcost > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("budget")


def write_markdown(
    out_md: Path, significance: pd.DataFrame, per_budget: pd.DataFrame, n_pairs_total: int
) -> None:
    lines: list[str] = []
    lines.append("# Significance snapshot (recomputed)")
    lines.append(f"Total paired comparisons: {n_pairs_total}")
    for _, row in significance.iterrows():
        lines.append(
            "- {metric}: wins={wins}, losses={losses}, ties={ties}, mean_delta={mean:.6f}, "
            "p={p:.8g}, significant={sig}".format(
                metric=row["metric"],
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                ties=int(row["ties"]),
                mean=float(row["mean_delta"]),
                p=float(row["p_value_sign_test_two_sided"]),
                sig=bool(row["significant_p_lt_0_05"]),
            )
        )
    if not per_budget.empty:
        lines.append("")
        lines.append("Per-budget means:")
        for _, row in per_budget.iterrows():
            lines.append(
                "- budget {b}: dq_mean={dq:.6f}, dtime_mean_s={dt:.3f}, dcost_mean={dc:.6f}, "
                "time_w/l={tw}/{tl}, cost_w/l={cw}/{cl}".format(
                    b=int(row["budget"]),
                    dq=float(row["dq_mean"]),
                    dt=float(row["dtime_mean_s"]),
                    dc=float(row["dcost_mean"]),
                    tw=int(row["time_wins"]),
                    tl=int(row["time_losses"]),
                    cw=int(row["cost_wins"]),
                    cl=int(row["cost_losses"]),
                )
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-deltas-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/paired_deltas_running.csv"
        ),
    )
    parser.add_argument(
        "--out-significance-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/significance_snapshot.csv"
        ),
    )
    parser.add_argument(
        "--out-per-budget-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/per_budget_snapshot.csv"
        ),
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/evidence_snapshot.md"
        ),
    )
    args = parser.parse_args()

    paired = pd.read_csv(args.paired_deltas_csv)
    significance = summarize_significance(paired)
    per_budget = summarize_per_budget(paired)

    args.out_significance_csv.parent.mkdir(parents=True, exist_ok=True)
    significance.to_csv(args.out_significance_csv, index=False)
    per_budget.to_csv(args.out_per_budget_csv, index=False)
    n_pairs_total = int(len(paired))
    write_markdown(args.out_markdown, significance, per_budget, n_pairs_total)

    print(f"Wrote: {args.out_significance_csv}")
    print(f"Wrote: {args.out_per_budget_csv}")
    print(f"Wrote: {args.out_markdown}")


if __name__ == "__main__":
    main()
