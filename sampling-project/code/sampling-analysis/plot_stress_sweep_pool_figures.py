#!/usr/bin/env python3
"""
Experiment 2 (stress-pool sweeps): figures from pooled paired random vs stratified deltas.

Reads stress_ab_all_runs.csv (same schema as scripts/run_stress_test_sweep.py output).

Writes under --out-dir (default: stress_test_dataset_analysis/figures/):
  - stress_pool_win_loss.png — stacked wins/losses/ties per metric (pooled across pairs)
  - stress_pool_paired_dtime.png — paired Δ wall-clock time per (dataset, seed, budget)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_SPECS: list[tuple[str, str, str]] = [
    ("mean_sentinel_quality", "higher", "Sentinel quality\n(Δ>0 = stratified higher)"),
    ("mean_plan_quality", "higher", "Plan quality\n(Δ>0 = stratified higher)"),
    ("total_time_s", "lower", "Wall time\n(Δ<0 = stratified faster)"),
    ("total_cost", "lower", "Total cost\n(Δ<0 = stratified cheaper)"),
]


def _two_sided_binom_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return float("nan")
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * p)


def _paired_metric_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    piv = df.pivot_table(
        index=["dataset_id", "exp_seed", "sample_budget"],
        columns="mode",
        values=metric_col,
        aggfunc="mean",
    )
    if "random" not in piv.columns or "stratified" not in piv.columns:
        return pd.DataFrame()
    p = piv[["random", "stratified"]].dropna().copy()
    p["delta"] = p["stratified"] - p["random"]
    return p.reset_index()


def _wins_losses_ties(d: pd.Series, direction: str) -> tuple[int, int, int]:
    if direction == "higher":
        wins = int((d > 0).sum())
        losses = int((d < 0).sum())
    else:
        wins = int((d < 0).sum())
        losses = int((d > 0).sum())
    ties = int((d == 0).sum())
    return wins, losses, ties


def _plot_win_loss(
    all_runs: pd.DataFrame,
    out: Path,
    title: str = "Experiment 2: stress-pool sweeps — pooled sign-test counts",
) -> int:
    labels: list[str] = []
    wins_l: list[int] = []
    losses_l: list[int] = []
    ties_l: list[int] = []
    n_pairs = 0

    for metric, direction, disp in METRIC_SPECS:
        if metric not in all_runs.columns:
            continue
        pt = _paired_metric_table(all_runs, metric)
        if pt.empty:
            continue
        d = pt["delta"]
        n_pairs = max(n_pairs, int(len(d)))
        w, l, t = _wins_losses_ties(d, direction)
        labels.append(disp)
        wins_l.append(w)
        losses_l.append(l)
        ties_l.append(t)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.55
    ax.bar(x, wins_l, width, label="Stratified better", color="#2ca02c")
    ax.bar(x, losses_l, width, bottom=wins_l, label="Random better", color="#d62728")
    ax.bar(
        x,
        ties_l,
        width,
        bottom=np.array(wins_l) + np.array(losses_l),
        label="Tie",
        color="#7f7f7f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Paired comparisons (n={n_pairs})")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ymax = max(n_pairs + 1, 1)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return n_pairs


def _plot_paired_dtime(all_runs: pd.DataFrame, out: Path) -> None:
    pt = _paired_metric_table(all_runs, "total_time_s")
    if pt.empty:
        return
    pt = pt.sort_values(["dataset_id", "exp_seed", "sample_budget"])
    pt["label"] = pt.apply(
        lambda r: f"{r['dataset_id']}|{int(r['exp_seed'])}|{int(r['sample_budget'])}",
        axis=1,
    )
    colors = np.where(pt["delta"] < 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(pt))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.scatter(x, pt["delta"], c=colors, s=40, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pt["label"].tolist(), fontsize=5.5, rotation=55, ha="right")
    ax.set_ylabel("Δ Wall time (s)\n(stratified − random)")
    ax.set_title("Stress sweep: paired time deltas (mixed; pooled sign test not significant)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_significance(all_runs: pd.DataFrame) -> None:
    for metric, direction, _ in METRIC_SPECS:
        if metric not in all_runs.columns:
            continue
        pt = _paired_metric_table(all_runs, metric)
        if pt.empty:
            continue
        d = pt["delta"]
        w, l, t = _wins_losses_ties(d, direction)
        pval = _two_sided_binom_p(w, l)
        sig = (not math.isnan(pval)) and pval < 0.05
        print(f"{metric}: n={len(d)} wins={w} losses={l} ties={t} p={pval:.6g} significant={sig}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stress-ab-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/stress_ab_all_runs.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/figures"
        ),
    )
    args = parser.parse_args()

    all_runs = pd.read_csv(args.stress_ab_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p1 = args.out_dir / "stress_pool_win_loss.png"
    p2 = args.out_dir / "stress_pool_paired_dtime.png"
    n = _plot_win_loss(all_runs, p1)
    _plot_paired_dtime(all_runs, p2)
    print(f"Pooled pairs: {n}")
    _print_significance(all_runs)
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
