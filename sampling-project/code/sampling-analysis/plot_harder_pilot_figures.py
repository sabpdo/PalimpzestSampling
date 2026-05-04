#!/usr/bin/env python3
"""
Plot Experiment 3 (harder-task pilot) figures from paired deltas.

Reads:
  - paired_deltas_running.csv (columns seed,budget,dq,dplanq,dtime,dcost)

Writes PNGs under --out-dir (default: signif_search/figures/):
  - harder_pilot_win_loss.png   — stacked wins / losses / ties per metric
  - harder_pilot_paired_deltas.png — paired Δcost and Δtime (stratified − random)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_harder_pilot_significance import summarize_significance


def _plot_win_loss_stacked(sig: pd.DataFrame, out: Path) -> None:
    labels = []
    wins = []
    losses = []
    ties = []
    display = {
        "dcost": "Total cost\n(Δ<0 = stratified cheaper)",
        "dtime": "Wall time\n(Δ<0 = stratified faster)",
        "dq": "Sentinel quality\n(Δ>0 = stratified higher)",
        "dplanq": "Plan quality\n(Δ>0 = stratified higher)",
    }
    for _, row in sig.iterrows():
        m = str(row["metric"])
        labels.append(display.get(m, m))
        wins.append(int(row["wins"]))
        losses.append(int(row["losses"]))
        ties.append(int(row["ties"]))

    x = np.arange(len(labels))
    width = 0.55
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, wins, width, label="Stratified better", color="#2ca02c")
    ax.bar(x, losses, width, bottom=wins, label="Random better", color="#d62728")
    ax.bar(x, ties, width, bottom=np.array(wins) + np.array(losses), label="Tie", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Paired comparisons (n=20)")
    ax.set_title("Experiment 3: harder-task pilot — sign-test counts per metric")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 22)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_paired_deltas(paired: pd.DataFrame, out: Path) -> None:
    paired = paired.copy()
    paired["pair_label"] = paired.apply(lambda r: f"s{int(r['seed'])}/b{int(r['budget'])}", axis=1)
    paired = paired.sort_values(["seed", "budget"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ycols = [("dcost", "Δ Total cost (USD)\n(stratified − random)"), ("dtime", "Δ Wall time (s)\n(stratified − random)")]
    colors_cost = np.where(paired["dcost"] < 0, "#2ca02c", "#d62728")
    colors_time = np.where(paired["dtime"] < 0, "#2ca02c", "#d62728")

    x = np.arange(len(paired))
    for ax, (col, ylab), colors in zip(axes, ycols, [colors_cost, colors_time]):
        vals = paired[col].to_numpy()
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.scatter(x, vals, c=colors, s=42, zorder=3, edgecolors="white", linewidths=0.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(paired["pair_label"].tolist(), rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Paired deltas by (seed, budget)")
    fig.suptitle("Experiment 3: harder-task pilot", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-deltas-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/"
            "signif_search/paired_deltas_running.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/figures"
        ),
    )
    args = parser.parse_args()

    paired = pd.read_csv(args.paired_deltas_csv)
    sig = summarize_significance(paired)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = args.out_dir / "harder_pilot_win_loss.png"
    p2 = args.out_dir / "harder_pilot_paired_deltas.png"
    _plot_win_loss_stacked(sig, p1)
    _plot_paired_deltas(paired, p2)
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
