#!/usr/bin/env python3
"""Generate slide-ready summary figures for paper and presentation use.

Inputs:
1) feature ablation summary table (CSV)
2) stress-significance summary table (CSV)

Outputs:
- PNG and PDF versions of each figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURE_RENAME = {
    "Feature axis": "feature_axis",
    "Mean Δ optimization_cost": "d_opt_cost",
    "Mean Δ total_cost": "d_total_cost",
    "Mean Δ quality_error": "d_quality_error",
}

STRESS_METRIC_ORDER = [
    "mean_sentinel_quality",
    "mean_plan_quality",
    "total_time_s",
    "total_cost",
]

STRESS_METRIC_LABELS = {
    "mean_sentinel_quality": "Sentinel quality\n(higher better)",
    "mean_plan_quality": "Plan quality\n(higher better)",
    "total_time_s": "Total time\n(lower better)",
    "total_cost": "Total cost\n(lower better)",
}


def _validate_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def _save_png_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _friendly_feature_name(name: str) -> str:
    return name.replace("_", " ")


def plot_feature_ablation_diverging(feature_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(feature_csv)
    df = df.rename(columns=FEATURE_RENAME)
    _validate_columns(
        df,
        required={"feature_axis", "d_opt_cost", "d_total_cost", "d_quality_error"},
        source=feature_csv,
    )

    # Sort by total-cost effect so best/worst are immediately visible.
    df = df.sort_values("d_total_cost", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    bar_h = 0.22

    fig, ax = plt.subplots(figsize=(10.2, 5.3))

    ax.barh(
        y - bar_h,
        df["d_opt_cost"],
        height=bar_h,
        color="#1f77b4",
        label="Δ optimization cost",
    )
    ax.barh(
        y,
        df["d_total_cost"],
        height=bar_h,
        color="#2ca02c",
        label="Δ total cost",
    )
    ax.barh(
        y + bar_h,
        df["d_quality_error"],
        height=bar_h,
        color="#9467bd",
        label="Δ quality error",
    )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_friendly_feature_name(x) for x in df["feature_axis"]], fontsize=10)
    ax.set_xlabel("Paired mean delta (stratified - random)")
    ax.set_title("Feature Ablation: Mean Paired Deltas by Feature Axis")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()

    _save_png_pdf(fig, out_dir, "feature_ablation_summary_bars")
    plt.close(fig)


def plot_stress_significance_stacked(stress_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(stress_csv)
    _validate_columns(
        df,
        required={"metric", "n_pairs", "stratified_wins", "random_wins", "ties"},
        source=stress_csv,
    )

    df = df[df["metric"].isin(STRESS_METRIC_ORDER)].copy()
    df["metric"] = pd.Categorical(df["metric"], categories=STRESS_METRIC_ORDER, ordered=True)
    df = df.sort_values("metric")

    labels = [STRESS_METRIC_LABELS[m] for m in df["metric"].tolist()]
    wins = df["stratified_wins"].astype(int).to_numpy()
    losses = df["random_wins"].astype(int).to_numpy()
    ties = df["ties"].astype(int).to_numpy()
    n_pairs = int(df["n_pairs"].max()) if len(df) else 0

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    ax.bar(x, wins, color="#2ca02c", label="Stratified wins")
    ax.bar(x, losses, bottom=wins, color="#d62728", label="Random wins")
    ax.bar(x, ties, bottom=wins + losses, color="#7f7f7f", label="Ties")

    for i, (w, l, t) in enumerate(zip(wins, losses, ties)):
        if t > 0:
            ax.text(i, w + l + t - 0.75, f"{t} ties", ha="center", va="top", fontsize=9, color="white")
        else:
            ax.text(i, w + l + 0.4, f"{w}/{l}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Paired comparisons (n={n_pairs})")
    ax.set_title("Stress Sweeps: Win/Loss/Tie Counts by Metric")
    ax.set_ylim(0, max(1, n_pairs + 1))
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()

    _save_png_pdf(fig, out_dir, "stress_significance_stacked_wlt")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-ready summary figures.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/feature_ablation/feature_ablation_4_28_table.csv"
        ),
    )
    parser.add_argument(
        "--stress-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/stress_significance_summary.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sampling-project/results/sampling-results/stress_test_dataset_analysis/figures"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_ablation_diverging(args.feature_csv, args.out_dir)
    plot_stress_significance_stacked(args.stress_csv, args.out_dir)

    print("Wrote presentation figures:")
    print(args.out_dir / "feature_ablation_summary_bars.png")
    print(args.out_dir / "feature_ablation_summary_bars.pdf")
    print(args.out_dir / "stress_significance_stacked_wlt.png")
    print(args.out_dir / "stress_significance_stacked_wlt.pdf")


if __name__ == "__main__":
    main()
