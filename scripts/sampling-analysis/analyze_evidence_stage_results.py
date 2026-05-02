#!/usr/bin/env python3
"""
Post-process stress sweep outputs into paper-ready evidence artifacts.

Inputs expected under --analysis-dir:
- stress_ab_all_runs.csv
- stress_dataset_manifest.csv
- sweep_run_meta.json

Outputs:
- evidence/paired_deltas_all.csv
- evidence/scenario_metric_significance.csv
- evidence/successful_scenarios.csv
- evidence/evidence_report.md
- evidence/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = [
    ("mean_sentinel_quality", "higher"),
    ("mean_plan_quality", "higher"),
    ("total_time_s", "lower"),
    ("total_cost", "lower"),
]


def _two_sided_binom_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * p)


def _paired_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    piv = df.pivot_table(
        index=["dataset_id", "exp_seed", "sample_budget"],
        columns="mode",
        values=metric_col,
        aggfunc="mean",
    )
    if "random" not in piv.columns or "stratified" not in piv.columns:
        return pd.DataFrame()
    out = piv[["random", "stratified"]].dropna().copy()
    out["delta"] = out["stratified"] - out["random"]
    return out.reset_index()


def _per_scenario_significance(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset_id, ds in all_runs.groupby("dataset_id"):
        for metric, direction in METRIC_SPECS:
            pt = _paired_metric(ds, metric)
            if pt.empty:
                continue
            delta = pt["delta"]
            if direction == "higher":
                wins = int((delta > 0).sum())
                losses = int((delta < 0).sum())
            else:
                wins = int((delta < 0).sum())
                losses = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            pval = _two_sided_binom_p(wins, losses)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "metric": metric,
                    "direction_better_for_stratified": direction,
                    "n_pairs": int(len(delta)),
                    "stratified_wins": wins,
                    "random_wins": losses,
                    "ties": ties,
                    "win_rate_excluding_ties": float(wins / (wins + losses)) if (wins + losses) > 0 else float("nan"),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
                    "sign_test_p_value_two_sided": float(pval),
                    "is_significant_p_lt_0_05": bool((pval < 0.05) if not pd.isna(pval) else False),
                    "supports_stratified": bool(
                        (wins > losses) and ((pval < 0.05) if not pd.isna(pval) else False)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _all_paired_deltas(all_runs: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for metric, direction in METRIC_SPECS:
        pt = _paired_metric(all_runs, metric)
        if pt.empty:
            continue
        p = pt.rename(columns={"delta": "delta_value"}).copy()
        p["metric"] = metric
        p["direction_better_for_stratified"] = direction
        pieces.append(p)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _save_global_delta_plot(paired: pd.DataFrame, fig_dir: Path) -> None:
    if paired.empty:
        return
    order = [m for m, _ in METRIC_SPECS]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.ravel(), order, strict=False):
        sub = paired[paired["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.hist(sub["delta_value"], bins=20, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("Delta (stratified - random)")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "global_delta_histograms.png", dpi=180)
    plt.close(fig)


def _save_scenario_plots(all_runs: pd.DataFrame, successful_ids: set[str], fig_dir: Path) -> None:
    metrics_to_plot = [m for m, _ in METRIC_SPECS]
    for dataset_id in sorted(successful_ids):
        ds = all_runs[all_runs["dataset_id"] == dataset_id].copy()
        if ds.empty:
            continue
        budgets = sorted(ds["sample_budget"].dropna().unique())

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, metric in zip(axes.ravel(), metrics_to_plot, strict=False):
            sub = ds.groupby(["sample_budget", "mode"], as_index=False)[metric].mean()
            r = sub[sub["mode"] == "random"].set_index("sample_budget")
            s = sub[sub["mode"] == "stratified"].set_index("sample_budget")
            ry = [float(r.loc[b, metric]) if b in r.index else float("nan") for b in budgets]
            sy = [float(s.loc[b, metric]) if b in s.index else float("nan") for b in budgets]
            ax.plot(budgets, ry, marker="o", label="random")
            ax.plot(budgets, sy, marker="o", label="stratified")
            ax.set_title(metric)
            ax.set_xlabel("Sample budget")
            ax.grid(True, alpha=0.3)
            if metric in {"total_time_s", "total_cost"}:
                ax.set_ylabel("Lower is better")
            else:
                ax.set_ylabel("Higher is better")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{dataset_id}: random vs stratified by budget", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / f"{dataset_id}_metric_curves.png", dpi=180)
        plt.close(fig)


def _write_report(
    out_md: Path,
    analysis_dir: Path,
    meta: dict,
    manifest: pd.DataFrame,
    significance: pd.DataFrame,
    successful: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Evidence-stage report")
    lines.append("")
    lines.append(f"- Analysis directory: `{analysis_dir}`")
    lines.append(f"- Cases manifested: {int(meta.get('cases_manifested', 0))}")
    lines.append(f"- A/B runs completed: {int(meta.get('ab_runs_completed', 0))}")
    lines.append(f"- Budgets: {meta.get('ab_settings', {}).get('budgets', [])}")
    lines.append(f"- Seeds: {meta.get('ab_settings', {}).get('exp_seeds', [])}")
    lines.append("")
    lines.append("## Scenario manifest")
    for _, row in manifest.sort_values("dataset_id").iterrows():
        lines.append(
            f"- `{row['dataset_id']}` | preset={row['preset']} | n_docs={int(row['n_docs'])} | notes={row['notes']}"
        )
    lines.append("")
    lines.append("## Significant scenario-level wins for stratified")
    if successful.empty:
        lines.append("- None at p < 0.05 under current run.")
    else:
        for _, row in successful.sort_values(["metric", "dataset_id"]).iterrows():
            lines.append(
                "- `{dataset}` metric=`{metric}` wins={wins} losses={losses} ties={ties} "
                "mean_delta={delta:.6f} p={p:.6g}".format(
                    dataset=row["dataset_id"],
                    metric=row["metric"],
                    wins=int(row["stratified_wins"]),
                    losses=int(row["random_wins"]),
                    ties=int(row["ties"]),
                    delta=float(row["mean_delta"]),
                    p=float(row["sign_test_p_value_two_sided"]),
                )
            )
    lines.append("")
    lines.append("## All scenario-level significance tables")
    for dataset_id, sub in significance.sort_values(["dataset_id", "metric"]).groupby("dataset_id"):
        lines.append(f"### {dataset_id}")
        for _, row in sub.iterrows():
            lines.append(
                "- metric=`{metric}` supports_stratified={support} wins={wins} losses={losses} "
                "ties={ties} p={p:.6g} mean_delta={delta:.6f}".format(
                    metric=row["metric"],
                    support=bool(row["supports_stratified"]),
                    wins=int(row["stratified_wins"]),
                    losses=int(row["random_wins"]),
                    ties=int(row["ties"]),
                    p=float(row["sign_test_p_value_two_sided"]),
                    delta=float(row["mean_delta"]),
                )
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    runs_csv = analysis_dir / "stress_ab_all_runs.csv"
    manifest_csv = analysis_dir / "stress_dataset_manifest.csv"
    meta_json = analysis_dir / "sweep_run_meta.json"

    if not runs_csv.is_file():
        raise FileNotFoundError(f"Missing required file: {runs_csv}")
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Missing required file: {manifest_csv}")
    if not meta_json.is_file():
        raise FileNotFoundError(f"Missing required file: {meta_json}")

    all_runs = pd.read_csv(runs_csv)
    manifest = pd.read_csv(manifest_csv)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))

    evidence_dir = analysis_dir / "evidence"
    fig_dir = evidence_dir / "figures"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paired = _all_paired_deltas(all_runs)
    paired.to_csv(evidence_dir / "paired_deltas_all.csv", index=False)

    significance = _per_scenario_significance(all_runs)
    significance.to_csv(evidence_dir / "scenario_metric_significance.csv", index=False)

    successful = significance[significance["supports_stratified"]].copy()
    successful.to_csv(evidence_dir / "successful_scenarios.csv", index=False)

    _save_global_delta_plot(paired, fig_dir)
    _save_scenario_plots(all_runs, set(successful["dataset_id"].tolist()), fig_dir)

    _write_report(
        evidence_dir / "evidence_report.md",
        analysis_dir,
        meta,
        manifest,
        significance,
        successful,
    )
    print(f"Wrote evidence artifacts under {evidence_dir}")


if __name__ == "__main__":
    main()
