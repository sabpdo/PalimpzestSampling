#!/usr/bin/env python3
"""
Rank single-feature stratification ablations by paired random vs stratified deltas.

Place exported ab_results CSVs in a folder (one file per stratify_features axis), e.g.:

  papers/feature_ablation/domain.csv
  papers/feature_ablation/word_count.csv
  ...

Or pass paths explicitly:

  python scripts/compare_stratify_feature_runs.py \\
    papers/feature_ablation/domain.csv papers/feature_ablation/word_count.csv

Requires each CSV to have columns: sample_budget, mode, and metric columns from run_sentinel_sampling_ab.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def _label_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("ab_results"):
        parent = path.parent.name
        return parent if parent and parent != "papers" else stem
    return stem


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    need = {"sample_budget", "mode"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns {need}; got {list(df.columns)}")
    rows = []
    for b, g in df.groupby("sample_budget"):
        r = g[g["mode"].eq("random")]
        s = g[g["mode"].eq("stratified")]
        if len(r) != 1 or len(s) != 1:
            continue
        r, s = r.iloc[0], s.iloc[0]
        row = {"sample_budget": int(b)}
        for col in df.columns:
            if col in ("sample_budget", "mode"):
                continue
            try:
                rv, sv = float(r[col]), float(s[col])
            except (TypeError, ValueError):
                continue
            row[f"d_{col}"] = sv - rv
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csv_paths",
        type=Path,
        nargs="*",
        help="ab_results CSV files (one per stratify_features ablation).",
    )
    p.add_argument(
        "--glob-dir",
        type=Path,
        default=None,
        help="If set, include every *.csv under this directory (non-recursive).",
    )
    args = p.parse_args()
    paths: list[Path] = list(args.csv_paths)
    if args.glob_dir:
        d = args.glob_dir.resolve()
        if d.is_dir():
            paths.extend(sorted(d.glob("*.csv")))
    paths = [x.resolve() for x in paths if x.is_file()]
    if not paths:
        print("No CSV files found. Pass paths or --glob-dir.", file=sys.stderr)
        sys.exit(2)

    metrics_priority = [
        "mean_plan_quality",
        "mean_sentinel_quality",
        "total_cost",
        "total_time_s",
        "optimization_cost",
        "plan_execution_cost",
    ]

    summaries: list[dict] = []
    for path in paths:
        df = pd.read_csv(path)
        feat = None
        if "stratify_features" in df.columns:
            feat = str(df["stratify_features"].iloc[0])
        if not feat:
            feat = _label_from_path(path)
        deltas = paired_deltas(df)
        if deltas.empty:
            print(f"warn: no paired rows in {path}", file=sys.stderr)
            continue
        row: dict = {"feature": feat, "file": str(path)}
        for m in metrics_priority:
            col = f"d_{m}"
            if col not in deltas.columns:
                continue
            s = deltas[col].dropna()
            if s.empty:
                continue
            row[f"mean_{m}_delta"] = float(s.mean())
            row[f"sum_abs_{m}_delta"] = float(s.abs().sum())
        summaries.append(row)

    if not summaries:
        print("No valid paired summaries.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(summaries)

    print("=== Mean paired delta (stratified - random), averaged over budgets ===\n")
    for m in metrics_priority:
        col = f"mean_{m}_delta"
        if col not in out.columns:
            continue
        sub = out[["feature", col, "file"]].dropna(subset=[col]).copy()
        if sub.empty:
            continue
        # Quality: higher better → largest positive mean Δ first.
        # Cost/time: lower better → stratified wins when Δ is negative → most negative mean Δ first.
        ascending = m in ("total_cost", "total_time_s", "optimization_cost", "plan_execution_cost", "plan_execution_time_s", "optimization_time_s")
        sub = sub.sort_values(col, ascending=ascending)
        print(f"-- {m} (sorted: quality=max first, cost/time=most negative first) --")
        print(sub.to_string(index=False))
        print()

    out_path = Path("results/stratify_feature_ablation_ranking.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
