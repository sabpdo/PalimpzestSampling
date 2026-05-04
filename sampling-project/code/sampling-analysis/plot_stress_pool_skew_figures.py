#!/usr/bin/env python3
"""
Stress evaluation pools: visualize induced heterogeneity / skew (Experiment 2 setup).

Reads each pool's paper_features.csv under a pools root directory, optionally enriches
titles from stress_dataset_manifest.csv (notes + preset).

Outputs (under --out-dir, default stress_test_dataset_analysis/figures/):
  - stress_pools_domain_share.png — 100% stacked bars of domain counts per pool
  - stress_pools_word_count_hist.png — per-pool histograms of word_count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOMAIN_ORDER = ("bio_medical", "cs", "math", "physics")
DOMAIN_COLORS = {
    "bio_medical": "#1f77b4",
    "cs": "#ff7f0e",
    "math": "#2ca02c",
    "physics": "#9467bd",
}


def _load_manifest(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _pool_sort_key(name: str) -> tuple[int, str]:
    """Order: rare_numeric_tail by increasing tail fraction hint, then other presets."""
    if name.startswith("rnt_tail"):
        try:
            pct = int(name.replace("rnt_tail", ""))
        except ValueError:
            pct = 999
        return (0, f"{pct:03d}_{name}")
    return (1, name)


def _discover_pools(pools_root: Path) -> list[Path]:
    out: list[Path] = []
    if not pools_root.is_dir():
        return out
    for child in sorted(pools_root.iterdir(), key=lambda p: _pool_sort_key(p.name)):
        if (child / "paper_features.csv").is_file():
            out.append(child)
    return out


def _manifest_row(manifest: pd.DataFrame, dataset_id: str) -> pd.Series | None:
    if manifest.empty or "dataset_id" not in manifest.columns:
        return None
    sub = manifest[manifest["dataset_id"] == dataset_id]
    if sub.empty:
        return None
    return sub.iloc[0]


def _plot_domain_stacked(pool_dirs: list[Path], manifest: pd.DataFrame, out: Path) -> None:
    pool_ids = [p.name for p in pool_dirs]
    counts = {d: [] for d in DOMAIN_ORDER}
    subtitles: list[str] = []

    for p in pool_dirs:
        df = pd.read_csv(p / "paper_features.csv")
        vc = df["domain"].value_counts()
        for d in DOMAIN_ORDER:
            counts[d].append(float(vc.get(d, 0)))
        row = _manifest_row(manifest, p.name)
        if row is not None:
            preset = row.get("preset", "")
            note = str(row.get("notes", ""))[:52]
            subtitles.append(f"{preset}\n{note}")
        else:
            subtitles.append(p.name)

    x = np.arange(len(pool_ids))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bottom = np.zeros(len(pool_ids))
    totals = np.array([sum(counts[d][i] for d in DOMAIN_ORDER) for i in range(len(pool_ids))])

    for d in DOMAIN_ORDER:
        h = np.array(counts[d])
        frac = np.divide(h, totals, out=np.zeros_like(h), where=totals > 0)
        ax.bar(x, frac, bottom=bottom, label=d.replace("_", " "), color=DOMAIN_COLORS.get(d, "#555555"), width=0.62)
        bottom = bottom + frac

    ax.set_xticks(x)
    ax.set_xticklabels([f"{pid}\n{subtitles[i]}" for i, pid in enumerate(pool_ids)], fontsize=7)
    ax.set_ylabel("Share of documents")
    ax.set_ylim(0, 1.0)
    ax.set_title("Stress pools: domain composition (synthetic feature tables over shared PDFs)")
    ax.legend(title="domain", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_word_count_hists(pool_dirs: list[Path], manifest: pd.DataFrame, out: Path) -> None:
    n = len(pool_dirs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, max(3.5, 2.8 * nrows)))
    axes_arr = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.set_visible(False)
            continue
        p = pool_dirs[i]
        df = pd.read_csv(p / "paper_features.csv")
        wc = pd.to_numeric(df["word_count"], errors="coerce").dropna()
        ax.hist(
            wc,
            bins=14,
            color="#4c72b0",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.88,
        )
        row = _manifest_row(manifest, p.name)
        title = p.name
        if row is not None:
            title = f"{p.name}\n({row.get('preset', '')})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("word_count")
        ax.set_ylabel("documents")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Stress pools: word-count distribution (induced structural skew / long tails)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pools-root",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/pools"
        ),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(
            "sampling-project/results/sampling-results/stress_test_dataset_analysis/stress_dataset_manifest.csv"
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

    manifest = _load_manifest(args.manifest_csv)
    pool_dirs = _discover_pools(args.pools_root)
    if not pool_dirs:
        raise SystemExit(f"No pools with paper_features.csv under {args.pools_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = args.out_dir / "stress_pools_domain_share.png"
    p2 = args.out_dir / "stress_pools_word_count_hist.png"
    _plot_domain_stacked(pool_dirs, manifest, p1)
    _plot_word_count_hists(pool_dirs, manifest, p2)
    print(f"Pools: {[p.name for p in pool_dirs]}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
