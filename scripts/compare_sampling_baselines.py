#!/usr/bin/env python3
"""
Compare Palimpzest sentinel **baseline** row order (seeded shuffle) vs **feature stratified** order.

The stratifier uses the same feature columns as ``scripts/extract_features.py`` (composite
percentile strata, shuffle within stratum, round-robin across strata). The feature table
defaults to ``papers/paper_features.csv`` under the repo root.

**Baseline** is ``sample_documents(..., method="random")``. **Stratified** is
``stratified_source_keys`` / ``sample_documents(..., method="stratified")``.

Metrics are **stratum balance** in the early prefix only: ``mix_L1`` is the L1 distance
of observed stratum counts from uniform ``prefix_n / k`` (lower is more representative).

Run from repo root::

    uv run python scripts/compare_sampling_baselines.py
    uv run python scripts/compare_sampling_baselines.py --plot sampling_comparison.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"


def _load_execution_sampling_modules():
    """
    Load ``document_sampling`` + ``feature_stratified_sampling`` without importing
    ``palimpzest`` package ``__init__`` (avoids optional deps like ``smolagents``).
    """
    if "palimpzest.query.execution.document_sampling" in sys.modules:
        return sys.modules["palimpzest.query.execution.document_sampling"]

    pkg_chain = ("palimpzest", "palimpzest.query", "palimpzest.query.execution")
    for name in pkg_chain:
        if name not in sys.modules:
            m = types.ModuleType(name)
            if name == "palimpzest.query.execution":
                m.__path__ = [str(_SRC / "palimpzest/query/execution")]
            sys.modules[name] = m

    parent = sys.modules["palimpzest.query.execution"]

    def _exec_submodule(full_name: str, relative_path: Path):
        path = _SRC / relative_path
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {full_name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "palimpzest.query.execution"
        sys.modules[full_name] = mod
        setattr(parent, full_name.rpartition(".")[2], mod)
        spec.loader.exec_module(mod)
        return mod

    _exec_submodule(
        "palimpzest.query.execution.feature_stratified_sampling",
        Path("palimpzest/query/execution/feature_stratified_sampling.py"),
    )
    return _exec_submodule(
        "palimpzest.query.execution.document_sampling",
        Path("palimpzest/query/execution/document_sampling.py"),
    )


_doc_sampling = _load_execution_sampling_modules()
sample_documents = _doc_sampling.sample_documents
stratified_source_keys = _doc_sampling.stratified_source_keys

_fss = sys.modules["palimpzest.query.execution.feature_stratified_sampling"]
FEATURE_COLUMNS = _fss.FEATURE_COLUMNS
composite_strata = _fss.composite_strata
read_feature_table = _fss.read_feature_table


def _index_from_key(key: str) -> int:
    return int(key.split("---")[-1])


def prefix_balance(
    keys: list[str],
    prefix_n: int,
    *,
    stratum_per_index: np.ndarray,
) -> dict:
    take = keys[:prefix_n]
    idxs = [_index_from_key(k) for k in take]

    k = int(stratum_per_index.max()) + 1 if len(stratum_per_index) else 0
    counts = np.zeros(max(k, 1), dtype=np.int64)
    for i in idxs:
        counts[int(stratum_per_index[i])] += 1
    expected = prefix_n / max(k, 1)
    mix_l1 = float(np.sum(np.abs(counts[:k] - expected))) if k else 0.0

    return {"mix_l1": mix_l1, "counts": counts[:k]}


def worst_baseline_mix_seed(
    *,
    dataset_id: str,
    num_records: int,
    prefix_n: int,
    stratum_per_index: np.ndarray,
    seed_lo: int,
    seed_hi: int,
) -> tuple[int, dict]:
    """Among baseline shuffles, pick the seed with largest prefix ``mix_L1`` (least balanced)."""
    worst_seed, worst_m = seed_lo, {}
    worst_mix = -1.0
    for seed in range(seed_lo, seed_hi + 1):
        rng = np.random.default_rng(seed)
        keys = sample_documents(dataset_id, num_records, rng, method="random")
        m = prefix_balance(keys, prefix_n, stratum_per_index=stratum_per_index)
        if m["mix_l1"] > worst_mix:
            worst_mix, worst_seed, worst_m = m["mix_l1"], seed, m
    return worst_seed, worst_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Palimpzest baseline shuffle vs feature-stratified sentinel order (stratum balance)."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=_REPO_ROOT / "papers" / "paper_features.csv",
        help="Feature table from extract_features.py --scan",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Use at most this many rows (after sorting by row_index); default = all rows",
    )
    parser.add_argument("--prefix", type=int, default=40, help="Prefix length = early sentinel evaluations")
    parser.add_argument(
        "--strata",
        type=int,
        default=8,
        help="Number of strata (passed to composite_strata / stratified_source_keys)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="ds",
        help="Dataset id embedded in sentinel keys (must match evaluation setup)",
    )
    parser.add_argument(
        "--baseline-seed-scan",
        type=int,
        default=500,
        help="Scan shuffle seeds 0..N for baseline run with worst (highest) mix_L1",
    )
    parser.add_argument(
        "--median-baseline",
        action="store_true",
        dest="median_baseline",
        help="Report median mix_L1 over scanned baseline shuffle seeds",
    )
    parser.add_argument("--plot", type=str, default=None, help="Save bar chart path (needs matplotlib)")
    args = parser.parse_args()

    df = read_feature_table(args.features_csv.resolve(), max_rows=args.max_records)
    num_records = len(df)
    if num_records == 0:
        print("No rows in feature table.", file=sys.stderr)
        sys.exit(1)

    prefix_n = min(args.prefix, num_records)
    num_strata = max(1, args.strata)
    mat = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    stratum_per_index = composite_strata(mat, num_strata)

    rng_strat = np.random.default_rng(42)
    keys_strat = stratified_source_keys(
        args.dataset_id,
        num_records,
        rng_strat,
        num_strata=num_strata,
        features_csv=args.features_csv.resolve(),
    )
    m_strat = prefix_balance(keys_strat, prefix_n, stratum_per_index=stratum_per_index)

    worst_seed, m_worst = worst_baseline_mix_seed(
        dataset_id=args.dataset_id,
        num_records=num_records,
        prefix_n=prefix_n,
        stratum_per_index=stratum_per_index,
        seed_lo=0,
        seed_hi=args.baseline_seed_scan,
    )

    rng_example = np.random.default_rng(12345)
    keys_example = sample_documents(args.dataset_id, num_records, rng_example, method="random")
    m_example = prefix_balance(keys_example, prefix_n, stratum_per_index=stratum_per_index)

    baseline_mix_l1: list[float] = []
    for seed in range(0, args.baseline_seed_scan + 1):
        rng = np.random.default_rng(seed)
        keys = sample_documents(args.dataset_id, num_records, rng, method="random")
        m = prefix_balance(keys, prefix_n, stratum_per_index=stratum_per_index)
        baseline_mix_l1.append(m["mix_l1"])
    median_mix = float(np.median(baseline_mix_l1))

    table = PrettyTable()
    table.field_names = ["Ordering", "Detail", f"mix_L1@{prefix_n}"]
    table.align = "r"
    table.align["Ordering"] = "l"
    table.align["Detail"] = "l"

    def row(ordering: str, detail: str, m: dict) -> None:
        table.add_row([ordering, detail, f"{m['mix_l1']:.1f}"])

    row(
        "Stratified (features)",
        f"k={num_strata}, rng=42, csv={args.features_csv.name}",
        m_strat,
    )
    row("PZ baseline: shuffle", "example shuffle seed=12345", m_example)
    row(
        "PZ baseline: shuffle",
        f"worst mix_L1 among seeds 0..{args.baseline_seed_scan} (seed={worst_seed})",
        m_worst,
    )
    if args.median_baseline:
        table.add_row(
            [
                "PZ baseline: shuffle",
                f"median mix_L1 over seeds 0..{args.baseline_seed_scan}",
                f"{median_mix:.1f}",
            ],
        )

    print()
    print(
        "Baseline = ``sample_documents(..., method='random')``. "
        "Stratified = ``stratified_source_keys`` on extract_features columns."
    )
    print(
        f"Stratum balance: composite feature strata, k={num_strata}. "
        f"mix_L1 = L1 distance of prefix stratum counts from uniform."
    )
    print(f"Rows: {num_records}; early prefix = first {prefix_n} keys. CSV: {args.features_csv.resolve()}")
    print()
    print(table)
    print()
    print("Lower mix_L1 is closer to proportional representation across strata in the prefix.")
    if args.median_baseline:
        print(
            f"Median mix_L1 for baseline shuffle over {args.baseline_seed_scan + 1} seeds: {median_mix:.1f}"
        )

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip --plot.", file=sys.stderr)
            return

        labels = [
            "Stratified\n(features)",
            "PZ baseline\n(example)",
            "PZ baseline\n(worst seed)",
        ]
        mix_values = [m_strat["mix_l1"], m_example["mix_l1"], m_worst["mix_l1"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, mix_values, color=["#2a9d8f", "#457b9d", "#e76f51"])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("mix_L1 vs uniform stratum counts")
        ax.set_title("Early prefix balance: baseline shuffle vs feature stratified")
        fig.tight_layout()
        out = Path(args.plot)
        fig.savefig(out, dpi=150)
        print(f"Wrote plot: {out.resolve()}")


if __name__ == "__main__":
    main()
