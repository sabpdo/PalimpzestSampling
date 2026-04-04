#!/usr/bin/env python3
"""
Compare Palimpzest's *existing* sentinel document order vs a *proposed* stratified order.

**Baseline** is not "random" in the abstract—it is the **current Palimpzest behavior**:
root-dataset rows are permuted with a seeded shuffle before ``OpFrontier`` walks them
(see ``sample_documents(..., method="random")`` / MAB ``execute_sentinel_plan`` shuffle).

**Proposed** row: uses ``stratified_source_keys`` from Palimpzest once your partner implements it.
Until then, this script falls back to a **local demo** round-robin ordering (same seed) so you
can still print/plot comparisons.

The rest is a small synthetic world (two quality regions by row parity) so early-prefix
quality estimates can be compared without running LLMs.

Run from repo root::

    uv run python scripts/compare_sampling_baselines.py
    uv run python scripts/compare_sampling_baselines.py --plot sampling_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from collections import deque

from palimpzest.query.execution.document_sampling import sample_documents, stratified_source_keys


def _index_from_key(key: str) -> int:
    return int(key.split("---")[-1])


def true_quality(idx: int) -> float:
    """Two-region corpus: evens low, odds high (population mean 0.5 for even N)."""
    return 0.2 if idx % 2 == 0 else 0.8


def population_mean_quality(num_records: int) -> float:
    return float(np.mean([true_quality(i) for i in range(num_records)]))


def prefix_metrics(keys: list[str], prefix_n: int, num_strata: int) -> dict:
    take = keys[:prefix_n]
    idxs = [_index_from_key(k) for k in take]
    q = [true_quality(i) for i in idxs]
    mean_q = float(np.mean(q))

    counts = np.zeros(num_strata, dtype=np.int64)
    for i in idxs:
        counts[i % num_strata] += 1
    expected = prefix_n / num_strata
    mix_l1 = float(np.sum(np.abs(counts - expected)))

    return {"mean_quality": mean_q, "mix_l1": mix_l1, "counts": counts}


def _script_demo_stratified_keys(
    dataset_id: str,
    num_records: int,
    rng: np.random.Generator,
    *,
    num_strata: int,
) -> list[str]:
    """
    Stand-in until ``stratified_source_keys`` is implemented in Palimpzest.

    Index-mod buckets, shuffle within bucket, round-robin merge — remove from this script
    once the real implementation lives in document_sampling.py.
    """
    if num_records == 0:
        return []
    k = max(1, min(num_strata, num_records))
    buckets: list[list[int]] = [[] for _ in range(k)]
    for i in range(num_records):
        buckets[i % k].append(i)
    for b in buckets:
        rng.shuffle(b)
    queues = [deque(b) for b in buckets]
    order: list[int] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.popleft())
    return [f"{dataset_id}---{int(idx)}" for idx in order]


def proposed_keys_or_demo(
    dataset_id: str,
    num_records: int,
    rng: np.random.Generator,
    *,
    num_strata: int,
) -> tuple[list[str], str]:
    """Try real hook; on NotImplementedError use script-local demo. Returns (keys, detail note)."""
    try:
        keys = stratified_source_keys(dataset_id, num_records, rng, num_strata=num_strata)
        return keys, "from stratified_source_keys()"
    except NotImplementedError:
        keys = _script_demo_stratified_keys(dataset_id, num_records, rng, num_strata=num_strata)
        return keys, "script demo (implement stratified_source_keys in PZ)"


def worst_baseline_seed(
    *,
    num_records: int,
    prefix_n: int,
    num_strata: int,
    seed_lo: int,
    seed_hi: int,
    target_mean: float,
) -> tuple[int, dict]:
    """Among Palimpzest baseline shuffles, pick the seed whose prefix is most biased vs population mean."""
    worst_seed, worst_m = seed_lo, {}
    worst_bias = -1.0
    for seed in range(seed_lo, seed_hi + 1):
        rng = np.random.default_rng(seed)
        keys = sample_documents("ds", num_records, rng, method="random")
        m = prefix_metrics(keys, prefix_n, num_strata)
        bias = abs(m["mean_quality"] - target_mean)
        if bias > worst_bias:
            worst_bias, worst_seed, worst_m = bias, seed, m
    return worst_seed, worst_m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Palimpzest baseline (shuffled row order) vs proposed stratified order."
    )
    parser.add_argument("--records", type=int, default=2000, help="Synthetic corpus size")
    parser.add_argument("--prefix", type=int, default=40, help="First N positions = early sentinel evaluations")
    parser.add_argument("--strata", type=int, default=2, help="Strata for proposed sampler (2 ≡ even/odd)")
    parser.add_argument(
        "--baseline-seed-scan",
        type=int,
        default=500,
        help="Scan shuffle seeds 0..N for worst baseline prefix bias",
    )
    parser.add_argument(
        "--also-median-baseline",
        action="store_true",
        help="Report median |bias| over scanned baseline shuffle seeds",
    )
    parser.add_argument("--plot", type=str, default=None, help="Save bar chart path (needs matplotlib)")
    args = parser.parse_args()

    num_records = args.records
    prefix_n = min(args.prefix, num_records)
    num_strata = max(2, args.strata)
    pop_mean = population_mean_quality(num_records)

    rng_strat = np.random.default_rng(42)
    keys_strat, proposed_detail = proposed_keys_or_demo(
        "ds", num_records, rng_strat, num_strata=num_strata
    )
    m_strat = prefix_metrics(keys_strat, prefix_n, num_strata)

    worst_seed, m_worst = worst_baseline_seed(
        num_records=num_records,
        prefix_n=prefix_n,
        num_strata=num_strata,
        seed_lo=0,
        seed_hi=args.baseline_seed_scan,
        target_mean=pop_mean,
    )

    rng_example = np.random.default_rng(12345)
    keys_example = sample_documents("ds", num_records, rng_example, method="random")
    m_example = prefix_metrics(keys_example, prefix_n, num_strata)

    baseline_biases = []
    for seed in range(0, args.baseline_seed_scan + 1):
        rng = np.random.default_rng(seed)
        keys = sample_documents("ds", num_records, rng, method="random")
        m = prefix_metrics(keys, prefix_n, num_strata)
        baseline_biases.append(abs(m["mean_quality"] - pop_mean))
    median_bias = float(np.median(baseline_biases))

    unit_cost = 0.001
    unit_latency_s = 0.05
    mock_cost = prefix_n * unit_cost
    mock_latency = prefix_n * unit_latency_s

    table = PrettyTable()
    table.field_names = [
        "Ordering",
        "Detail",
        f"mean_Q@{prefix_n}",
        "|bias| vs pop",
        "mix_L1 vs uniform",
        "mock $",
        "mock s",
    ]
    table.align = "r"
    table.align["Ordering"] = "l"
    table.align["Detail"] = "l"

    def row(ordering: str, detail: str, m: dict) -> None:
        bias = abs(m["mean_quality"] - pop_mean)
        table.add_row(
            [
                ordering,
                detail,
                f"{m['mean_quality']:.4f}",
                f"{bias:.4f}",
                f"{m['mix_l1']:.1f}",
                f"{mock_cost:.4f}",
                f"{mock_latency:.2f}",
            ],
        )

    row("Proposed: stratified", f"k={num_strata}, rng=42 — {proposed_detail}", m_strat)
    row("PZ baseline: shuffle", "example shuffle seed=12345", m_example)
    row(
        "PZ baseline: shuffle",
        f"worst of seeds 0..{args.baseline_seed_scan} (seed={worst_seed})",
        m_worst,
    )
    if args.also_median_baseline:
        table.add_row(
            [
                "PZ baseline: shuffle",
                f"median |bias| over seeds 0..{args.baseline_seed_scan}",
                "—",
                f"{median_bias:.4f}",
                "—",
                "—",
                "—",
            ],
        )

    print()
    print(
        "Palimpzest **baseline** = current sentinel behavior: full shuffle of row indices "
        "(``sample_documents(..., method='random')``). **Proposed** = ``stratified_source_keys`` (stub until your partner implements it)."
    )
    print(
        "Synthetic truth: true_quality(i) = 0.2 if i even else 0.8  →  population mean = {:.4f}".format(
            pop_mean
        )
    )
    print(f"Corpus: {num_records} rows; early prefix = first {prefix_n} evaluations.")
    print()
    print(table)
    print()
    print(
        "Early prefixes under the baseline shuffle can over/under-sample evens vs odds, so mean_Q "
        "drifts from the population mean. A good stratified implementation keeps early prefixes "
        "representative. Implement ``stratified_source_keys`` in document_sampling.py (or use "
        "``sample_documents(..., sampler=…)``)."
    )
    if args.also_median_baseline:
        print(
            f"Median |bias| for baseline shuffle over {args.baseline_seed_scan + 1} seeds: {median_bias:.4f}"
        )

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip --plot.", file=sys.stderr)
            return

        labels = [
            "Proposed\nstratified",
            "PZ baseline\n(example)",
            "PZ baseline\n(worst seed)",
        ]
        biases = [
            abs(m_strat["mean_quality"] - pop_mean),
            abs(m_example["mean_quality"] - pop_mean),
            abs(m_worst["mean_quality"] - pop_mean),
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, biases, color=["#2a9d8f", "#457b9d", "#e76f51"])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("|bias| in mean quality vs population")
        ax.set_title("Early prefix: PZ baseline shuffle vs proposed stratified")
        fig.tight_layout()
        out = Path(args.plot)
        fig.savefig(out, dpi=150)
        print(f"Wrote plot: {out.resolve()}")


if __name__ == "__main__":
    main()
