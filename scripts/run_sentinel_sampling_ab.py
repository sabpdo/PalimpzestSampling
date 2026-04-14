#!/usr/bin/env python3
"""
Run the same Palimpzest optimize-and-execute workload twice: sentinel **document order**
is either baseline shuffle (``random``) or **feature stratified** (``stratified``).

This measures **real** post-sample behavior: optimization cost/time, final plan cost/time,
totals, and mean record-level **quality** from sentinel ``RecordOpStats`` (when the
validator assigns scores).

**Alignment with ``paper_features.csv``:** Row indices in the CSV must match
``sorted(root.rglob('*.pdf'))`` — the same rule as ``scripts/extract_features.py --scan``.
Use the same ``--papers`` root you used to build the feature table. Stratified ordering
only affects the **training** corpus (first ``--train-n`` PDFs in that sort order).

Requires API keys and dependencies as for other Palimpzest demos (see README).

::

    python scripts/run_sentinel_sampling_ab.py --papers ./papers --train-n 40
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prettytable import PrettyTable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if _REPO_ROOT.is_dir():
    os.chdir(_REPO_ROOT)

import palimpzest as pz
from palimpzest.core.data.iter_dataset import IterDataset
from palimpzest.core.lib.schemas import PDFFile
from palimpzest.core.models import ExecutionStats
from palimpzest.tools.pdfparser import get_text_from_pdf


def iter_pdf_paths(papers_root: Path) -> list[Path]:
    """Same ordering as ``scripts/extract_features.iter_pdf_paths``."""
    if not papers_root.is_dir():
        return []
    return sorted(papers_root.rglob("*.pdf"))


class PDFPathsDataset(IterDataset):
    """Root dataset over an explicit list of PDF paths (recursive scan, stable order)."""

    def __init__(self, dataset_id: str, paths: list[Path]) -> None:
        super().__init__(id=dataset_id, schema=PDFFile)
        self.paths = [str(p) for p in paths]
        self.pdfprocessor = "pypdf"
        self.file_cache_dir = "/tmp"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.paths[idx]
        pdf_filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()
        text_content = get_text_from_pdf(
            pdf_filename,
            pdf_bytes,
            pdfprocessor=self.pdfprocessor,
            file_cache_dir=self.file_cache_dir,
        )
        return {"filename": pdf_filename, "contents": pdf_bytes, "text_contents": text_content}


_TITLE_MAP_FIELDS = [
    {
        "name": "title",
        "type": str,
        "desc": "A short descriptive title for the paper from its first page or heading.",
    },
]


def mean_sentinel_record_quality(stats: ExecutionStats) -> float | None:
    """Average ``RecordOpStats.quality`` across all sentinel operator stats (if any)."""
    qualities: list[float] = []
    for plan_stats in stats.sentinel_plan_stats.values():
        for phys_map in plan_stats.operator_stats.values():
            if not isinstance(phys_map, dict):
                continue
            for op_stats in phys_map.values():
                for r in op_stats.record_op_stats_lst:
                    if r.quality is not None:
                        qualities.append(float(r.quality))
    if not qualities:
        return None
    return sum(qualities) / len(qualities)


def summarize_run(label: str, stats: ExecutionStats) -> dict:
    mq = mean_sentinel_record_quality(stats)
    return {
        "mode": label,
        "optimization_cost": stats.optimization_cost,
        "optimization_time_s": stats.optimization_time,
        "plan_execution_cost": stats.plan_execution_cost,
        "plan_execution_time_s": stats.plan_execution_time,
        "total_cost": stats.total_execution_cost,
        "total_time_s": stats.total_execution_time,
        "mean_sentinel_quality": mq,
    }


def run_once(
    *,
    document_sampling_method: str,
    eval_paths: list[Path],
    train_paths: list[Path],
    dataset_id: str,
    validator: pz.Validator,
    features_csv: Path,
    sample_budget: int,
    seed: int,
    strata: int,
    k: int,
    j: int,
    progress: bool,
    max_workers: int | None,
    available_models: list[str] | None,
) -> ExecutionStats:
    os.environ["PALIMPZEST_STRATIFIED_FEATURES_PATH"] = str(features_csv.resolve())
    train_ds = PDFPathsDataset(dataset_id, train_paths)
    eval_ds = PDFPathsDataset(dataset_id, eval_paths)
    train_dataset = {dataset_id: train_ds}
    plan = eval_ds.sem_map(_TITLE_MAP_FIELDS)

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        sentinel_execution_strategy="mab",
        document_sampling_method=document_sampling_method,  # type: ignore[arg-type]
        stratified_num_strata=strata,
        sample_budget=sample_budget,
        seed=seed,
        k=k,
        j=j,
        progress=progress,
        max_workers=max_workers,
    )
    if available_models:
        config.available_models = available_models
    result = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)
    if result.execution_stats is None:
        raise RuntimeError("optimize_and_run returned no execution_stats")
    return result.execution_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B: MAB sentinel run with random vs feature-stratified document order."
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("papers"),
        help="Root directory scanned for PDFs (same as extract_features --scan)",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("papers/paper_features.csv"),
        help="Feature table for stratified_source_keys (row_index matches sorted PDF order)",
    )
    parser.add_argument(
        "--train-n",
        type=int,
        default=40,
        help="Number of PDFs (prefix of sorted list) used as sentinel training corpus",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=None,
        help="Cap evaluation corpus to first N PDFs (default: all under --papers)",
    )
    parser.add_argument("--sample-budget", type=int, default=80, help="MAB sentinel sample budget")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for MAB / shuffle")
    parser.add_argument("--strata", type=int, default=8, help="stratified_num_strata")
    parser.add_argument("--k", type=int, default=6, help="MAB k")
    parser.add_argument("--j", type=int, default=4, help="MAB j")
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel execution workers")
    parser.add_argument(
        "--available-models",
        type=str,
        nargs="+",
        default=None,
        help="Model(s) to use, e.g. gpt-4o-mini gpt-4o (overrides auto-detection)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Only run baseline shuffle (skip stratified pass)",
    )
    parser.add_argument(
        "--stratified-only",
        action="store_true",
        help="Only run stratified pass (skip random)",
    )
    args = parser.parse_args()

    if args.random_only and args.stratified_only:
        print("Choose at most one of --random-only / --stratified-only", file=sys.stderr)
        sys.exit(2)

    load_dotenv(override=True)

    papers_root = args.papers.resolve()
    all_paths = iter_pdf_paths(papers_root)
    if not all_paths:
        print(f"No PDFs under {papers_root}", file=sys.stderr)
        sys.exit(1)

    eval_paths = all_paths if args.eval_n is None else all_paths[: args.eval_n]
    train_n = min(args.train_n, len(eval_paths))
    if train_n < 1:
        print("train-n must be >= 1 after clamping to corpus size", file=sys.stderr)
        sys.exit(1)

    train_paths = eval_paths[:train_n]
    dataset_id = "papers"

    if not args.features_csv.is_file():
        print(
            f"Feature CSV not found: {args.features_csv}. Build with:\n"
            f"  python scripts/extract_features.py --scan {papers_root} -o {args.features_csv}",
            file=sys.stderr,
        )
        sys.exit(1)

    with args.features_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        feat_rows = sum(1 for _ in reader)
    if feat_rows < train_n:
        print(
            f"Feature CSV has {feat_rows} rows but --train-n={train_n}; rebuild CSV for this corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    validator = pz.Validator()
    progress = not args.no_progress
    rows: list[dict] = []

    run_kwargs = dict(
        eval_paths=eval_paths,
        train_paths=train_paths,
        dataset_id=dataset_id,
        validator=validator,
        features_csv=args.features_csv.resolve(),
        sample_budget=args.sample_budget,
        seed=args.seed,
        strata=args.strata,
        k=args.k,
        j=args.j,
        progress=progress,
        max_workers=args.max_workers,
        available_models=args.available_models,
    )

    if not args.stratified_only:
        print("=== Run A: document_sampling_method=random (baseline shuffle) ===", flush=True)
        stats_r = run_once(document_sampling_method="random", **run_kwargs)
        rows.append(summarize_run("random", stats_r))

    if not args.random_only:
        print("=== Run B: document_sampling_method=stratified (feature table) ===", flush=True)
        stats_s = run_once(document_sampling_method="stratified", **run_kwargs)
        rows.append(summarize_run("stratified", stats_s))

    table = PrettyTable()
    table.field_names = [
        "Mode",
        "opt_cost",
        "opt_s",
        "plan_cost",
        "plan_s",
        "total_cost",
        "total_s",
        "mean_Q_sentinel",
    ]
    table.align = "r"
    table.align["Mode"] = "l"

    for r in rows:
        mq = r["mean_sentinel_quality"]
        table.add_row(
            [
                r["mode"],
                f"{r['optimization_cost']:.4f}",
                f"{r['optimization_time_s']:.2f}",
                f"{r['plan_execution_cost']:.4f}",
                f"{r['plan_execution_time_s']:.2f}",
                f"{r['total_cost']:.4f}",
                f"{r['total_time_s']:.2f}",
                "—" if mq is None else f"{mq:.4f}",
            ],
        )

    print()
    print(f"Papers root: {papers_root}  |  eval PDFs: {len(eval_paths)}  |  train PDFs: {train_n}")
    print(f"Features CSV: {args.features_csv.resolve()}")
    print(f"sample_budget={args.sample_budget} seed={args.seed} strata={args.strata} k={args.k} j={args.j}")
    print()
    print(table)
    print()
    print(
        "mean_Q_sentinel = mean of non-null RecordOpStats.quality over sentinel ops "
        "(default Validator uses an LLM judge when map_score_fn is not overridden)."
    )


if __name__ == "__main__":
    main()
