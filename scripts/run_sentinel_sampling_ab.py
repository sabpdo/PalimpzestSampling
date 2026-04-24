#!/usr/bin/env python3
"""
Run Palimpzest sentinel optimization at increasing sample budgets and compare
random vs. feature-stratified document ordering.

For each budget in --budgets, both methods are run and the quality of the plan
chosen by MAB is recorded. Results are plotted as quality (and error) vs. sample budget.

Ground truth is the quality at the largest budget.

::

    python scripts/run_sentinel_sampling_ab.py --papers ./papers --train-n 40
    python scripts/run_sentinel_sampling_ab.py --papers ./papers --train-n 20 --budgets 5 10 15 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    if not papers_root.is_dir():
        return []
    return sorted(papers_root.rglob("*.pdf"))


class PDFPathsDataset(IterDataset):
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
        "name": "primary_contribution",
        "type": str,
        "desc": "The single most important technical contribution of the paper in one sentence.",
    },
    {
        "name": "methodology",
        "type": str,
        "desc": "The core method or approach used (e.g. algorithm name, experimental design, proof technique).",
    },
    {
        "name": "domain",
        "type": str,
        "desc": "The research domain: one of 'cs', 'biomedical', 'math', or 'physics'.",
    },
    {
        "name": "uses_experiments",
        "type": bool,
        "desc": "True if the paper includes empirical experiments or evaluations, False if purely theoretical.",
    },
]


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
) -> tuple[ExecutionStats, float | None]:
    """Run once and return (stats, final_best_op_quality).

    final_best_op_quality is the quality of the operator MAB selects at the
    end of sampling — i.e. the quality of the chosen plan.
    """
    os.environ["PALIMPZEST_STRATIFIED_FEATURES_PATH"] = str(features_csv.resolve())
    train_ds = PDFPathsDataset(dataset_id, train_paths)
    eval_ds = PDFPathsDataset(dataset_id, eval_paths)
    train_dataset = {dataset_id: train_ds}
    plan = eval_ds.sem_map(_TITLE_MAP_FIELDS)

    last_quality: list[float] = []

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
        on_sample=lambda n, q: last_quality.append(q),
    )
    if available_models:
        config.available_models = available_models

    result = plan.optimize_and_run(config=config, train_dataset=train_dataset, validator=validator)
    if result.execution_stats is None:
        raise RuntimeError("optimize_and_run returned no execution_stats")

    final_quality = last_quality[-1] if last_quality else None
    chosen_plans = list(result.execution_stats.plan_strs.values())
    chosen_plan = chosen_plans[0] if chosen_plans else "unknown"
    return result.execution_stats, final_quality, chosen_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample-efficiency sweep: random vs. stratified sentinel ordering across budgets."
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("papers"),
        help="Root directory scanned for PDFs",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("papers/paper_features.csv"),
        help="Feature table CSV for stratified ordering",
    )
    parser.add_argument(
        "--train-n",
        type=int,
        default=40,
        help="Number of PDFs used as sentinel training corpus",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=None,
        help="Cap evaluation corpus to first N PDFs (default: all)",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="Sample budgets to sweep over (default: 5 10 15 20 25 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--available-models", type=str, nargs="+", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--stratified-only", action="store_true")
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
        print("train-n must be >= 1", file=sys.stderr)
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
            f"Feature CSV has {feat_rows} rows but --train-n={train_n}; rebuild CSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    validator = pz.Validator()
    progress = not args.no_progress
    budgets = sorted(args.budgets)

    run_kwargs = dict(
        eval_paths=eval_paths,
        train_paths=train_paths,
        dataset_id=dataset_id,
        validator=validator,
        features_csv=args.features_csv.resolve(),
        seed=args.seed,
        strata=args.strata,
        k=args.k,
        j=args.j,
        progress=progress,
        max_workers=args.max_workers,
        available_models=args.available_models,
    )

    # results[mode] = list of row dicts in budget order
    results: dict[str, list[dict]] = {"random": [], "stratified": []}

    for budget in budgets:
        print(f"\n=== sample_budget={budget} ===", flush=True)

        if not args.stratified_only:
            print(f"  [random]     budget={budget}", flush=True)
            stats, q, plan = run_once(document_sampling_method="random", sample_budget=budget, **run_kwargs)
            if q is not None:
                results["random"].append({
                    "budget": budget,
                    "plan_quality": q,
                    "chosen_plan": plan,
                    "opt_cost": stats.optimization_cost,
                    "opt_s": stats.optimization_time,
                    "plan_cost": stats.plan_execution_cost,
                    "plan_s": stats.plan_execution_time,
                    "total_cost": stats.total_execution_cost,
                    "total_s": stats.total_execution_time,
                })
                print(f"  -> plan quality: {q:.4f}  plan: {plan}")

        if not args.random_only:
            print(f"  [stratified] budget={budget}", flush=True)
            stats, q, plan = run_once(document_sampling_method="stratified", sample_budget=budget, **run_kwargs)
            if q is not None:
                results["stratified"].append({
                    "budget": budget,
                    "plan_quality": q,
                    "chosen_plan": plan,
                    "opt_cost": stats.optimization_cost,
                    "opt_s": stats.optimization_time,
                    "plan_cost": stats.plan_execution_cost,
                    "plan_s": stats.plan_execution_time,
                    "total_cost": stats.total_execution_cost,
                    "total_s": stats.total_execution_time,
                })
                print(f"  -> plan quality: {q:.4f}  plan: {plan}")

    # ground truth = quality at the largest budget
    ground_truth: dict[str, float] = {}
    for mode, rows in results.items():
        if rows:
            ground_truth[mode] = rows[-1]["plan_quality"]

    shared_gt = sum(ground_truth.values()) / len(ground_truth) if ground_truth else None

    # summary table
    table = PrettyTable()
    table.field_names = ["Mode", "budget", "chosen_plan", "plan_quality", "error_vs_gt", "opt_cost", "opt_s", "plan_cost", "plan_s", "total_cost", "total_s"]
    table.align = "r"
    table.align["Mode"] = "l"
    table.align["chosen_plan"] = "l"
    for mode, rows in results.items():
        for row in rows:
            q = row["plan_quality"]
            err = f"{abs(q - shared_gt):.4f}" if shared_gt is not None else "—"
            table.add_row([
                mode,
                row["budget"],
                row["chosen_plan"],
                f"{q:.4f}",
                err,
                f"{row['opt_cost']:.4f}",
                f"{row['opt_s']:.2f}",
                f"{row['plan_cost']:.4f}",
                f"{row['plan_s']:.2f}",
                f"{row['total_cost']:.4f}",
                f"{row['total_s']:.2f}",
            ])

    print()
    print(f"Papers root: {papers_root}  |  eval PDFs: {len(eval_paths)}  |  train PDFs: {train_n}")
    print(f"Budgets: {budgets}  |  seed={args.seed}  strata={args.strata}  k={args.k}  j={args.j}")
    print()
    print(table)

    # plot
    colors = {"random": "steelblue", "stratified": "darkorange"}
    labels = {"random": "Random (baseline)", "stratified": "Feature-stratified"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, rows in results.items():
        if not rows:
            continue
        xs = [r["budget"] for r in rows]
        qs = [r["plan_quality"] for r in rows]
        color = colors[mode]
        label = labels[mode]

        ax1.plot(xs, qs, marker="o", label=label, color=color, linewidth=2)

        if shared_gt is not None:
            errs = [abs(q - shared_gt) for q in qs]
            ax2.plot(xs, errs, marker="o", label=label, color=color, linewidth=2)

    ax1.set_xlabel("Sample budget")
    ax1.set_ylabel("Plan quality (best operator mean quality)")
    ax1.set_title("Plan quality vs. sample budget")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sample budget")
    ax2.set_ylabel("Absolute error vs. shared ground truth")
    ax2.set_title("Sample efficiency: convergence to ground truth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = Path("sample_efficiency.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
