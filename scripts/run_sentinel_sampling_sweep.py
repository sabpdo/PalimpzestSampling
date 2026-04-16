#!/usr/bin/env python3
"""Run multi-seed sentinel A/B with downstream metrics + plots.

Two conditions are compared under fixed policy/model/budget:
1) baseline: ``document_sampling_method="random"``
2) stratified: ``document_sampling_method="stratified"``

To avoid binary-PDF hashing issues in sentinel cache keys, this script evaluates on a
text-only ``MemoryDataset`` built from extracted PDF text. It still executes real
``optimize_and_run()`` with MAB sentinel optimization.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if _REPO_ROOT.is_dir():
    os.chdir(_REPO_ROOT)
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_seeds(s: str) -> list[int]:
    vals = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("No seeds parsed from --seeds")
    return vals


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.mean(clean))


def _safe_std(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(statistics.stdev(clean))


def _safe_var(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(statistics.variance(clean))


def _iter_pdf_paths(papers_root: Path) -> list[Path]:
    if not papers_root.is_dir():
        return []
    return sorted(papers_root.rglob("*.pdf"))


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)


def _build_text_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows.append(
            {
                "title": p.stem,
                "text": _extract_pdf_text(p),
            }
        )
    return rows


def _mean_sentinel_record_quality(stats) -> float | None:
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


def _run_once(
    *,
    pz,
    document_sampling_method: str,
    eval_rows: list[dict],
    train_rows: list[dict],
    sample_budget: int,
    seed: int,
    strata: int,
    k: int,
    j: int,
    max_workers: int | None,
    available_models: list[str] | None,
) -> dict:
    schema = [
        {"name": "title", "type": str, "desc": "Paper title"},
        {"name": "text", "type": str, "desc": "Extracted plain text"},
    ]
    dataset_id = "papers-text"
    eval_ds = pz.MemoryDataset(id=dataset_id, vals=eval_rows, schema=schema)
    train_ds = pz.MemoryDataset(id=dataset_id, vals=train_rows, schema=schema)

    plan = eval_ds.sem_filter(
        "The document is about machine learning, deep learning, or neural networks. "
        "Return True or False.",
        depends_on=["text"],
    ).project(["title"])

    validator = pz.Validator()
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
        progress=False,
        max_workers=max_workers,
    )
    if available_models:
        config.available_models = available_models

    result = plan.optimize_and_run(config=config, train_dataset={dataset_id: train_ds}, validator=validator)
    if result.execution_stats is None:
        raise RuntimeError("optimize_and_run returned no execution_stats")
    stats = result.execution_stats
    pred_titles = sorted([r["title"] for r in result])
    return {
        "optimization_cost": float(stats.optimization_cost),
        "optimization_time_s": float(stats.optimization_time),
        "plan_execution_cost": float(stats.plan_execution_cost),
        "plan_execution_time_s": float(stats.plan_execution_time),
        "total_cost": float(stats.total_execution_cost),
        "total_time_s": float(stats.total_execution_time),
        "mean_sentinel_quality": _mean_sentinel_record_quality(stats),
        "num_outputs": len(pred_titles),
        "pred_titles": pred_titles,
    }


def _plot_summary(summary_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("total_cost", "Total Execution Cost ($)"),
        ("total_time_s", "Total Execution Time (s)"),
        ("mean_sentinel_quality", "Mean Sentinel Quality (proxy)"),
    ]

    methods = [r["mode"] for r in summary_rows]

    for metric_key, y_label in metrics:
        means = [r[f"{metric_key}_mean"] for r in summary_rows]
        stds = [r[f"{metric_key}_std"] for r in summary_rows]

        # replace None with 0 so matplotlib handles missing quality values
        means_plot = [0.0 if v is None else float(v) for v in means]
        stds_plot = [0.0 if v is None else float(v) for v in stds]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(methods, means_plot, yerr=stds_plot, capsize=6, color=["#457b9d", "#2a9d8f"])
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label}: mean ± std across seeds")
        fig.tight_layout()
        out = out_dir / f"{metric_key}_mean_std.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)

    # Cost-time tradeoff scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in summary_rows:
        x = row["total_cost_mean"]
        y = row["total_time_s_mean"]
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=120, label=row["mode"])
        ax.annotate(row["mode"], (x, y), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Mean Total Cost ($)")
    ax.set_ylabel("Mean Total Time (s)")
    ax.set_title("Cost vs Latency across sampling methods")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = out_dir / "cost_vs_time_scatter.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed sentinel A/B sweep with plots (random vs stratified)."
    )
    parser.add_argument("--papers", type=Path, default=Path("papers"))
    parser.add_argument("--features-csv", type=Path, default=Path("papers/paper_features.csv"))
    parser.add_argument("--train-n", type=int, default=40)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument("--sample-budget", type=int, default=80)
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--available-models",
        type=str,
        nargs="+",
        default=None,
        help="Optional model allowlist, e.g. openai/gpt-4o-mini",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="41,42,43",
        help="Comma-separated RNG seeds, e.g. 41,42,43",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sentinel_sweep"))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    papers_root = args.papers.resolve()
    all_paths = _iter_pdf_paths(papers_root)
    if not all_paths:
        raise SystemExit(f"No PDFs under {papers_root}")
    eval_paths = all_paths if args.eval_n is None else all_paths[: args.eval_n]
    train_n = min(args.train_n, len(eval_paths))
    if train_n < 1:
        raise SystemExit("train_n must be >= 1")
    train_paths = eval_paths[:train_n]

    if not args.features_csv.is_file():
        raise SystemExit(
            f"Missing features CSV: {args.features_csv}. "
            f"Build with: python scripts/extract_features.py --scan {papers_root} -o {args.features_csv}"
        )
    with args.features_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        feat_rows = sum(1 for _ in reader)
    if feat_rows < train_n:
        raise SystemExit(
            f"Feature CSV has {feat_rows} rows but train_n={train_n}; rebuild CSV for this corpus."
        )

    import palimpzest as pz

    seeds = _parse_seeds(args.seeds)
    progress = not args.no_progress

    all_runs: list[dict] = []
    predictions_by_mode: dict[str, list[set[str]]] = defaultdict(list)
    eval_rows = _build_text_rows(eval_paths)
    train_rows = _build_text_rows(train_paths)

    for seed in seeds:
        os.environ["PALIMPZEST_STRATIFIED_FEATURES_PATH"] = str(args.features_csv.resolve())
        print(f"\n=== seed={seed} | random ===", flush=True)
        out_random = _run_once(
            pz=pz,
            document_sampling_method="random",
            eval_rows=eval_rows,
            train_rows=train_rows,
            sample_budget=args.sample_budget,
            seed=seed,
            strata=args.strata,
            k=args.k,
            j=args.j,
            max_workers=args.max_workers,
            available_models=args.available_models,
        )
        row_random = {"mode": "random", "seed": seed, **out_random}
        row_random.pop("pred_titles")
        all_runs.append(row_random)
        predictions_by_mode["random"].append(set(out_random["pred_titles"]))

        print(f"=== seed={seed} | stratified ===", flush=True)
        out_strat = _run_once(
            pz=pz,
            document_sampling_method="stratified",
            eval_rows=eval_rows,
            train_rows=train_rows,
            sample_budget=args.sample_budget,
            seed=seed,
            strata=args.strata,
            k=args.k,
            j=args.j,
            max_workers=args.max_workers,
            available_models=args.available_models,
        )
        row_strat = {"mode": "stratified", "seed": seed, **out_strat}
        row_strat.pop("pred_titles")
        all_runs.append(row_strat)
        predictions_by_mode["stratified"].append(set(out_strat["pred_titles"]))

    raw_csv = args.out_dir / "runs.csv"
    raw_fields = [
        "mode",
        "seed",
        "optimization_cost",
        "optimization_time_s",
        "plan_execution_cost",
        "plan_execution_time_s",
        "total_cost",
        "total_time_s",
        "mean_sentinel_quality",
        "num_outputs",
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in all_runs:
            w.writerow(r)

    by_mode: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    for r in all_runs:
        m = r["mode"]
        by_mode[m]["total_cost"].append(float(r["total_cost"]))
        by_mode[m]["total_time_s"].append(float(r["total_time_s"]))
        by_mode[m]["mean_sentinel_quality"].append(r["mean_sentinel_quality"])
        by_mode[m]["num_outputs"].append(float(r["num_outputs"]))

    def _mean_pairwise_jaccard(sets: list[set[str]]) -> float | None:
        if len(sets) < 2:
            return None
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                den = len(a | b)
                vals.append(1.0 if den == 0 else len(a & b) / den)
        return float(statistics.mean(vals)) if vals else None

    summary_rows: list[dict] = []
    for mode in sorted(by_mode.keys()):
        vals = by_mode[mode]
        row = {
            "mode": mode,
            "n_runs": len(vals["total_cost"]),
            "total_cost_mean": _safe_mean(vals["total_cost"]),
            "total_cost_std": _safe_std(vals["total_cost"]),
            "total_cost_var": _safe_var(vals["total_cost"]),
            "total_time_s_mean": _safe_mean(vals["total_time_s"]),
            "total_time_s_std": _safe_std(vals["total_time_s"]),
            "total_time_s_var": _safe_var(vals["total_time_s"]),
            "mean_sentinel_quality_mean": _safe_mean(vals["mean_sentinel_quality"]),
            "mean_sentinel_quality_std": _safe_std(vals["mean_sentinel_quality"]),
            "mean_sentinel_quality_var": _safe_var(vals["mean_sentinel_quality"]),
            "num_outputs_mean": _safe_mean(vals["num_outputs"]),
            "num_outputs_std": _safe_std(vals["num_outputs"]),
            "consistency_jaccard_mean": _mean_pairwise_jaccard(predictions_by_mode.get(mode, [])),
        }
        summary_rows.append(row)

    summary_csv = args.out_dir / "summary_by_method.csv"
    summary_fields = [
        "mode",
        "n_runs",
        "total_cost_mean",
        "total_cost_std",
        "total_cost_var",
        "total_time_s_mean",
        "total_time_s_std",
        "total_time_s_var",
        "mean_sentinel_quality_mean",
        "mean_sentinel_quality_std",
        "mean_sentinel_quality_var",
        "num_outputs_mean",
        "num_outputs_std",
        "consistency_jaccard_mean",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    _plot_summary(summary_rows, args.out_dir)

    print("\nWrote:")
    print(f"  - {raw_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {args.out_dir / 'total_cost_mean_std.png'}")
    print(f"  - {args.out_dir / 'total_time_s_mean_std.png'}")
    print(f"  - {args.out_dir / 'mean_sentinel_quality_mean_std.png'}")
    print(f"  - {args.out_dir / 'cost_vs_time_scatter.png'}")


if __name__ == "__main__":
    main()
