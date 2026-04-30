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
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if _REPO_ROOT.is_dir():
    os.chdir(_REPO_ROOT)

try:
    import palimpzest as pz
    from palimpzest.core.data.iter_dataset import IterDataset
    from palimpzest.core.lib.schemas import PDFFile
    from palimpzest.core.models import ExecutionStats
    from palimpzest.tools.pdfparser import get_text_from_pdf
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "unknown")
    print(
        "Missing Python dependency while importing Palimpzest components: "
        f"{missing}\n"
        "Install project dependencies in your active environment, then rerun.\n"
        "Suggested commands:\n"
        "  python -m pip install -e .\n"
        "or (minimal for this error):\n"
        f"  python -m pip install {missing}",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

STRAT_FEATURE_COLUMNS = (
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
    "domain",
)


def _print_results_table(rows: list[dict]) -> None:
    columns = [
        ("budget", lambda r: str(r["sample_budget"])),
        ("Mode", lambda r: str(r["mode"])),
        ("opt_cost", lambda r: f"{r['optimization_cost']:.4f}"),
        ("opt_s", lambda r: f"{r['optimization_time_s']:.2f}"),
        ("eval_cost", lambda r: f"{r['plan_execution_cost']:.4f}"),
        ("eval_s", lambda r: f"{r['plan_execution_time_s']:.2f}"),
        ("total_cost", lambda r: f"{r['total_cost']:.4f}"),
        ("total_s", lambda r: f"{r['total_time_s']:.2f}"),
        ("mean_Q_sentinel", lambda r: "—" if r["mean_sentinel_quality"] is None else f"{r['mean_sentinel_quality']:.4f}"),
        ("mean_Q_plan", lambda r: "—" if r["mean_plan_quality"] is None else f"{r['mean_plan_quality']:.4f}"),
    ]

    if PrettyTable is not None:
        table = PrettyTable()
        table.field_names = [name for name, _ in columns]
        table.align = "r"
        table.align["Mode"] = "l"
        for r in rows:
            table.add_row([fn(r) for _, fn in columns])
        print(table)
        return

    # Plain-text fallback if prettytable is unavailable.
    headers = [name for name, _ in columns]
    rendered_rows = [[fn(r) for _, fn in columns] for r in rows]
    widths = [len(h) for h in headers]
    for row in rendered_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(row_vals: list[str]) -> str:
        parts: list[str] = []
        for i, val in enumerate(row_vals):
            align_left = headers[i] == "Mode"
            parts.append(val.ljust(widths[i]) if align_left else val.rjust(widths[i]))
        return " | ".join(parts)

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered_rows:
        print(fmt(row))


def iter_pdf_paths(papers_root: Path) -> list[Path]:
    if not papers_root.is_dir():
        return []
    return sorted(papers_root.rglob("*.pdf"))


def _eval_feature_frame(eval_paths: list[Path], features_csv: Path) -> pd.DataFrame:
    from palimpzest.query.execution.feature_stratified_sampling import read_feature_table

    feature_df = read_feature_table(features_csv)
    # Normalize path keys so CSVs containing relative paths (e.g. "papers/foo.pdf")
    # still match eval paths resolved to absolute filesystem paths.
    feat_by_path: dict[str, pd.Series] = {}
    for _, row in feature_df.iterrows():
        raw = str(row["path"])
        csv_path = Path(raw).expanduser()
        if not csv_path.is_absolute():
            csv_path = (_REPO_ROOT / csv_path).resolve(strict=False)
        else:
            csv_path = csv_path.resolve(strict=False)
        feat_by_path.setdefault(str(csv_path), row)

    rows = []
    for p in eval_paths:
        key = str(Path(p).resolve(strict=False))
        if key not in feat_by_path:
            raise ValueError(f"Path missing from features CSV: {key}")
        rows.append(feat_by_path[key])
    return pd.DataFrame(rows).reset_index(drop=True)


def _ranked_indices_by_method(
    *,
    eval_paths: list[Path],
    seed: int,
    method: str,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    strata: int,
) -> list[int]:
    if method == "prefix":
        return list(range(len(eval_paths)))
    if method == "random":
        rng = random.Random(seed)
        idxs = list(range(len(eval_paths)))
        rng.shuffle(idxs)
        return idxs
    if method == "stratified":
        selected_cols = feature_columns
        missing_cols = sorted(set(selected_cols) - set(eval_df.columns))
        if missing_cols:
            raise ValueError(f"Missing columns for train stratification: {missing_cols}")
        bins = max(1, min(int(strata), len(eval_df)))
        col_scores = []
        for col in selected_cols:
            series = eval_df[col]
            if not pd.api.types.is_numeric_dtype(series):
                codes, _ = pd.factorize(series.astype(str), sort=True)
                series = pd.Series(codes, index=series.index)
            col_scores.append(series.rank(pct=True, method="average"))
        comp = sum(col_scores) / len(col_scores)
        raw = (comp.to_numpy(dtype=float) * bins).astype(int)
        strata_ids = raw.clip(0, bins - 1)
        bucket_indices: dict[int, list[int]] = {}
        for idx, sid in enumerate(strata_ids.tolist()):
            bucket_indices.setdefault(int(sid), []).append(idx)
        rng = random.Random(seed)
        for ids in bucket_indices.values():
            rng.shuffle(ids)
        ranked: list[int] = []
        while any(bucket_indices.values()):
            for sid in sorted(bucket_indices):
                ids = bucket_indices[sid]
                if ids:
                    ranked.append(ids.pop(0))
        return ranked
    raise ValueError(f"Unsupported --train-selection method: {method}")


def _parse_domain_ratios(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for chunk in [p.strip() for p in raw.split(",") if p.strip()]:
        if "=" not in chunk:
            raise ValueError(f"Invalid domain ratio entry: {chunk!r}; expected form domain=value")
        k, v = [x.strip() for x in chunk.split("=", 1)]
        ratios[k] = float(v)
    if not ratios:
        raise ValueError("No domain ratios parsed.")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Domain ratios must sum to a positive value.")
    return {k: v / total for k, v in ratios.items()}


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {k: total * v for k, v in ratios.items()}
    base = {k: int(x) for k, x in exact.items()}
    remain = total - sum(base.values())
    order = sorted(exact, key=lambda k: (exact[k] - base[k]), reverse=True)
    for i in range(remain):
        base[order[i % len(order)]] += 1
    return base


def choose_train_paths(
    *,
    eval_paths: list[Path],
    train_n: int,
    seed: int,
    method: str,
    features_csv: Path,
    feature_columns: list[str],
    strata: int,
    skew: str,
    skew_focus_domain: str | None = None,
    skew_domain_ratios: str | None = None,
) -> list[Path]:
    """Select train subset from eval paths using requested strategy."""
    if train_n >= len(eval_paths):
        return list(eval_paths)
    eval_df = _eval_feature_frame(eval_paths, features_csv)
    if "domain" not in eval_df.columns:
        raise ValueError("Features CSV must include a 'domain' column for train skew controls.")
    ranked = _ranked_indices_by_method(
        eval_paths=eval_paths,
        seed=seed,
        method=method,
        eval_df=eval_df,
        feature_columns=feature_columns,
        strata=strata,
    )
    if skew == "natural":
        return [eval_paths[i] for i in sorted(ranked[:train_n])]

    domains = eval_df["domain"].astype(str).tolist()
    by_domain: dict[str, list[int]] = {}
    for idx in ranked:
        by_domain.setdefault(domains[idx], []).append(idx)
    all_domains = sorted(by_domain.keys())

    if skew == "balanced_domain":
        ratios = {d: 1.0 / len(all_domains) for d in all_domains}
    elif skew == "min_one_per_domain":
        if train_n < len(all_domains):
            raise ValueError(
                f"train_n={train_n} is smaller than number of domains={len(all_domains)} for min_one_per_domain."
            )
        chosen: list[int] = []
        for d in all_domains:
            if by_domain[d]:
                chosen.append(by_domain[d].pop(0))
        for idx in ranked:
            if len(chosen) >= train_n:
                break
            if idx not in chosen:
                chosen.append(idx)
        return [eval_paths[i] for i in sorted(chosen[:train_n])]
    elif skew == "focus_domain":
        if not skew_focus_domain:
            raise ValueError("--train-skew-focus-domain is required for --train-skew=focus_domain")
        if skew_focus_domain not in by_domain:
            raise ValueError(f"Focus domain {skew_focus_domain!r} not present in eval corpus domains {all_domains}")
        other_domains = [d for d in all_domains if d != skew_focus_domain]
        ratios = {skew_focus_domain: 0.7}
        rem = 0.3
        if other_domains:
            each = rem / len(other_domains)
            ratios.update({d: each for d in other_domains})
    elif skew == "custom_domain_ratios":
        if not skew_domain_ratios:
            raise ValueError("--train-skew-domain-ratios is required for --train-skew=custom_domain_ratios")
        ratios = _parse_domain_ratios(skew_domain_ratios)
        unknown = sorted(set(ratios) - set(all_domains))
        if unknown:
            raise ValueError(f"Custom ratio domains not present in eval corpus: {unknown}")
        for d in all_domains:
            ratios.setdefault(d, 0.0)
    else:
        raise ValueError(f"Unsupported --train-skew: {skew}")

    targets = _allocate_counts(train_n, ratios)
    chosen: list[int] = []
    for d, want in targets.items():
        take = min(want, len(by_domain.get(d, [])))
        chosen.extend(by_domain.get(d, [])[:take])
        by_domain[d] = by_domain.get(d, [])[take:]
    chosen_set = set(chosen)
    if len(chosen) < train_n:
        for idx in ranked:
            if len(chosen) >= train_n:
                break
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
    return [eval_paths[i] for i in sorted(chosen[:train_n])]


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


def mean_plan_record_quality(stats: ExecutionStats) -> float | None:
    """Average ``RecordOpStats.quality`` across final plan operator stats (if any)."""
    qualities: list[float] = []
    for plan_stats in stats.plan_stats.values():
        for op_stats in plan_stats.operator_stats.values():
            for r in op_stats.record_op_stats_lst:
                if r.quality is not None:
                    qualities.append(float(r.quality))
    if not qualities:
        return None
    return sum(qualities) / len(qualities)


def summarize_run(label: str, stats: ExecutionStats) -> dict:
    mq = mean_sentinel_record_quality(stats)
    plan_q = mean_plan_record_quality(stats)
    return {
        "mode": label,
        "optimization_cost": stats.optimization_cost,
        "optimization_time_s": stats.optimization_time,
        "plan_execution_cost": stats.plan_execution_cost,
        "plan_execution_time_s": stats.plan_execution_time,
        "total_cost": stats.total_execution_cost,
        "total_time_s": stats.total_execution_time,
        "mean_sentinel_quality": mq,
        "mean_plan_quality": plan_q,
        "sentinel_plan_count": len(stats.sentinel_plan_stats),
        "final_plan_count": len(stats.plan_stats),
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
    strat_feature_columns: list[str] | None = None,
    strat_mode: str = "composite",
    strat_single_feature: str | None = None,
    map_fields: list[dict] | None = None,
) -> ExecutionStats:
    os.environ["PALIMPZEST_STRATIFIED_FEATURES_PATH"] = str(features_csv.resolve())
    if strat_feature_columns:
        os.environ["PALIMPZEST_STRATIFIED_FEATURE_COLUMNS"] = ",".join(strat_feature_columns)
    else:
        os.environ.pop("PALIMPZEST_STRATIFIED_FEATURE_COLUMNS", None)
    os.environ["PALIMPZEST_STRATIFIED_MODE"] = strat_mode
    if strat_single_feature:
        os.environ["PALIMPZEST_STRATIFIED_SINGLE_FEATURE"] = strat_single_feature
    else:
        os.environ.pop("PALIMPZEST_STRATIFIED_SINGLE_FEATURE", None)
    train_ds = PDFPathsDataset(dataset_id, train_paths)
    eval_ds = PDFPathsDataset(dataset_id, eval_paths)
    train_dataset = {dataset_id: train_ds}
    plan = eval_ds.sem_map(map_fields if map_fields is not None else _TITLE_MAP_FIELDS)

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

    return result.execution_stats


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
        "--sample-budget",
        type=int,
        default=None,
        help="Single MAB sentinel sample budget (deprecated; use --budgets).",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="One or more sample budgets to sweep (e.g. --budgets 5 10 15 20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for MAB / shuffle")
    parser.add_argument("--strata", type=int, default=8, help="stratified_num_strata")
    parser.add_argument("--k", type=int, default=6, help="MAB k")
    parser.add_argument("--j", type=int, default=4, help="MAB j")
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel execution workers")
    parser.add_argument(
        "--train-selection",
        choices=["prefix", "random", "stratified"],
        default="prefix",
        help=(
            "How to choose train subset from eval corpus: "
            "prefix (first N, current behavior), random, or stratified."
        ),
    )
    parser.add_argument(
        "--train-selection-strata",
        type=int,
        default=8,
        help="Number of strata bins when --train-selection=stratified.",
    )
    parser.add_argument(
        "--train-selection-features",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature columns for stratified train-set selection. "
            "Defaults to --stratify-features (or all defaults if unset)."
        ),
    )
    parser.add_argument(
        "--train-skew",
        choices=["natural", "balanced_domain", "min_one_per_domain", "focus_domain", "custom_domain_ratios"],
        default="natural",
        help=(
            "Target training-set domain composition: natural, balanced_domain, "
            "min_one_per_domain, focus_domain, or custom_domain_ratios."
        ),
    )
    parser.add_argument(
        "--train-skew-focus-domain",
        type=str,
        default=None,
        help="Domain to emphasize when --train-skew=focus_domain.",
    )
    parser.add_argument(
        "--train-skew-domain-ratios",
        type=str,
        default=None,
        help="Comma-separated ratios for --train-skew=custom_domain_ratios, e.g. cs=0.5,biomedical=0.2,math=0.2,physics=0.1.",
    )
    parser.add_argument(
        "--stratify-features",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature columns to use for stratification. "
            f"Default: {', '.join(STRAT_FEATURE_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--strata-composition",
        choices=["cartesian", "composite", "exclusive"],
        default="cartesian",
        help=(
            "cartesian: one stratified run using the Cartesian product of all selected features. "
            "composite: one stratified run using all selected features. "
            "exclusive: one stratified run per selected feature (mutually exclusive)."
        ),
    )
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
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save detailed run results CSV.",
    )
    parser.add_argument(
        "--fields-json",
        type=str,
        default=None,
        help=(
            "JSON array defining extraction fields, e.g. "
            "'[{\"name\": \"title\", \"type\": \"str\", \"desc\": \"Paper title.\"}]'. "
            "Overrides the default _TITLE_MAP_FIELDS. Supported types: str, bool, int, float."
        ),
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
        print("train-n must be >= 1", file=sys.stderr)
        sys.exit(1)
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
    rows: list[dict] = []
    features_csv_path = args.features_csv.resolve()

    if args.budgets is not None and len(args.budgets) == 0:
        print("--budgets cannot be empty", file=sys.stderr)
        sys.exit(2)
    if args.budgets is not None and args.sample_budget is not None:
        print("Pass either --sample-budget or --budgets (not both).", file=sys.stderr)
        sys.exit(2)
    budgets = args.budgets if args.budgets is not None else [args.sample_budget or 80]
    selected_features = list(args.stratify_features or STRAT_FEATURE_COLUMNS)
    unknown_features = sorted(set(selected_features) - set(STRAT_FEATURE_COLUMNS))
    if unknown_features:
        print(
            f"Unknown --stratify-features columns: {unknown_features}. "
            f"Allowed: {list(STRAT_FEATURE_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(selected_features) == 0:
        print("At least one stratification feature is required.", file=sys.stderr)
        sys.exit(2)
    train_selection_features = list(args.train_selection_features or selected_features)
    unknown_train_features = sorted(set(train_selection_features) - set(STRAT_FEATURE_COLUMNS))
    if unknown_train_features:
        print(
            f"Unknown --train-selection-features columns: {unknown_train_features}. "
            f"Allowed: {list(STRAT_FEATURE_COLUMNS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(train_selection_features) == 0:
        print("At least one --train-selection-features column is required.", file=sys.stderr)
        sys.exit(2)
    try:
        train_paths = choose_train_paths(
            eval_paths=eval_paths,
            train_n=train_n,
            seed=args.seed,
            method=args.train_selection,
            features_csv=features_csv_path,
            feature_columns=train_selection_features,
            strata=args.train_selection_strata,
            skew=args.train_skew,
            skew_focus_domain=args.train_skew_focus_domain,
            skew_domain_ratios=args.train_skew_domain_ratios,
        )
    except Exception as exc:
        print(f"Failed to build train subset: {exc}", file=sys.stderr)
        sys.exit(2)

    def stratified_runs() -> list[tuple[str, str, str | None]]:
        if args.strata_composition == "exclusive":
            return [(f"stratified:{feat}", "single", feat) for feat in selected_features]
        if args.strata_composition == "cartesian":
            return [("stratified", "cartesian", None)]
        return [("stratified", "composite", None)]

    # parse --fields-json if provided
    _TYPE_MAP = {"str": str, "bool": bool, "int": int, "float": float}
    map_fields: list[dict] | None = None
    if args.fields_json:
        try:
            raw_fields = json.loads(args.fields_json)
            map_fields = []
            for f in raw_fields:
                type_str = f.get("type", "str")
                if type_str not in _TYPE_MAP:
                    print(f"Unknown field type {type_str!r}. Use one of: {list(_TYPE_MAP)}", file=sys.stderr)
                    sys.exit(2)
                map_fields.append({"name": f["name"], "type": _TYPE_MAP[type_str], "desc": f["desc"]})
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"Invalid --fields-json: {exc}", file=sys.stderr)
            sys.exit(2)

    for budget in budgets:
        run_kwargs = dict(
            eval_paths=eval_paths,
            train_paths=train_paths,
            dataset_id=dataset_id,
            validator=validator,
            features_csv=features_csv_path,
            sample_budget=budget,
            seed=args.seed,
            strata=args.strata,
            k=args.k,
            j=args.j,
            progress=progress,
            max_workers=args.max_workers,
            available_models=args.available_models,
            strat_feature_columns=selected_features,
            map_fields=map_fields,
        )

        if not args.stratified_only:
            print(
                f"=== Run A: document_sampling_method=random (baseline shuffle), budget={budget} ===",
                flush=True,
            )
            stats_r = run_once(document_sampling_method="random", **run_kwargs)
            row_r = summarize_run("random", stats_r)
            row_r["sample_budget"] = budget
            row_r["stratify_features"] = ",".join(selected_features)
            row_r["strata_composition"] = args.strata_composition
            row_r["strata_feature"] = ""
            row_r["train_selection"] = args.train_selection
            row_r["train_selection_features"] = ",".join(train_selection_features)
            row_r["train_skew"] = args.train_skew
            row_r["train_skew_focus_domain"] = args.train_skew_focus_domain or ""
            row_r["train_skew_domain_ratios"] = args.train_skew_domain_ratios or ""
            rows.append(row_r)

        if not args.random_only:
            for mode_label, strat_mode, strat_single in stratified_runs():
                print(
                    "=== Run B: document_sampling_method=stratified "
                    f"({mode_label}), budget={budget} ===",
                    flush=True,
                )
                stats_s = run_once(
                    document_sampling_method="stratified",
                    strat_mode=strat_mode,
                    strat_single_feature=strat_single,
                    **run_kwargs,
                )
                row_s = summarize_run(mode_label, stats_s)
                row_s["sample_budget"] = budget
                row_s["stratify_features"] = ",".join(selected_features)
                row_s["strata_composition"] = args.strata_composition
                row_s["strata_feature"] = strat_single or ""
                row_s["train_selection"] = args.train_selection
                row_s["train_selection_features"] = ",".join(train_selection_features)
                row_s["train_skew"] = args.train_skew
                row_s["train_skew_focus_domain"] = args.train_skew_focus_domain or ""
                row_s["train_skew_domain_ratios"] = args.train_skew_domain_ratios or ""
                rows.append(row_s)

    print()
    print(f"Papers root: {papers_root}  |  eval PDFs: {len(eval_paths)}  |  train PDFs: {len(train_paths)}")
    print(f"Features CSV: {features_csv_path}")
    print(f"stratify_features={selected_features} strata_composition={args.strata_composition}")
    print(
        "train_selection="
        f"{args.train_selection} "
        f"(features={train_selection_features}, strata={args.train_selection_strata})"
    )
    print(
        "train_skew="
        f"{args.train_skew} "
        f"(focus_domain={args.train_skew_focus_domain}, ratios={args.train_skew_domain_ratios})"
    )
    print(f"budgets={budgets} seed={args.seed} strata={args.strata} k={args.k} j={args.j}")
    print()
    _print_results_table(rows)
    print()
    print(
        "mean_Q_sentinel = mean of non-null RecordOpStats.quality over sentinel ops "
        "(default Validator uses an LLM judge when map_score_fn is not overridden)."
    )
    print("mean_Q_plan = mean of non-null RecordOpStats.quality over final executed plan operators.")

    if args.output_csv is not None:
        out_csv = args.output_csv.resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_budget",
            "mode",
            "optimization_cost",
            "optimization_time_s",
            "plan_execution_cost",
            "plan_execution_time_s",
            "total_cost",
            "total_time_s",
            "mean_sentinel_quality",
            "mean_plan_quality",
            "sentinel_plan_count",
            "final_plan_count",
            "stratify_features",
            "strata_composition",
            "strata_feature",
            "train_selection",
            "train_selection_features",
            "train_skew",
            "train_skew_focus_domain",
            "train_skew_domain_ratios",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Wrote results CSV: {out_csv}")


if __name__ == "__main__":
    main()
