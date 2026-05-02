#!/usr/bin/env python3
"""
Automated stress-dataset sweep: generate skewed feature pools, run random vs stratified A/B,
aggregate CSVs, save figures under ``results/stress_test_dataset_analysis/``.

Usage::

    python scripts/run_stress_test_sweep.py --repo-root . --max-cases 6

Requires the same env/API setup as ``run_sentinel_sampling_ab.py`` (``.env``, keys, etc.).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

STRAT_FEATURES = [
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
    "domain",
]

# Lightweight extraction to keep sweep runtime manageable.
SWEEP_FIELDS_JSON = json.dumps(
    [
        {
            "name": "primary_contribution",
            "type": "str",
            "desc": "Main contribution in one concise sentence.",
        },
        {
            "name": "methodology",
            "type": "str",
            "desc": "Core method or approach used in the paper.",
        },
        {
            "name": "domain",
            "type": "str",
            "desc": "One of cs, biomedical, math, physics.",
        },
        {
            "name": "uses_experiments",
            "type": "bool",
            "desc": "True if empirical experiments are reported.",
        },
    ]
)


@dataclass(frozen=True)
class SweepCase:
    dataset_id: str
    preset: str
    gen_seed: int
    n_docs: int
    stress_overrides: dict
    notes: str


DEFAULT_CASES: tuple[SweepCase, ...] = (
    SweepCase(
        "rnt_tail08",
        "rare_numeric_tail",
        101,
        28,
        {"tail_fraction": 0.08},
        "Rare tail ~8% — subtle minority.",
    ),
    SweepCase(
        "rnt_tail15",
        "rare_numeric_tail",
        102,
        28,
        {"tail_fraction": 0.15},
        "Rare tail ~15% — stronger heterogeneity.",
    ),
    SweepCase(
        "rnt_tail25",
        "rare_numeric_tail",
        103,
        28,
        {"tail_fraction": 0.25},
        "Rare tail ~25% — heavy minority.",
    ),
    SweepCase(
        "dhs_maj090",
        "domain_heavy_skew",
        201,
        28,
        {"majority_fraction": 0.90},
        "Domain skew 90/10.",
    ),
    SweepCase(
        "dhs_maj095",
        "domain_heavy_skew",
        202,
        28,
        {"majority_fraction": 0.95},
        "Domain skew 95/5.",
    ),
    SweepCase(
        "bim_short088",
        "bimodal_sections",
        301,
        28,
        {"short_section_fraction": 0.88},
        "Bimodal sections ~88% short.",
    ),
    SweepCase(
        "bim_short092",
        "bimodal_sections",
        302,
        28,
        {"short_section_fraction": 0.92},
        "Bimodal sections ~92% short.",
    ),
    SweepCase(
        "rnt_tail20_wide",
        "rare_numeric_tail",
        401,
        32,
        {
            "tail_fraction": 0.20,
            "bulk_word_low": 2500,
            "bulk_word_high": 7500,
            "tail_word_low": 95000,
            "tail_word_high": 125000,
        },
        "Wider bulk/tail word separation.",
    ),
    SweepCase(
        "ultra_tail03",
        "ultra_rare_domain_tail",
        501,
        36,
        {
            "ultra_rare_fraction": 0.03,
            "ultra_rare_word_low": 145000,
            "ultra_rare_word_high": 245000,
            "ultra_rare_section_low": 280,
            "ultra_rare_section_high": 560,
        },
        "Needle-in-haystack (<3.5%) ultra-extreme outliers.",
    ),
    SweepCase(
        "ultra_tail02",
        "ultra_rare_domain_tail",
        502,
        40,
        {
            "ultra_rare_fraction": 0.02,
            "ultra_rare_word_low": 165000,
            "ultra_rare_word_high": 265000,
        },
        "Worst-case ultra-rare tail (~2%).",
    ),
    SweepCase(
        "adv_conflict45",
        "adversarial_feature_conflict",
        601,
        36,
        {
            "conflict_fraction": 0.45,
            "conflict_a_word_low": 5000,
            "conflict_a_word_high": 18000,
            "conflict_b_word_low": 60000,
            "conflict_b_word_high": 130000,
        },
        "Adversarial conflicting feature regimes (45/55 split).",
    ),
    SweepCase(
        "adv_conflict65",
        "adversarial_feature_conflict",
        602,
        36,
        {
            "conflict_fraction": 0.65,
            "conflict_a_word_low": 4500,
            "conflict_a_word_high": 16000,
            "conflict_b_word_low": 70000,
            "conflict_b_word_high": 150000,
        },
        "Adversarial conflict with stronger cohort-A skew.",
    ),
)


def _load_generator():
    path = REPO_ROOT / "scripts" / "generate_stratifier_stress_dataset.py"
    name = "_palimpzest_stress_dataset_gen"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_dirs(out_root: Path) -> tuple[Path, Path, Path]:
    pools = out_root / "pools"
    runs = out_root / "runs"
    figs = out_root / "figures"
    for d in (out_root, pools, runs, figs):
        d.mkdir(parents=True, exist_ok=True)
    return pools, runs, figs


def _generate_pool(mod, repo: Path, out_root: Path, case: SweepCase, source_papers: str, link_mode: str) -> dict:
    repo_r = repo.resolve()
    out_r = out_root.resolve()
    rel_pool = str(out_r.relative_to(repo_r) / "pools" / case.dataset_id)
    params = mod.merge_stress_params(mod.StressDatasetParams.defaults(), case.stress_overrides)
    summary = mod.generate_stratifier_stress_dataset(
        repo,
        output_dir=rel_pool,
        seed=case.gen_seed,
        n_docs=case.n_docs,
        preset=case.preset,
        source_papers=source_papers,
        link_mode=link_mode,
        stress_params=params,
    )
    summary["dataset_id"] = case.dataset_id
    summary["case_notes"] = case.notes
    summary["stress_overrides_applied"] = case.stress_overrides
    return summary


def _run_ab(
    repo: Path,
    *,
    papers_rel: str,
    features_rel: str,
    out_csv: Path,
    train_n: int,
    eval_n: int,
    budgets: list[int],
    exp_seed: int,
    strata: int,
    k: int,
    j: int,
    max_workers: int | None,
    no_progress: bool,
) -> int:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_sentinel_sampling_ab.py"),
        "--papers",
        papers_rel,
        "--features-csv",
        features_rel,
        "--train-n",
        str(train_n),
        "--eval-n",
        str(eval_n),
        "--budgets",
        *[str(b) for b in budgets],
        "--seed",
        str(exp_seed),
        "--strata",
        str(strata),
        "--k",
        str(k),
        "--j",
        str(j),
        "--strata-composition",
        "cartesian",
        "--stratify-features",
        *STRAT_FEATURES,
        "--train-selection",
        "prefix",
        "--train-selection-strata",
        str(strata),
        "--train-selection-features",
        *STRAT_FEATURES,
        "--train-skew",
        "natural",
        "--output-csv",
        str(out_csv),
        "--fields-json",
        SWEEP_FIELDS_JSON,
    ]
    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])
    if no_progress:
        cmd.append("--no-progress")
    proc = subprocess.run(cmd, cwd=str(repo), env=dict(**__import__("os").environ))
    return int(proc.returncode)


def _paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    need = {"dataset_id", "sample_budget", "mode", "mean_sentinel_quality"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    g = df.groupby(["dataset_id", "sample_budget", "mode"], dropna=False)["mean_sentinel_quality"].mean().unstack("mode")
    if "random" not in g.columns or "stratified" not in g.columns:
        return pd.DataFrame()
    out = g[["random", "stratified"]].copy()
    out["delta_sentinel_quality"] = out["stratified"] - out["random"]
    out = out.reset_index().rename(columns={"random": "mean_sentinel_random", "stratified": "mean_sentinel_stratified"})
    return out


def _two_sided_binom_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5**n)
    p *= 2.0
    return min(1.0, p)


def _paired_metric_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    piv = df.pivot_table(
        index=["dataset_id", "exp_seed", "sample_budget"],
        columns="mode",
        values=metric_col,
        aggfunc="mean",
    )
    if "random" not in piv.columns or "stratified" not in piv.columns:
        return pd.DataFrame()
    p = piv[["random", "stratified"]].dropna().copy()
    p["delta"] = p["stratified"] - p["random"]
    return p.reset_index()


def _significance_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("mean_sentinel_quality", "higher"),
        ("mean_plan_quality", "higher"),
        ("total_time_s", "lower"),
        ("total_cost", "lower"),
    ]
    for metric, direction in specs:
        pt = _paired_metric_table(all_runs, metric)
        if pt.empty:
            continue
        d = pt["delta"]
        if direction == "higher":
            wins = int((d > 0).sum())
            losses = int((d < 0).sum())
        else:
            wins = int((d < 0).sum())
            losses = int((d > 0).sum())
        ties = int((d == 0).sum())
        pval = _two_sided_binom_p(wins, losses)
        rows.append(
            {
                "metric": metric,
                "direction_better_for_stratified": direction,
                "n_pairs": int(len(d)),
                "stratified_wins": wins,
                "random_wins": losses,
                "ties": ties,
                "win_rate_excluding_ties": float(wins / (wins + losses)) if (wins + losses) > 0 else float("nan"),
                "mean_delta": float(d.mean()),
                "median_delta": float(d.median()),
                "std_delta": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                "sign_test_p_value_two_sided": float(pval),
                "is_significant_p_lt_0_05": bool((pval < 0.05) if not pd.isna(pval) else False),
            }
        )
    return pd.DataFrame(rows)


def _plot_deltas(summary: pd.DataFrame, fig_dir: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    budgets = sorted(summary["sample_budget"].unique())
    datasets = summary["dataset_id"].unique()
    x = range(len(budgets))
    width = min(0.8 / max(len(datasets), 1), 0.12)
    for i, did in enumerate(datasets):
        sub = summary[summary["dataset_id"] == did].set_index("sample_budget")
        ys = [float(sub.loc[b, "delta_sentinel_quality"]) if b in sub.index else float("nan") for b in budgets]
        offset = (i - len(datasets) / 2) * width + width / 2
        ax.bar([xi + offset for xi in x], ys, width=width * 0.95, label=did, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.set_xlabel("Sample budget")
    ax.set_ylabel("Δ sentinel quality (stratified − random)")
    ax.set_title("Stress sweep: stratified minus random")
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.25))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "delta_sentinel_stratified_minus_random.png", dpi=160)
    plt.close(fig)

    # Heatmap-style: dataset × budget
    pivot = summary.pivot_table(
        index="dataset_id",
        columns="sample_budget",
        values="delta_sentinel_quality",
        aggfunc="mean",
    )
    fig2, ax2 = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.45)))
    im = ax2.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels(list(pivot.index), fontsize=8)
    ax2.set_xlabel("Sample budget")
    ax2.set_ylabel("Stress dataset")
    ax2.set_title("Δ sentinel quality (green=stratified higher)")
    fig2.colorbar(im, ax=ax2, shrink=0.6)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "delta_sentinel_heatmap.png", dpi=160)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "stress_test_dataset_analysis",
        help="Analysis output root (manifest, pooled CSVs, figures).",
    )
    parser.add_argument(
        "--case-start",
        type=int,
        default=0,
        help="0-based index into DEFAULT_CASES; use with --max-cases to slice cases (enables parallel shard sweeps).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=8,
        help="Take this many cases beginning at --case-start (default: first N cases).",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        nargs="*",
        default=None,
        metavar="ID",
        help=(
            "Run only these dataset_id entries from DEFAULT_CASES (preserves order of IDs listed). "
            "Overrides --case-start and --max-cases."
        ),
    )
    parser.add_argument("--source-papers", type=str, default="papers")
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--n-docs", type=int, default=None, help="Override n_docs for every case.")
    parser.add_argument("--train-n", type=int, default=8)
    parser.add_argument("--eval-n", type=int, default=24)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument(
        "--exp-seeds",
        type=int,
        nargs="+",
        default=[42, 43],
        help="One or more experiment seeds passed to run_sentinel_sampling_ab.",
    )
    parser.add_argument("--strata", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--j", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--skip-ab",
        action="store_true",
        help="Only generate pools + manifest; do not run sentinel A/B (offline smoke).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing pools/runs under out-dir before starting.",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out_root = args.out_dir
    if not str(out_root.resolve()).startswith(str(repo)):
        print("error: --out-dir must live inside --repo-root", file=sys.stderr)
        sys.exit(2)

    pools_root, runs_root, fig_dir = _ensure_dirs(out_root)
    if args.clean:
        shutil.rmtree(pools_root, ignore_errors=True)
        shutil.rmtree(runs_root, ignore_errors=True)
        for p in fig_dir.glob("*.png"):
            p.unlink(missing_ok=True)
        pools_root.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)

    mod = _load_generator()
    if args.dataset_ids:
        wanted = [x.strip() for x in args.dataset_ids if x.strip()]
        if not wanted:
            print("error: --dataset-ids is empty", file=sys.stderr)
            sys.exit(2)
        order = {cid: i for i, cid in enumerate(wanted)}
        cases = [c for c in DEFAULT_CASES if c.dataset_id in order]
        cases.sort(key=lambda c: order[c.dataset_id])
        missing_ids = set(wanted) - {c.dataset_id for c in cases}
        if missing_ids:
            print(f"error: unknown dataset_id(s): {sorted(missing_ids)}", file=sys.stderr)
            sys.exit(2)
    else:
        lo = max(0, int(args.case_start))
        hi = lo + int(args.max_cases)
        cases = list(DEFAULT_CASES[lo:hi])
        if not cases:
            print("error: --case-start/--max-cases produced an empty case list", file=sys.stderr)
            sys.exit(2)
    if args.n_docs is not None:
        cases = [
            SweepCase(c.dataset_id, c.preset, c.gen_seed, args.n_docs, dict(c.stress_overrides), c.notes) for c in cases
        ]

    manifest_rows: list[dict] = []
    all_rows: list[pd.DataFrame] = []
    failures: list[dict] = []

    for case in cases:
        try:
            summ = _generate_pool(mod, repo, out_root, case, args.source_papers, args.link_mode)
        except Exception as exc:
            failures.append({"dataset_id": case.dataset_id, "phase": "generate", "error": str(exc)})
            continue

        manifest_rows.append(
            {
                "dataset_id": case.dataset_id,
                "preset": summ["preset"],
                "gen_seed": summ["seed"],
                "n_docs": summ["n_docs"],
                "papers_dir_rel": summ["papers_dir_rel"],
                "features_csv_rel": summ["features_csv_rel"],
                "link_mode": summ["link_mode"],
                "notes": case.notes,
                "stress_params_json": json.dumps(summ.get("stress_params"), sort_keys=True),
                "domain_counts_json": json.dumps(summ.get("domain_counts"), sort_keys=True),
            }
        )

        if args.skip_ab:
            continue

        for exp_seed in args.exp_seeds:
            run_csv = runs_root / f"{case.dataset_id}_seed{int(exp_seed)}_ab.csv"
            rc = _run_ab(
                repo,
                papers_rel=summ["papers_dir_rel"],
                features_rel=summ["features_csv_rel"],
                out_csv=run_csv,
                train_n=min(args.train_n, summ["n_docs"]),
                eval_n=min(args.eval_n, summ["n_docs"]),
                budgets=args.budgets,
                exp_seed=int(exp_seed),
                strata=args.strata,
                k=args.k,
                j=args.j,
                max_workers=args.max_workers,
                no_progress=args.no_progress,
            )
            if rc != 0:
                failures.append(
                    {
                        "dataset_id": case.dataset_id,
                        "exp_seed": int(exp_seed),
                        "phase": "ab_run",
                        "error": f"exit_code={rc}",
                    }
                )
                continue

            if run_csv.is_file():
                part = pd.read_csv(run_csv)
                part.insert(0, "dataset_id", case.dataset_id)
                part["exp_seed"] = int(exp_seed)
                part["case_notes"] = case.notes
                part["stress_preset"] = case.preset
                part["pool_gen_seed"] = case.gen_seed
                part["stress_overrides_json"] = json.dumps(case.stress_overrides, sort_keys=True)
                all_rows.append(part)

    pd.DataFrame(manifest_rows).to_csv(out_root / "stress_dataset_manifest.csv", index=False)

    ab_settings: dict = {
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "budgets": args.budgets,
        "exp_seeds": args.exp_seeds,
        "strata": args.strata,
        "k": args.k,
        "j": args.j,
        "max_workers": args.max_workers,
        "fields_json": SWEEP_FIELDS_JSON,
    }
    if args.dataset_ids:
        ab_settings["dataset_ids"] = [x.strip() for x in args.dataset_ids if x.strip()]
    else:
        ab_settings["case_start"] = max(0, int(args.case_start))
        ab_settings["max_cases"] = int(args.max_cases)
    if args.n_docs is not None:
        ab_settings["n_docs_override"] = int(args.n_docs)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "cases_requested": len(cases),
        "cases_manifested": len(manifest_rows),
        "ab_runs_completed": len(all_rows),
        "failures": failures,
        "ab_settings": ab_settings,
    }
    (out_root / "sweep_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        big.to_csv(out_root / "stress_ab_all_runs.csv", index=False)
        sig = _significance_summary(big)
        sig.to_csv(out_root / "stress_significance_summary.csv", index=False)
        summ = _paired_deltas(big)
        if not summ.empty:
            summ.to_csv(out_root / "stress_summary_stratified_minus_random.csv", index=False)
            wins = (
                summ.groupby("dataset_id")["delta_sentinel_quality"]
                .agg(lambda s: float((s > 0).sum()))
                .rename("n_budgets_strat_wins")
                .reset_index()
                .merge(
                    summ.groupby("dataset_id")["delta_sentinel_quality"].mean().rename("mean_delta_sentinel").reset_index(),
                    on="dataset_id",
                )
            )
            wins.to_csv(out_root / "stress_win_counts.csv", index=False)
            _plot_deltas(summ, fig_dir)
    else:
        pd.DataFrame(columns=["dataset_id", "sample_budget", "delta_sentinel_quality"]).to_csv(
            out_root / "stress_summary_stratified_minus_random.csv",
            index=False,
        )

    if failures:
        pd.DataFrame(failures).to_csv(out_root / "stress_failures.csv", index=False)

    print(f"Wrote analysis under {out_root}")
    if failures:
        print(f"{len(failures)} failure(s); see stress_failures.csv", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
