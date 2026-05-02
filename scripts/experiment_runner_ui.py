#!/usr/bin/env python3
"""
Simple Streamlit UI for running sentinel A/B sampling experiments.

Run:
    streamlit run scripts/experiment_runner_ui.py
"""

from __future__ import annotations

import functools
import importlib.util
import io
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


STRAT_FEATURE_COLUMNS = [
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
    "domain",
]

TRAIN_SELECTION_OPTIONS = ["prefix", "random", "stratified"]
TRAIN_SKEW_OPTIONS = [
    "natural",
    "balanced_domain",
    "min_one_per_domain",
    "focus_domain",
    "custom_domain_ratios",
]


def parse_int_list(raw: str) -> list[int]:
    values = [v.strip() for v in raw.replace(",", " ").split() if v.strip()]
    return [int(v) for v in values]


def parse_text_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.replace(",", " ").split() if v.strip()]


def _pid_is_alive(pid_val: object) -> bool:
    try:
        pid = int(pid_val)
        if pid <= 0:
            return False
    except (TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


DEFAULT_FIELDS_JSON = json.dumps(
    [
        {
            "name": "reported_main_metric",
            "type": "str",
            "desc": (
                "The primary numeric result the paper emphasizes for its main claim "
                "(e.g. F1=0.62, AUROC=0.91, error 3.2%). If multiple metrics appear, give only the one "
                "highlighted as main in the abstract or introduction. If none is clearly numeric, answer "
                "exactly none_stated."
            ),
        },
        {
            "name": "metric_evidence_quote",
            "type": "str",
            "desc": (
                "Verbatim at most 25 words copied from the paper that contains the number or token from "
                "reported_main_metric (same sentence or immediately adjacent). If you cannot find a "
                "supporting span, answer exactly not_found."
            ),
        },
        {
            "name": "primary_baseline_name",
            "type": "str",
            "desc": (
                "The single main baseline the authors compare against in the central experiments "
                "(proper noun or method name). If unclear, answer exactly unclear."
            ),
        },
    ],
    indent=2,
)


QUICK_GRAPH_FIELDS_JSON = json.dumps(
    [
        {
            "name": "reported_main_metric",
            "type": "str",
            "desc": "Primary numeric main result (e.g. F1=0.62); none_stated if none.",
        },
    ],
    indent=2,
)

MEDIUM_FIELDS_JSON = json.dumps(
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
            "name": "uses_experiments",
            "type": "bool",
            "desc": "True if empirical experiments are reported.",
        },
    ],
    indent=2,
)

FIELD_PRESETS: dict[str, str] = {
    "Easy (1 field, fast smoke test)": QUICK_GRAPH_FIELDS_JSON,
    "Medium (3 fields, balanced)": MEDIUM_FIELDS_JSON,
    "Hard (3 strict fields, report-grade)": DEFAULT_FIELDS_JSON,
}

FIGSIZE_LINE = (4.8, 2.8)
FIGSIZE_BAR = (4.8, 2.8)
FIGSIZE_WIN = (4.8, 2.8)

# Top segmented-control label (was "Stratifier bench"): curate synthetic stratification tables.
SYNTHETIC_POOLS_TAB = "Synthetic pools"

SYNTHESIS_PRESET_LABELS = {
    "rare_numeric_tail": "Rare “long paper” tail (word_count)—most docs look alike, few are extremes",
    "domain_heavy_skew": "One common research domain vs a scarce second domain",
    "bimodal_sections": "Mostly low section-count papers, minority with huge section counts",
    "ultra_rare_domain_tail": "Needle-in-haystack rare tail (<5%) with extreme outliers",
    "adversarial_feature_conflict": "Adversarial conflicting feature signals across two regimes",
}

EVAL_CORPUS_INTRO_MARKDOWN = """
**Papers directory** (below in the experiment form)

Folder the job scans recursively for `.pdf` files. Every PDF matched here is eligible for optimization and scoring.

**Features CSV**

A table with columns like `path`, `word_count`, `domain`, etc.—one row per PDF. The **stratified** sampler sorts and buckets documents **using this file only**. For real workloads you usually build it with `extract_features.py`. For deliberate stress tests you can instead use a synthetic table that does **not** match PDF text—only filenames must line up.

**Synthetic pool builder** (button)

Samples PDFs from a tree you trust (typically your real corpus), duplicates or symlink-copies them into a dedicated folder, and writes fresh **skewed** feature rows (`paper_features.csv`) so minority slices of the corpus are scientifically rare but still reachable at low sample budgets. After **Generate**, the form’s **Papers directory** + **Features CSV** point at that synthetic pool automatically.

Tip: finish tuning on the dedicated **Synthetic pools** tab when the popup feels cramped.

For the full meaning of sliders and JSON knobs, open **What each synthetic parameter means** inside the builder.
"""

STRESS_POOL_PARAM_GUIDE = """
**Preset (recipe)** — Chooses how fake feature rows differ from each other. Each recipe only uses its own sliders; switch preset to reshape a different dimension of imbalance.

---

**Rare “long paper” tail** (`rare_numeric_tail`)

- **Tail fraction** — Share of docs that belong to the *minority* regime (normally very high `word_count` and different layout stats). Larger values make the imbalance less extreme numerically but still two separate clouds.
- **Tail domain** — Reported label for minority rows only (the stratifier may treat categorical `domain` as its own strata).
- **Bulk domains** — Labels rotated among typical “bulk” docs; widen variety in the dense cloud.
- **Bulk word low / high** — Synthetic word-count interval for bulk papers only.
- **Tail word low / high** — Interval for minority papers—keep this **much higher** than bulk to widen the quantitative gap stratification can latch onto.

---

**Domain-heavy skew** (`domain_heavy_skew`)

- **Majority fraction** — Probability a row uses the dominant domain slice and its tighter numeric spreads; small remainder goes to minority slice.
- **Majority domain / Minority domain** — Which categorical labels dominate vs appear rarely (rarer minority usually makes stratified coverage matter more).

---

**Bimodal sections** (`bimodal_sections`)

- **Short-section fraction** — Majority of PDFs synthesized with modest `section_count`; remainder use the large-count band.
- **Short section low / high** — `section_count` span for bulk “short-structure” synthetic papers.
- **Long section low / high** — Span for minority “many-section” outliers.
- **Domains** — Allowed labels spun through both modes (mixes categorical variety without changing preset kind).

---

**Ultra rare domain tail** (`ultra_rare_domain_tail`)

- **Ultra-rare fraction** — Tiny minority share (default ~3%) used for extreme outlier rows.
- **Ultra-rare domain** — Categorical label attached to those extreme minority rows.
- **Ultra bulk domains** — Labels used across the dominant majority rows.
- **Ultra bulk/rare word ranges** — Numeric separation between normal and extreme rows.
- **Ultra-rare section range** — Extra structural exaggeration for minority outliers.

---

**Adversarial feature conflict** (`adversarial_feature_conflict`)

- **Conflict cohort-A fraction** — Share of rows in cohort A; rest are cohort B.
- **Cohort A/B domains** — Distinct labels per cohort.
- **Cohort A/B word ranges** — Opposing word-length regimes to make mixed feature signals harder.

---

**Output directory**

Where under the repo the tool writes `doc_0000.pdf`… and `paper_features.csv`. Regenerating wipes previous `doc_*.pdf` in that folder only.

---

**Random seed (pool)**

RNG for *which PDFs get sampled from the corpus* and *exact synthetic numbers/draw*, not the sentinel MAB experiment seed (`Seed` farther down the form—that one runs the A/B harness).

---

**Number of PDFs**

How many files end up in the synthetic pool (= rows in CSV). Larger pools diversify rare strata but lengthen runs.

---

**Source PDF tree**

Existing folder of real PDFs to sample from (`papers` recommended). Copies or symlinks are created; originals stay untouched.

---

**PDF storage mode**

`copy`: duplicate bytes (portable folders).  
`symlink`: instant, points back to originals (fast; moving originals later breaks paths).

---

**Set Eval N to pool size**

When checked, the Run form “Eval N” becomes the synthetic pool length so sentinel never asks for PDFs absent from `paper_features.csv`.

---

**Optional JSON overrides**

Valid keys match `StressDatasetParams` inside `scripts/generate_stratifier_stress_dataset.py`; merged over slider values—for example `{\"tail_fraction\": 0.2}` adjusts tail share without redoing sliders.
"""


def _apply_quick_graph_test_defaults() -> None:
    """Set lightweight defaults intended only for fast plot smoke-tests."""
    st.session_state["train_selection"] = "prefix"
    st.session_state["train_skew"] = "natural"
    st.session_state["papers"] = "papers"
    st.session_state["features_csv"] = "papers/paper_features.csv"
    st.session_state["output_csv"] = "sampling-project/results/sampling-results/runs/ab_results_smoke.csv"
    st.session_state["train_n"] = 4
    st.session_state["eval_n_raw"] = "6"
    st.session_state["budgets_raw"] = "2 4"
    st.session_state["seed"] = 42
    st.session_state["strata"] = 4
    st.session_state["max_workers_raw"] = "2"
    st.session_state["k"] = 2
    st.session_state["j"] = 1
    st.session_state["strata_composition"] = "cartesian"
    st.session_state["stratify_features"] = STRAT_FEATURE_COLUMNS.copy()
    st.session_state["random_only"] = False
    st.session_state["stratified_only"] = False
    st.session_state["no_progress"] = False
    st.session_state["fields_json_raw"] = QUICK_GRAPH_FIELDS_JSON


def _init_session_defaults() -> None:
    defaults: dict[str, object] = {
        "train_selection": "prefix",
        "train_selection_strata": 8,
        "train_selection_features": STRAT_FEATURE_COLUMNS.copy(),
        "train_skew": "natural",
        "train_skew_focus_domain": "",
        "train_skew_domain_ratios": "",
        "papers": "papers",
        "features_csv": "papers/paper_features.csv",
        "output_csv": "sampling-project/results/sampling-results/runs/ab_results.csv",
        "train_n": 20,
        "eval_n_raw": "20",
        "budgets_raw": "5 10 15 20",
        "seed": 42,
        "strata": 8,
        "max_workers_raw": "2",
        "models_raw": "",
        "k": 6,
        "j": 4,
        "strata_composition": "cartesian",
        "stratify_features": STRAT_FEATURE_COLUMNS.copy(),
        "random_only": False,
        "stratified_only": False,
        "no_progress": False,
        "fields_json_raw": DEFAULT_FIELDS_JSON,
        "queue_max_parallel": 2,
        "fields_preset_name": "Hard (3 strict fields, report-grade)",
        "fields_preset_last_applied": "Hard (3 strict fields, report-grade)",
        "output_append_timestamp": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _read_results_csv(repo_root: Path, output_csv: str | None) -> pd.DataFrame | None:
    if not output_csv:
        return None
    csv_path = Path(output_csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "sample_budget",
        "optimization_cost",
        "optimization_time_s",
        "plan_execution_cost",
        "plan_execution_time_s",
        "total_cost",
        "total_time_s",
        "mean_sentinel_quality",
        "mean_plan_quality",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Enforce strict semantics: mean_plan_quality is valid only when sourced from true final-plan scoring.
    if "mean_plan_quality" in df.columns:
        if "plan_quality_source" in df.columns:
            valid_final = df["plan_quality_source"].astype(str).str.strip().str.lower() == "final_plan"
            df.loc[~valid_final, "mean_plan_quality"] = pd.NA
        else:
            # Older CSVs without provenance are treated as unknown and excluded from final-plan quality analysis.
            df["mean_plan_quality"] = pd.NA
    return df


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _render_and_download(fig: plt.Figure, label: str, filename: str) -> None:
    png = _fig_to_png_bytes(fig)
    st.pyplot(fig, width="content")
    plt.close(fig)
    st.download_button(label, data=png, file_name=filename, mime="image/png")


def _graph_heading(title: str, chart_description: str) -> None:
    # Keep chart guidance visible in-line; hover-only help was unreliable.
    st.markdown(f"**{title}**")
    st.caption(chart_description)


def _plot_line_by_mode(df: pd.DataFrame, y_col: str, title: str, y_label: str) -> bytes:
    fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
    style_by_mode = {
        "random": {"linestyle": "--", "marker": "o"},
        "stratified": {"linestyle": "-", "marker": "s"},
    }
    plotted: list[pd.Series] = []
    for mode, grp in df.groupby("mode", dropna=False):
        g = grp.sort_values("sample_budget")
        mode_key = str(mode).lower()
        style = style_by_mode.get(mode_key, {"linestyle": "-", "marker": "o"})
        ax.plot(
            g["sample_budget"],
            g[y_col],
            linewidth=2,
            markersize=7,
            label=str(mode),
            alpha=0.9,
            **style,
        )
        plotted.append(g[y_col].reset_index(drop=True))
    ax.set_xlabel("Sample budget")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # If all lines are flat/overlapping, force a small y-range so they remain visible.
    yvals = pd.to_numeric(df[y_col], errors="coerce").dropna()
    if not yvals.empty and float(yvals.max() - yvals.min()) == 0.0:
        center = float(yvals.iloc[0])
        pad = 0.05 if abs(center) < 1e-9 else max(abs(center) * 0.05, 0.01)
        ax.set_ylim(center - pad, center + pad)
        ax.text(
            0.02,
            0.95,
            "Random and stratified overlap at this scale.",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            alpha=0.8,
        )
    png = _fig_to_png_bytes(fig)
    st.pyplot(fig, width="content")
    plt.close(fig)
    return png


def _choose_quality_metric(df: pd.DataFrame) -> tuple[str | None, str]:
    if "mean_plan_quality" in df.columns and df["mean_plan_quality"].notna().any():
        return "mean_plan_quality", "Mean plan quality"
    if "mean_sentinel_quality" in df.columns and df["mean_sentinel_quality"].notna().any():
        return "mean_sentinel_quality", "Mean sentinel quality"
    return None, "Quality"


def _paired_budget_deltas(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    mode_col = df["mode"].astype(str).str.lower()
    pair = df[mode_col.isin(["random", "stratified"])].copy()
    if pair.empty:
        return pd.DataFrame()
    piv = pair.pivot_table(index="sample_budget", columns="mode", values=metric_col, aggfunc="mean")
    if "random" not in piv.columns or "stratified" not in piv.columns:
        return pd.DataFrame()
    out = piv.reset_index()
    out["delta"] = out["stratified"] - out["random"]
    return out.sort_values("sample_budget")


def _render_results_analysis(repo_root: Path, output_csv: str | None) -> None:
    df = _read_results_csv(repo_root, output_csv)
    if df is None or df.empty:
        return

    st.markdown("---")
    st.subheader("Experiment plots")

    csv_path = Path(output_csv) if output_csv else None
    if csv_path is not None and not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    if csv_path and csv_path.is_file():
        st.download_button(
            "Download results CSV",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
        )

    quality_metrics: list[tuple[str, str, str]] = []
    if "mean_plan_quality" in df.columns and df["mean_plan_quality"].notna().any():
        quality_metrics.append(("mean_plan_quality", "Mean final-plan quality", "final_plan"))
    if "mean_sentinel_quality" in df.columns and df["mean_sentinel_quality"].notna().any():
        quality_metrics.append(("mean_sentinel_quality", "Mean sentinel quality", "sentinel"))
    time_delta = _paired_budget_deltas(df.dropna(subset=["total_time_s"]), "total_time_s") if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns) else pd.DataFrame()
    final_source_available = "plan_quality_source" in df.columns
    if any(q[0] == "mean_plan_quality" for q in quality_metrics) and not final_source_available:
        st.warning("Final-plan quality source labels are unavailable in this CSV; strict final-only filtering cannot be enforced.")

    # Compact baseline-vs-stratified summary first.
    if not time_delta.empty:
        denom = time_delta["random"].replace(0, pd.NA)
        speedup_series = (time_delta["random"] - time_delta["stratified"]) / denom * 100.0
        mean_speedup = float(speedup_series.mean()) if speedup_series.notna().any() else 0.0
        faster_budgets = int((time_delta["delta"] < 0).sum())
        total_budgets = int(len(time_delta))
        c1, c2 = st.columns(2)
        c1.metric("Stratified faster budgets", f"{faster_budgets}/{total_budgets}")
        c2.metric("Avg runtime speedup", f"{mean_speedup:.1f}%")
    for q_col, q_label, q_slug in quality_metrics:
        if {"sample_budget", "mode", q_col}.issubset(df.columns):
            source_df = df.copy()
            if q_slug == "final_plan" and "plan_quality_source" in source_df.columns:
                source_df = source_df[source_df["plan_quality_source"].astype(str) == "final_plan"].copy()
            qd = _paired_budget_deltas(source_df.dropna(subset=[q_col]), q_col)
            if not qd.empty:
                q_wins = int((qd["delta"] > 0).sum())
                q_total = int(len(qd))
                st.metric(f"Stratified quality wins ({q_label})", f"{q_wins}/{q_total}")

    left_col, right_col = st.columns(2)
    panel = 0

    def _next_col():
        nonlocal panel
        col = left_col if panel % 2 == 0 else right_col
        panel += 1
        return col

    # Plot quality curves/deltas/errors for each available quality metric.
    for q_col, q_label, q_slug in quality_metrics:
        if not {"sample_budget", "mode", q_col}.issubset(df.columns):
            continue
        qdf = df.copy()
        if q_slug == "final_plan" and "plan_quality_source" in qdf.columns:
            qdf = qdf[qdf["plan_quality_source"].astype(str) == "final_plan"].copy()
        qdf = qdf.dropna(subset=[q_col]).copy()
        if qdf.empty:
            if q_slug == "final_plan":
                with _next_col():
                    st.info("No true final-plan quality rows for this run (only fallback or missing).")
            continue
        with _next_col():
            _graph_heading(
                f"{q_label} vs sample budget",
                "What it shows: average output quality at each budget for random vs stratified sampling. How to use it: if the stratified line is higher earlier, it reaches good quality with fewer samples.",
            )
            p1 = _plot_line_by_mode(
                qdf,
                q_col,
                f"{q_label} vs sample budget",
                q_label,
            )
            st.download_button(
                f"Download {q_label} plot (PNG)",
                data=p1,
                file_name=f"{q_slug}_quality_vs_budget.png",
                mime="image/png",
            )

        ref = (
            qdf.sort_values("sample_budget")
            .groupby("mode", as_index=False)
            .tail(1)
            .loc[:, ["mode", q_col]]
            .rename(columns={q_col: "ref_quality"})
        )
        err_df = qdf.merge(ref, on="mode", how="left")
        err_df["abs_quality_error"] = (err_df[q_col] - err_df["ref_quality"]).abs()
        with _next_col():
            _graph_heading(
                f"Quality estimation error vs sample budget ({q_label.lower()})",
                "What it shows: how far each budget is from that mode's highest-budget quality in this run. How to use it: lower values mean the estimate is stabilizing sooner.",
            )
            p2 = _plot_line_by_mode(
                err_df,
                "abs_quality_error",
                f"Absolute quality error vs sample budget ({q_label.lower()})",
                "|quality - quality@max_budget|",
            )
            st.download_button(
                f"Download {q_label} error plot (PNG)",
                data=p2,
                file_name=f"{q_slug}_quality_error_vs_budget.png",
                mime="image/png",
            )

        ddf = _paired_budget_deltas(qdf, q_col)
        if not ddf.empty:
            with _next_col():
                _graph_heading(
                    f"Stratified quality lift vs random (by budget, {q_label.lower()})",
                    "What it shows: stratified quality minus random quality at each budget. How to use it: bars above zero mean stratified performed better.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
                colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ddf["delta"].tolist()]
                ax.bar(
                    ddf["sample_budget"].astype(int).astype(str),
                    ddf["delta"],
                    color=colors,
                    alpha=0.55,
                    edgecolor="#333333",
                    linewidth=1.2,
                )
                ax.axhline(0.0, color="black", linewidth=1)
                ax.set_xlabel("Sample budget")
                ax.set_ylabel(f"Delta {q_label} (stratified - random)")
                ax.set_title("Quality lift by budget")
                ax.grid(True, axis="y", alpha=0.3)
                if (ddf["delta"] == 0).all():
                    ax.set_ylim(-0.05, 0.05)
                    ax.text(0.02, 0.95, "No quality difference vs random in this run.", transform=ax.transAxes, va="top", fontsize=8)
                _render_and_download(fig, f"Download {q_label} lift plot (PNG)", f"{q_slug}_quality_lift_vs_random.png")

    # Plot 4: total cost vs budget (random baseline vs stratified).
    if {"sample_budget", "mode", "total_cost"}.issubset(df.columns):
        cst = df.dropna(subset=["total_cost"]).copy()
        if not cst.empty:
            with _next_col():
                _graph_heading(
                    "Total cost vs sample budget (baseline comparison)",
                    "What it shows: total run cost (optimization plus execution) by budget for both methods. How to use it: the lower line is cheaper.",
                )
                p_cost = _plot_line_by_mode(
                    cst,
                    "total_cost",
                    "Total cost vs sample budget",
                    "Total cost",
                )
                st.download_button(
                    "Download total cost plot (PNG)",
                    data=p_cost,
                    file_name="total_cost_vs_budget.png",
                    mime="image/png",
                )

    # Plot 5: cost delta (stratified - random) by budget.
    if {"sample_budget", "mode", "total_cost"}.issubset(df.columns):
        cost_delta = _paired_budget_deltas(df.dropna(subset=["total_cost"]), "total_cost")
        if not cost_delta.empty:
            with _next_col():
                _graph_heading(
                    "Cost delta vs random (by budget)",
                    "What it shows: stratified cost minus random cost for each budget. How to use it: bars below zero mean stratified saved money.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
                colors = ["#2ca02c" if v <= 0 else "#d62728" for v in cost_delta["delta"].tolist()]
                ax.bar(
                    cost_delta["sample_budget"].astype(int).astype(str),
                    cost_delta["delta"],
                    color=colors,
                    alpha=0.55,
                    edgecolor="#333333",
                    linewidth=1.2,
                )
                ax.axhline(0.0, color="black", linewidth=1)
                ax.set_xlabel("Sample budget")
                ax.set_ylabel("Delta cost (stratified - random)")
                ax.set_title("Cost improvement by budget (lower is better)")
                ax.grid(True, axis="y", alpha=0.3)
                if (cost_delta["delta"] == 0).all():
                    ax.set_ylim(-0.05, 0.05)
                    ax.text(0.02, 0.95, "No cost difference vs random in this run.", transform=ax.transAxes, va="top", fontsize=8)
                _render_and_download(fig, "Download cost delta plot (PNG)", "cost_delta_vs_random.png")

    # Plot 6: runtime vs budget
    if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns):
        tdf = df.dropna(subset=["total_time_s"])
        if not tdf.empty:
            with _next_col():
                _graph_heading(
                    "Total runtime vs sample budget",
                    "What it shows: total wall-clock runtime by budget for both methods. How to use it: the lower line is faster.",
                )
                p4 = _plot_line_by_mode(
                    tdf,
                    "total_time_s",
                    "Total runtime vs sample budget",
                    "Total runtime (seconds)",
                )
                st.download_button(
                    "Download runtime plot (PNG)",
                    data=p4,
                    file_name="runtime_vs_budget.png",
                    mime="image/png",
                )

    # Plot 7: stratified speedup over random at each budget.
    if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns):
        if not time_delta.empty:
            denom = time_delta["random"].replace(0, pd.NA)
            time_delta["speedup_pct"] = (time_delta["random"] - time_delta["stratified"]) / denom * 100.0
            with _next_col():
                _graph_heading(
                    "Stratified runtime speedup vs random (by budget)",
                    "What it shows: runtime percent difference versus random at each budget. How to use it: values above zero mean stratified was faster.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
                colors = ["#2ca02c" if v >= 0 else "#d62728" for v in time_delta["speedup_pct"].tolist()]
                ax.bar(
                    time_delta["sample_budget"].astype(int).astype(str),
                    time_delta["speedup_pct"],
                    color=colors,
                    alpha=0.55,
                    edgecolor="#333333",
                    linewidth=1.2,
                )
                ax.axhline(0.0, color="black", linewidth=1)
                ax.set_xlabel("Sample budget")
                ax.set_ylabel("Speedup (%) = (random - stratified) / random")
                ax.set_title("Runtime speedup by budget")
                ax.grid(True, axis="y", alpha=0.3)
                _render_and_download(fig, "Download runtime speedup plot (PNG)", "runtime_speedup_vs_random.png")

    # Plot 8: optimization coverage metrics from sentinel phase.
    st.markdown("---")
    st.markdown("### Optimization Coverage Metrics")
    metric_specs = [
        (
            "total_sampled_records",
            "Total sampled records vs sample budget",
            "Total sampled records",
            "total_sampled_records",
            "What it shows: number of unique sampled records used by sentinel optimization at each budget. How to use it: higher values mean broader sampling coverage.",
        ),
        (
            "quality_scored_records",
            "Quality-scored records vs sample budget",
            "Quality-scored records",
            "quality_scored_records",
            "What it shows: number of unique sampled records that received a quality score. How to use it: larger counts mean stronger quality-estimation signal.",
        ),
        (
            "candidate_ops_explored",
            "Candidate operators explored vs sample budget",
            "Candidate ops explored",
            "candidate_ops_explored",
            "What it shows: how many operator candidates got sampled evidence during optimization. How to use it: higher values indicate broader exploration.",
        ),
        (
            "candidate_ops_pruned_estimate",
            "Candidate operators pruned estimate vs sample budget",
            "Candidate ops pruned (estimate)",
            "candidate_ops_pruned_estimate",
            "What it shows: estimated count of candidates not explored with sampled evidence. How to use it: higher values suggest stronger narrowing/focus of search.",
        ),
    ]
    for metric_col, title, y_label, slug, description in metric_specs:
        if not {"sample_budget", "mode", metric_col}.issubset(df.columns):
            continue
        mdf = df.dropna(subset=[metric_col]).copy()
        if mdf.empty:
            continue
        c_metric_line, c_metric_delta = st.columns(2)
        with c_metric_line:
            _graph_heading(title, description)
            metric_png = _plot_line_by_mode(
                mdf,
                metric_col,
                title,
                y_label,
            )
            st.download_button(
                f"Download {slug} plot (PNG)",
                data=metric_png,
                file_name=f"{slug}_vs_budget.png",
                mime="image/png",
            )
        with c_metric_delta:
            delta_df = _paired_budget_deltas(mdf, metric_col)
            if delta_df.empty:
                st.caption("No paired random/stratified rows for delta view.")
            else:
                _graph_heading(
                    f"{y_label} delta vs random (by budget)",
                    "What it shows: stratified minus random for this metric at each budget. How to use it: positive means stratified is higher, negative means lower.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
                vals = delta_df["delta"].tolist()
                colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
                ax.bar(
                    delta_df["sample_budget"].astype(int).astype(str),
                    vals,
                    color=colors,
                    alpha=0.55,
                    edgecolor="#333333",
                    linewidth=1.2,
                )
                ax.axhline(0.0, color="black", linewidth=1)
                ax.set_xlabel("Sample budget")
                ax.set_ylabel(f"Delta {y_label} (stratified - random)")
                ax.grid(True, axis="y", alpha=0.3)
                _render_and_download(fig, f"Download {slug} delta plot (PNG)", f"{slug}_delta_vs_random.png")

    # Plot 9: head-to-head win rates as separate sentinel/final/runtime bars.
    td = _paired_budget_deltas(df.dropna(subset=["total_time_s"]), "total_time_s") if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns) else pd.DataFrame()
    sentinel_qd = _paired_budget_deltas(df.dropna(subset=["mean_sentinel_quality"]), "mean_sentinel_quality") if {"sample_budget", "mode", "mean_sentinel_quality"}.issubset(df.columns) else pd.DataFrame()
    if {"sample_budget", "mode", "mean_plan_quality"}.issubset(df.columns):
        final_df = df.copy()
        if "plan_quality_source" in final_df.columns:
            final_df = final_df[final_df["plan_quality_source"].astype(str) == "final_plan"].copy()
        final_qd = _paired_budget_deltas(final_df.dropna(subset=["mean_plan_quality"]), "mean_plan_quality")
    else:
        final_qd = pd.DataFrame()
    if not td.empty:
        labels: list[str] = []
        vals: list[float] = []
        colors: list[str] = []
        if not sentinel_qd.empty:
            labels.append("Sentinel quality win rate")
            vals.append(float((sentinel_qd["delta"] > 0).mean() * 100.0))
            colors.append("#1f77b4")
        if not final_qd.empty:
            labels.append("Final-plan quality win rate")
            vals.append(float((final_qd["delta"] > 0).mean() * 100.0))
            colors.append("#2ca02c")
        labels.append("Runtime win rate")
        vals.append(float((td["delta"] < 0).mean() * 100.0))
        colors.append("#ff7f0e")
        with _next_col():
            _graph_heading(
                "Stratified win rate across budgets",
                "What it shows: the percentage of tested budgets where stratified beats random, reported separately for sentinel quality, final-plan quality, and runtime.",
            )
            fig, ax = plt.subplots(figsize=FIGSIZE_WIN)
            bars = ax.bar(labels, vals, color=colors, alpha=0.9)
            for b, v in zip(bars, vals):
                y = v - 3 if v >= 10 else v + 1.5
                va = "top" if v >= 10 else "bottom"
                ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.1f}%", ha="center", va=va, fontsize=9)
            ax.set_ylim(0, 110)
            ax.set_ylabel("Win rate (%) vs random baseline")
            ax.set_title("Stratified win rate across budgets", pad=8)
            ax.grid(True, axis="y", alpha=0.3)
            _render_and_download(fig, "Download win-rate plot (PNG)", "win_rate_vs_random.png")

    # Render + download run tables directly inside the popup.
    st.markdown("---")
    st.markdown("### Run Tables")
    run_table = df.copy()
    show_cols = [
        c
        for c in [
            "sample_budget",
            "mode",
            "mean_sentinel_quality",
            "mean_plan_quality",
            "plan_quality_source",
            "total_sampled_records",
            "candidate_ops_explored",
            "candidate_ops_pruned_estimate",
            "quality_scored_records",
            "total_time_s",
            "total_cost",
        ]
        if c in run_table.columns
    ]
    if show_cols:
        st.dataframe(run_table[show_cols].sort_values(["sample_budget", "mode"]), width="stretch", hide_index=True)
        st.download_button(
            "Download run table (CSV)",
            data=run_table[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="run_popup_table.csv",
            mime="text/csv",
        )
    if {"sample_budget", "mode", "mean_sentinel_quality"}.issubset(df.columns):
        paired_q_s = _paired_budget_deltas(df.dropna(subset=["mean_sentinel_quality"]), "mean_sentinel_quality")
        if not paired_q_s.empty:
            st.markdown("**Paired quality deltas (sentinel quality, stratified - random)**")
            st.dataframe(paired_q_s, width="stretch", hide_index=True)
            st.download_button(
                "Download paired sentinel quality deltas (CSV)",
                data=paired_q_s.to_csv(index=False).encode("utf-8"),
                file_name="paired_sentinel_quality_deltas.csv",
                mime="text/csv",
            )
    if {"sample_budget", "mode", "mean_plan_quality"}.issubset(df.columns):
        final_df = df.copy()
        if "plan_quality_source" in final_df.columns:
            final_df = final_df[final_df["plan_quality_source"].astype(str) == "final_plan"].copy()
        paired_q_f = _paired_budget_deltas(final_df.dropna(subset=["mean_plan_quality"]), "mean_plan_quality")
        if not paired_q_f.empty:
            st.markdown("**Paired quality deltas (final-plan quality, stratified - random)**")
            st.dataframe(paired_q_f, width="stretch", hide_index=True)
            st.download_button(
                "Download paired final-plan quality deltas (CSV)",
                data=paired_q_f.to_csv(index=False).encode("utf-8"),
                file_name="paired_final_plan_quality_deltas.csv",
                mime="text/csv",
            )
    if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns):
        paired_t = _paired_budget_deltas(df.dropna(subset=["total_time_s"]), "total_time_s")
        if not paired_t.empty:
            st.markdown("**Paired runtime deltas (stratified - random)**")
            st.dataframe(paired_t, width="stretch", hide_index=True)
            st.download_button(
                "Download paired runtime deltas (CSV)",
                data=paired_t.to_csv(index=False).encode("utf-8"),
                file_name="paired_runtime_deltas.csv",
                mime="text/csv",
            )


def _clean_stdout(stdout: str) -> tuple[str, int]:
    """Strip ANSI noise and collapse repeated non-actionable runtime spam."""
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    cleaned = ansi_re.sub("", stdout or "")
    noise = "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
    box_chars = ("╭", "╰", "│", "━", "⠋")
    lines = cleaned.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if noise in line or "LiteLLM.Info: If you need to debug this error" in line:
            removed += 1
            continue
        if any(ch in line for ch in box_chars):
            removed += 1
            continue
        kept.append(line)
    # Drop repeated blank lines for readability.
    collapsed: list[str] = []
    prev_blank = False
    for line in kept:
        blank = len(line.strip()) == 0
        if blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = blank
    return ("\n".join(collapsed).strip(), removed)


def _summarize_stdout(stdout: str) -> str:
    """Extract concise run facts from cleaned stdout."""
    if not stdout.strip():
        return "(empty)"
    lines = stdout.splitlines()
    keep: list[str] = []
    in_table = False
    for line in lines:
        s = line.strip()
        if s.startswith("=== Run "):
            keep.append(s)
            continue
        if s.startswith("Papers root:") or s.startswith("Features CSV:") or s.startswith("budgets="):
            keep.append(s)
            continue
        if s.startswith("Total opt. time:") or s.startswith("Total opt. cost:"):
            keep.append(s)
            continue
        if s.startswith("Total time:") or s.startswith("Total cost:"):
            keep.append(s)
            continue
        if s.startswith("+--------+") or s.startswith("| budget ") or s.startswith("|      "):
            keep.append(s)
            in_table = True
            continue
        if in_table and s.startswith("+--------+"):
            keep.append(s)
            in_table = False
            continue
    if not keep:
        return stdout
    return "\n".join(keep)


def _clean_stderr(stderr: str) -> tuple[str, int]:
    """Strip ANSI noise and collapse repeated LiteLLM help spam."""
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    cleaned = ansi_re.sub("", stderr or "")
    noise = "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
    lines = cleaned.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        if noise in line or "LiteLLM.Info: If you need to debug this error" in line:
            removed += 1
            continue
        if "RuntimeWarning: Mean of empty slice." in line or "RuntimeWarning: invalid value encountered in scalar divide" in line:
            removed += 1
            continue
        if "/site-packages/numpy/_core/fromnumeric.py:" in line or "/site-packages/numpy/_core/_methods.py:" in line:
            removed += 1
            continue
        s = line.strip()
        if s.startswith("return _methods._mean(") or s.startswith("ret = ret.dtype.type(ret / rcount)"):
            removed += 1
            continue
        kept.append(line)
    return ("\n".join(kept).strip(), removed)


def _history_db_path(repo_root: Path) -> Path:
    return repo_root / "papers" / "experiment_history.sqlite3"


def _history_db_url() -> str | None:
    url = os.getenv("PALIMPZEST_HISTORY_DB_URL", "").strip()
    if not url:
        return None
    # Force SQLAlchemy to use psycopg (psycopg3), which is installed via psycopg[binary].
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _current_run_by() -> str:
    return (
        os.getenv("PALIMPZEST_RUN_BY", "").strip()
        or os.getenv("USER", "").strip()
        or os.getenv("USERNAME", "").strip()
        or "unknown"
    )


def _init_history_db(repo_root: Path) -> str:
    db_url = _history_db_url()
    if db_url:
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        command TEXT NOT NULL,
                        output_csv TEXT,
                        return_code INTEGER NOT NULL,
                        budgets TEXT,
                        train_n INTEGER,
                        eval_n INTEGER,
                        strata INTEGER,
                        k INTEGER,
                        j INTEGER,
                        notes TEXT
                    )
                    """
                )
            )
            for col_name, col_type in [
                ("seed", "INTEGER"),
                ("strata_composition", "TEXT"),
                ("stratify_features", "TEXT"),
                ("train_selection", "TEXT"),
                ("train_skew", "TEXT"),
                ("run_by", "TEXT"),
                ("run_duration_s", "DOUBLE PRECISION"),
            ]:
                conn.execute(text(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS run_rows (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL,
                        sample_budget INTEGER,
                        mode TEXT,
                        total_time_s DOUBLE PRECISION,
                        total_cost DOUBLE PRECISION,
                        mean_sentinel_quality DOUBLE PRECISION,
                        mean_plan_quality DOUBLE PRECISION,
                        plan_quality_source TEXT,
                        total_sampled_records DOUBLE PRECISION,
                        candidate_ops_explored DOUBLE PRECISION,
                        candidate_ops_pruned_estimate DOUBLE PRECISION,
                        quality_scored_records DOUBLE PRECISION
                    )
                    """
                )
            )
            conn.execute(text("ALTER TABLE run_rows ADD COLUMN IF NOT EXISTS plan_quality_source TEXT"))
            conn.execute(text("ALTER TABLE run_rows ADD COLUMN IF NOT EXISTS total_sampled_records DOUBLE PRECISION"))
            conn.execute(text("ALTER TABLE run_rows ADD COLUMN IF NOT EXISTS candidate_ops_explored DOUBLE PRECISION"))
            conn.execute(text("ALTER TABLE run_rows ADD COLUMN IF NOT EXISTS candidate_ops_pruned_estimate DOUBLE PRECISION"))
            conn.execute(text("ALTER TABLE run_rows ADD COLUMN IF NOT EXISTS quality_scored_records DOUBLE PRECISION"))
        return db_url

    db_path = _history_db_path(repo_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                command TEXT NOT NULL,
                output_csv TEXT,
                return_code INTEGER NOT NULL,
                budgets TEXT,
                train_n INTEGER,
                eval_n INTEGER,
                strata INTEGER,
                k INTEGER,
                j INTEGER,
                notes TEXT
            )
            """
        )
        # Backward-compatible schema evolution for older local DB files.
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(runs)")
        }
        for col_name, col_type in [
            ("seed", "INTEGER"),
            ("strata_composition", "TEXT"),
            ("stratify_features", "TEXT"),
            ("train_selection", "TEXT"),
            ("train_skew", "TEXT"),
            ("run_by", "TEXT"),
            ("run_duration_s", "REAL"),
        ]:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                sample_budget INTEGER,
                mode TEXT,
                total_time_s REAL,
                total_cost REAL,
                mean_sentinel_quality REAL,
                mean_plan_quality REAL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        row_cols = {row[1] for row in conn.execute("PRAGMA table_info(run_rows)")}
        if "plan_quality_source" not in row_cols:
            conn.execute("ALTER TABLE run_rows ADD COLUMN plan_quality_source TEXT")
        if "total_sampled_records" not in row_cols:
            conn.execute("ALTER TABLE run_rows ADD COLUMN total_sampled_records REAL")
        if "candidate_ops_explored" not in row_cols:
            conn.execute("ALTER TABLE run_rows ADD COLUMN candidate_ops_explored REAL")
        if "candidate_ops_pruned_estimate" not in row_cols:
            conn.execute("ALTER TABLE run_rows ADD COLUMN candidate_ops_pruned_estimate REAL")
        if "quality_scored_records" not in row_cols:
            conn.execute("ALTER TABLE run_rows ADD COLUMN quality_scored_records REAL")
    return str(db_path)


def _save_run_history(
    repo_root: Path,
    *,
    command: str,
    output_csv: str | None,
    return_code: int,
    budgets: list[int],
    train_n: int,
    eval_n: int | None,
    strata: int,
    k: int,
    j: int,
    seed: int,
    strata_composition: str,
    stratify_features: list[str],
    train_selection: str,
    train_skew: str,
) -> int:
    db_ref = _init_history_db(repo_root)
    csv_df = _read_results_csv(repo_root, output_csv)
    run_duration_s: float | None = None
    if csv_df is not None and not csv_df.empty and "total_time_s" in csv_df.columns:
        total_series = pd.to_numeric(csv_df["total_time_s"], errors="coerce").dropna()
        if not total_series.empty:
            run_duration_s = float(total_series.sum())
    vals = {
        "command": command,
        "output_csv": output_csv or "",
        "return_code": int(return_code),
        "budgets": " ".join(str(b) for b in budgets),
        "train_n": int(train_n),
        "eval_n": None if eval_n is None else int(eval_n),
        "strata": int(strata),
        "k": int(k),
        "j": int(j),
        "seed": int(seed),
        "strata_composition": strata_composition,
        "stratify_features": ",".join(stratify_features),
        "train_selection": train_selection,
        "train_skew": train_skew,
        "run_by": _current_run_by(),
        "run_duration_s": run_duration_s,
    }
    if db_ref.startswith("postgres"):
        engine = create_engine(db_ref)
        with engine.begin() as conn:
            run_id = int(
                conn.execute(
                    text(
                        """
                        INSERT INTO runs(
                            command, output_csv, return_code, budgets, train_n, eval_n, strata, k, j,
                            seed, strata_composition, stratify_features, train_selection, train_skew, run_by, run_duration_s
                        )
                        VALUES (
                            :command, :output_csv, :return_code, :budgets, :train_n, :eval_n, :strata, :k, :j,
                            :seed, :strata_composition, :stratify_features, :train_selection, :train_skew, :run_by, :run_duration_s
                        )
                        RETURNING run_id
                        """
                    ),
                    vals,
                ).scalar_one()
            )
            if csv_df is not None and not csv_df.empty:
                for _, row in csv_df.iterrows():
                    conn.execute(
                        text(
                            """
                            INSERT INTO run_rows(
                                run_id, sample_budget, mode, total_time_s, total_cost, mean_sentinel_quality, mean_plan_quality, plan_quality_source,
                                total_sampled_records, candidate_ops_explored, candidate_ops_pruned_estimate, quality_scored_records
                            )
                            VALUES (
                                :run_id, :sample_budget, :mode, :total_time_s, :total_cost, :mean_sentinel_quality, :mean_plan_quality, :plan_quality_source,
                                :total_sampled_records, :candidate_ops_explored, :candidate_ops_pruned_estimate, :quality_scored_records
                            )
                            """
                        ),
                        {
                            "run_id": run_id,
                            "sample_budget": int(row["sample_budget"]) if pd.notna(row.get("sample_budget")) else None,
                            "mode": str(row.get("mode", "")),
                            "total_time_s": float(row["total_time_s"]) if pd.notna(row.get("total_time_s")) else None,
                            "total_cost": float(row["total_cost"]) if pd.notna(row.get("total_cost")) else None,
                            "mean_sentinel_quality": float(row["mean_sentinel_quality"]) if pd.notna(row.get("mean_sentinel_quality")) else None,
                            "mean_plan_quality": float(row["mean_plan_quality"]) if pd.notna(row.get("mean_plan_quality")) else None,
                            "plan_quality_source": str(row.get("plan_quality_source", "")) if pd.notna(row.get("plan_quality_source")) else None,
                            "total_sampled_records": float(row["total_sampled_records"]) if pd.notna(row.get("total_sampled_records")) else None,
                            "candidate_ops_explored": float(row["candidate_ops_explored"]) if pd.notna(row.get("candidate_ops_explored")) else None,
                            "candidate_ops_pruned_estimate": float(row["candidate_ops_pruned_estimate"]) if pd.notna(row.get("candidate_ops_pruned_estimate")) else None,
                            "quality_scored_records": float(row["quality_scored_records"]) if pd.notna(row.get("quality_scored_records")) else None,
                        },
                    )
        return run_id
    with sqlite3.connect(db_ref) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs(
                command, output_csv, return_code, budgets, train_n, eval_n, strata, k, j,
                seed, strata_composition, stratify_features, train_selection, train_skew, run_by, run_duration_s
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(vals[k] for k in ["command","output_csv","return_code","budgets","train_n","eval_n","strata","k","j","seed","strata_composition","stratify_features","train_selection","train_skew","run_by","run_duration_s"]),
        )
        run_id = int(cur.lastrowid)
        if csv_df is not None and not csv_df.empty:
            rows = []
            for _, row in csv_df.iterrows():
                rows.append(
                    (
                        run_id,
                        int(row["sample_budget"]) if pd.notna(row.get("sample_budget")) else None,
                        str(row.get("mode", "")),
                        float(row["total_time_s"]) if pd.notna(row.get("total_time_s")) else None,
                        float(row["total_cost"]) if pd.notna(row.get("total_cost")) else None,
                        float(row["mean_sentinel_quality"]) if pd.notna(row.get("mean_sentinel_quality")) else None,
                        float(row["mean_plan_quality"]) if pd.notna(row.get("mean_plan_quality")) else None,
                        str(row.get("plan_quality_source", "")) if pd.notna(row.get("plan_quality_source")) else None,
                        float(row["total_sampled_records"]) if pd.notna(row.get("total_sampled_records")) else None,
                        float(row["candidate_ops_explored"]) if pd.notna(row.get("candidate_ops_explored")) else None,
                        float(row["candidate_ops_pruned_estimate"]) if pd.notna(row.get("candidate_ops_pruned_estimate")) else None,
                        float(row["quality_scored_records"]) if pd.notna(row.get("quality_scored_records")) else None,
                    )
                )
            conn.executemany(
                """
                INSERT INTO run_rows(
                    run_id, sample_budget, mode, total_time_s, total_cost, mean_sentinel_quality, mean_plan_quality, plan_quality_source,
                    total_sampled_records, candidate_ops_explored, candidate_ops_pruned_estimate, quality_scored_records
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return run_id


def _batch_manifest_path(repo_root: Path) -> Path:
    return repo_root / "results" / "batch_logs" / "batch_manifest.json"


def _load_batch_manifest(repo_root: Path) -> list[dict]:
    manifest_path = _batch_manifest_path(repo_root)
    if not manifest_path.is_file():
        return []
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _save_batch_manifest(repo_root: Path, rows: list[dict]) -> None:
    manifest_path = _batch_manifest_path(repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _job_state(repo_root: Path, row: dict) -> str:
    """Derive queue state with backward compatibility for older manifest rows."""
    state = str(row.get("state", "") or "").strip().lower()
    if state in {"queued", "running", "finished", "imported"}:
        if state == "running":
            status_file = str(row.get("status_file", "") or "").strip()
            if status_file and (repo_root / status_file).is_file():
                return "finished"
            # If the child shell process died before writing status_file,
            # still treat it as finished so it can be imported as a failed run.
            pid = row.get("pid")
            if pid is not None and not _pid_is_alive(pid):
                return "finished"
        if state == "finished" and row.get("history_imported"):
            return "imported"
        return state
    # Backward-compat for pre-state rows:
    if row.get("history_imported"):
        return "imported"
    status_file = str(row.get("status_file", "") or "").strip()
    if status_file and (repo_root / status_file).is_file():
        return "finished"
    return "running"


def _new_job_id() -> str:
    return str(int(time.time() * 1000))


def _queue_state_counts(repo_root: Path, rows: list[dict]) -> tuple[int, int, int]:
    queued = 0
    running = 0
    finished = 0
    for r in rows:
        s = _job_state(repo_root, r)
        if s == "queued":
            queued += 1
        elif s == "running":
            running += 1
        else:
            finished += 1
    return queued, running, finished


def _job_runtime_minutes(row: dict) -> float | None:
    started = row.get("started_at_epoch_s")
    if started is None:
        return None
    try:
        return max(0.0, (time.time() - float(started)) / 60.0)
    except Exception:
        return None


def _job_started_at_text(row: dict) -> str:
    started = row.get("started_at_epoch_s")
    if started is None:
        return "unknown"
    try:
        ts = time.localtime(float(started))
        return time.strftime("%Y-%m-%d %H:%M:%S", ts)
    except Exception:
        return "unknown"


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _stratified_passes_per_budget(strata_composition: str, stratify_features: list | None) -> int:
    feats = stratify_features if isinstance(stratify_features, list) else []
    if strata_composition == "exclusive":
        return max(1, len(feats))
    return 1


def _runs_per_budget_from_job(job: dict) -> int:
    cmd = str(job.get("command", "") or "")
    random_only = bool(job.get("random_only")) or "--random-only" in cmd
    stratified_only = bool(job.get("stratified_only")) or "--stratified-only" in cmd
    comp = str(job.get("strata_composition") or "composite")
    raw_feats = job.get("stratify_features")
    feats_list = raw_feats if isinstance(raw_feats, list) else None
    strat_passes = _stratified_passes_per_budget(comp, feats_list)
    n = 0
    if not stratified_only:
        n += 1
    if not random_only:
        n += strat_passes
    return max(1, n)


def _expected_ab_phases_from_job(job: dict) -> int:
    budgets = job.get("budgets")
    if isinstance(budgets, list) and budgets:
        n_budgets = len(budgets)
    else:
        cmd = str(job.get("command", "") or "")
        m = re.search(r"--budgets\s+([\d\s]+)", cmd)
        if m:
            n_budgets = len([p for p in m.group(1).split() if p.isdigit()])
        else:
            n_budgets = 1
    return max(1, n_budgets * _runs_per_budget_from_job(job))


def _read_file_tail_bytes(path: Path, max_bytes: int = 262_144) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size <= max_bytes:
                return f.read().decode("utf-8", errors="replace")
            f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _count_lines_matching(path: Path, pred) -> int:
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if pred(line):
                    n += 1
    except OSError:
        return 0
    return n


def _last_percent_from_text(text: str) -> float | None:
    plain = _strip_ansi(text)
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*%", plain))
    if not matches:
        return None
    v = float(matches[-1].group(1))
    if 0.0 <= v <= 100.0:
        return v
    return None


def _job_progress_bars_enabled(job: dict) -> bool:
    """False when --no-progress / MockProgressManager (no Total opt/time lines)."""
    np = job.get("no_progress")
    if np is True:
        return False
    if np is False:
        return True
    return "--no-progress" not in str(job.get("command", "") or "")


def _is_optimizer_finish_line(line: str) -> bool:
    """One line per sub-phase: sentinel finish, then final-plan finish (see QueryProcessor.execute)."""
    s = line.strip()
    return s.startswith("Total opt. time:") or s.startswith("Total time:")


def _job_progress_from_log(repo_root: Path, job: dict) -> tuple[float, str]:
    """
    Estimate overall completion ratio in [0, 1] from the job log and manifest.
    Returns (ratio, human-readable detail).
    """
    total_phases = _expected_ab_phases_from_job(job)
    progress_bars = _job_progress_bars_enabled(job)
    log_rel = str(job.get("log_file", "") or "").strip()
    if not log_rel:
        return 0.0, f"No log path (expected {total_phases} run phase(s))."
    path = repo_root / log_rel
    if not path.is_file():
        return 0.0, f"Log not created yet ({total_phases} phase(s) planned)."

    finish_lines = _count_lines_matching(path, _is_optimizer_finish_line)
    run_headers = _count_lines_matching(
        path, lambda ln: "=== Run A:" in ln or "=== Run B:" in ln
    )

    tail = _read_file_tail_bytes(path)
    tail_pct = _last_percent_from_text(tail)

    if progress_bars:
        # Each A/B "phase" runs sentinel optimization then final physical plan — two finish prints.
        finish_target = total_phases * 2
        if finish_lines >= finish_target:
            ratio = 1.0
            detail = (
                f"All {total_phases} phase(s) done ({finish_lines}/{finish_target} finish lines: "
                f"sentinel + final plan per phase)."
            )
            return ratio, detail

        within = 0.0
        if tail_pct is not None:
            within = tail_pct / 100.0
        elif run_headers > (finish_lines + 1) // 2:
            within = 0.05

        ratio = (finish_lines + within) / float(finish_target)
        ratio = max(0.0, min(1.0, ratio))
        remain = max(0.0, 1.0 - ratio)
        detail = (
            f"Finish lines in log: {finish_lines}/{finish_target} "
            f"(2 per phase: `Total opt. time` after sentinel, `Total time` after final plan). "
            f"Within current sub-step: ~{within * 100:.0f}% from Rich % in log tail. "
            f"~{remain * 100:.0f}% of total remaining (estimate)."
        )
        if finish_lines == 0 and run_headers == 0:
            detail = (
                f"Starting… ({total_phases} phase(s) planned, {finish_target} finish lines expected). "
                "Parsing log tail for Rich progress."
            )
        return ratio, detail

    # --no-progress: progress managers do not print finish lines; use Run A/B headers only.
    eff = max(0.0, float(run_headers) - 0.5)
    ratio = min(1.0, eff / float(total_phases))
    detail = (
        f"Progress bars disabled — estimating from `=== Run A/B ===` headers: "
        f"{run_headers}/{total_phases} started (~{ratio * 100:.0f}% overall, coarse)."
    )
    return ratio, detail


def _maybe_start_queued_jobs(repo_root: Path, max_parallel: int = 1) -> tuple[int, list[str]]:
    """
    Start queued jobs up to max_parallel active workers.
    Returns (num_started, messages).
    """
    rows = _load_batch_manifest(repo_root)
    if not rows:
        return 0, []
    max_parallel = max(1, int(max_parallel))
    running_now = sum(1 for r in rows if _job_state(repo_root, r) == "running")
    slots = max(0, max_parallel - running_now)
    if slots <= 0:
        return 0, []

    queued_indices = [i for i, r in enumerate(rows) if _job_state(repo_root, r) == "queued"][:slots]
    if not queued_indices:
        return 0, []

    logs_dir = repo_root / "results" / "batch_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = (
        "ignore:Mean of empty slice:RuntimeWarning,"
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    )
    messages: list[str] = []
    started = 0
    for next_idx in queued_indices:
        row = rows[next_idx]
        cmd = str(row.get("command", "") or "").strip()
        if not cmd:
            row["state"] = "finished"
            row["history_imported"] = True
            continue
        job_id = str(row.get("job_id", "") or _new_job_id())
        row["job_id"] = job_id
        suffix = f"{job_id}_seed{row.get('seed','na')}_{row.get('strata_composition','na')}_{row.get('train_selection','na')}_{row.get('train_skew','na')}"
        log_path = logs_dir / f"queue_{suffix}.log"
        status_path = logs_dir / f"queue_{suffix}.status"
        # Ensure stale files from prior runs never short-circuit state detection.
        if status_path.is_file():
            status_path.unlink(missing_ok=True)
        shell_cmd = f"{cmd} > {shlex.quote(str(log_path))} 2>&1; echo $? > {shlex.quote(str(status_path))}"
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", shell_cmd],
            cwd=str(repo_root),
            text=True,
            env=env,
        )
        row["pid"] = int(proc.pid)
        row["log_file"] = str(log_path.relative_to(repo_root))
        row["status_file"] = str(status_path.relative_to(repo_root))
        row["state"] = "running"
        row["started_at_epoch_s"] = time.time()
        started += 1
        messages.append(
            f"Started queued job pid={proc.pid} (seed={row.get('seed')}, comp={row.get('strata_composition')}, train={row.get('train_selection')}, skew={row.get('train_skew')})."
        )
    _save_batch_manifest(repo_root, rows)
    return started, messages


def _purge_queue_artifacts(repo_root: Path, row: dict) -> None:
    for key in ("log_file", "status_file"):
        rel = str(row.get(key, "") or "").strip()
        if not rel:
            continue
        p = repo_root / rel
        if p.is_file():
            p.unlink(missing_ok=True)


def _sync_completed_batch_runs(repo_root: Path) -> tuple[int, int]:
    rows = _load_batch_manifest(repo_root)
    if not rows:
        return 0, 0
    imported = 0
    pending = 0
    kept_rows: list[dict] = []
    for row in rows:
        state = _job_state(repo_root, row)
        row["state"] = state
        if state in {"queued", "running"}:
            kept_rows.append(row)
            pending += 1
            continue
        status_path = repo_root / str(row.get("status_file", ""))
        if status_path.is_file():
            try:
                return_code = int(status_path.read_text(encoding="utf-8").strip())
            except Exception:
                return_code = 1
        else:
            # No status file means the background wrapper likely died early.
            # Record as failed so it is visible in History and leaves the queue.
            return_code = 1
        _save_run_history(
            repo_root,
            command=str(row.get("command", "")),
            output_csv=str(row.get("output_csv", "")) or None,
            return_code=return_code,
            budgets=[int(x) for x in row.get("budgets", [])],
            train_n=int(row.get("train_n", 0)),
            eval_n=(None if row.get("eval_n") is None else int(row.get("eval_n"))),
            strata=int(row.get("strata", 0)),
            k=int(row.get("k", 0)),
            j=int(row.get("j", 0)),
            seed=int(row.get("seed", 0)),
            strata_composition=str(row.get("strata_composition", "")),
            stratify_features=[str(x) for x in row.get("stratify_features", [])],
            train_selection=str(row.get("train_selection", "")),
            train_skew=str(row.get("train_skew", "")),
        )
        row["history_imported"] = True
        row["return_code"] = return_code
        row["state"] = "imported"
        _purge_queue_artifacts(repo_root, row)
        imported += 1
    _save_batch_manifest(repo_root, kept_rows)
    return imported, pending


def _maybe_migrate_local_sqlite_to_remote(repo_root: Path) -> tuple[int, int]:
    """
    One-time migration helper:
    - If PALIMPZEST_HISTORY_DB_URL is set (remote mode)
    - And remote runs table is empty
    - And local sqlite has data
    then copy local runs/run_rows to remote.
    """
    db_url = _history_db_url()
    if not db_url:
        return 0, 0
    local_path = _history_db_path(repo_root)
    if not local_path.is_file():
        return 0, 0

    # Ensure remote schema exists first.
    _init_history_db(repo_root)
    engine = create_engine(db_url)
    with engine.begin() as rconn:
        remote_runs_count = int(rconn.execute(text("SELECT COUNT(*) FROM runs")).scalar() or 0)
        if remote_runs_count > 0:
            return 0, 0

    with sqlite3.connect(local_path) as lconn:
        local_runs = pd.read_sql_query(
            """
            SELECT
                run_id, created_at, command, output_csv, return_code, budgets, train_n, eval_n,
                strata, k, j, notes, seed, strata_composition, stratify_features, train_selection, train_skew, run_by, run_duration_s
            FROM runs
            ORDER BY run_id
            """,
            lconn,
        )
        local_rows = pd.read_sql_query(
            """
            SELECT
                id, run_id, sample_budget, mode, total_time_s, total_cost, mean_sentinel_quality, mean_plan_quality, plan_quality_source,
                total_sampled_records, candidate_ops_explored, candidate_ops_pruned_estimate, quality_scored_records
            FROM run_rows
            ORDER BY id
            """,
            lconn,
        )

    if local_runs.empty:
        return 0, 0

    with engine.begin() as rconn:
        for rec in local_runs.to_dict(orient="records"):
            rconn.execute(
                text(
                    """
                    INSERT INTO runs(
                        run_id, created_at, command, output_csv, return_code, budgets, train_n, eval_n,
                        strata, k, j, notes, seed, strata_composition, stratify_features, train_selection, train_skew, run_by, run_duration_s
                    )
                    VALUES (
                        :run_id, :created_at, :command, :output_csv, :return_code, :budgets, :train_n, :eval_n,
                        :strata, :k, :j, :notes, :seed, :strata_composition, :stratify_features, :train_selection, :train_skew, :run_by, :run_duration_s
                    )
                    """
                ),
                rec,
            )
        for rec in local_rows.to_dict(orient="records"):
            rconn.execute(
                text(
                    """
                    INSERT INTO run_rows(
                        id, run_id, sample_budget, mode, total_time_s, total_cost, mean_sentinel_quality, mean_plan_quality, plan_quality_source,
                        total_sampled_records, candidate_ops_explored, candidate_ops_pruned_estimate, quality_scored_records
                    )
                    VALUES (
                        :id, :run_id, :sample_budget, :mode, :total_time_s, :total_cost, :mean_sentinel_quality, :mean_plan_quality, :plan_quality_source,
                        :total_sampled_records, :candidate_ops_explored, :candidate_ops_pruned_estimate, :quality_scored_records
                    )
                    """
                ),
                rec,
            )
        # Advance sequences so future inserts don't collide with migrated IDs.
        rconn.execute(text("SELECT setval(pg_get_serial_sequence('runs', 'run_id'), COALESCE((SELECT MAX(run_id) FROM runs), 1), true)"))
        rconn.execute(text("SELECT setval(pg_get_serial_sequence('run_rows', 'id'), COALESCE((SELECT MAX(id) FROM run_rows), 1), true)"))

    return int(len(local_runs)), int(len(local_rows))


def _load_history_runs(repo_root: Path) -> pd.DataFrame:
    db_ref = _init_history_db(repo_root)
    query = """
        SELECT
            run_id, created_at, return_code, budgets, train_n, eval_n, strata, k, j,
            seed, strata_composition, stratify_features, train_selection, train_skew, run_by, output_csv, run_duration_s
        FROM runs
        ORDER BY run_id DESC
        LIMIT 200
    """
    if db_ref.startswith("postgres"):
        engine = create_engine(db_ref)
        with engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)
    with sqlite3.connect(db_ref) as conn:
        return pd.read_sql_query(query, conn)


def _load_history_rows(repo_root: Path, run_ids: list[int]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()
    db_ref = _init_history_db(repo_root)
    placeholders = ",".join(str(int(x)) for x in run_ids)
    query = f"""
        SELECT
            rr.run_id, rr.sample_budget, rr.mode, rr.total_time_s, rr.total_cost,
            rr.mean_sentinel_quality, rr.mean_plan_quality, rr.plan_quality_source,
            rr.total_sampled_records, rr.candidate_ops_explored, rr.candidate_ops_pruned_estimate, rr.quality_scored_records
        FROM run_rows rr
        WHERE rr.run_id IN ({placeholders})
        ORDER BY rr.run_id, rr.sample_budget, rr.mode
    """
    if db_ref.startswith("postgres"):
        engine = create_engine(db_ref)
        with engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)
    with sqlite3.connect(db_ref) as conn:
        return pd.read_sql_query(query, conn)


def _load_history_rows_with_fallback(repo_root: Path, run_ids: list[int]) -> pd.DataFrame:
    """Load per-budget rows from DB, falling back to run CSVs when rows are missing."""
    rows_df = _load_history_rows(repo_root, run_ids)
    have_ids = set(int(x) for x in rows_df["run_id"].dropna().tolist()) if not rows_df.empty else set()
    missing_ids = [int(x) for x in run_ids if int(x) not in have_ids]
    if not missing_ids:
        return rows_df

    db_ref = _init_history_db(repo_root)
    placeholders = ",".join(str(int(x)) for x in missing_ids)
    query = f"""
        SELECT run_id, output_csv
        FROM runs
        WHERE run_id IN ({placeholders})
    """
    fallback_rows: list[pd.DataFrame] = []
    if db_ref.startswith("postgres"):
        engine = create_engine(db_ref)
        with engine.connect() as conn:
            run_meta = pd.read_sql_query(text(query), conn)
    else:
        with sqlite3.connect(db_ref) as conn:
            run_meta = pd.read_sql_query(query, conn)
    for _, meta in run_meta.iterrows():
        run_id = int(meta["run_id"])
        output_csv = str(meta.get("output_csv", "") or "").strip()
        if not output_csv:
            continue
        csv_df = _read_results_csv(repo_root, output_csv)
        if csv_df is None or csv_df.empty:
            continue
        needed_cols = [
            "sample_budget", "mode", "total_time_s", "total_cost", "mean_sentinel_quality", "mean_plan_quality", "plan_quality_source",
            "total_sampled_records", "candidate_ops_explored", "candidate_ops_pruned_estimate", "quality_scored_records",
        ]
        safe_cols = [c for c in needed_cols if c in csv_df.columns]
        if "sample_budget" not in safe_cols or "mode" not in safe_cols:
            continue
        sub = csv_df.loc[:, safe_cols].copy()
        sub["run_id"] = run_id
        for col in needed_cols:
            if col not in sub.columns:
                sub[col] = pd.NA
        fallback_rows.append(
            sub[
                [
                    "run_id", "sample_budget", "mode", "total_time_s", "total_cost",
                    "mean_sentinel_quality", "mean_plan_quality", "plan_quality_source",
                    "total_sampled_records", "candidate_ops_explored", "candidate_ops_pruned_estimate", "quality_scored_records",
                ]
            ]
        )

    if fallback_rows:
        merged = pd.concat([rows_df] + fallback_rows, ignore_index=True) if not rows_df.empty else pd.concat(fallback_rows, ignore_index=True)
        return merged.sort_values(["run_id", "sample_budget", "mode"])
    return rows_df


def _render_history_tab(repo_root: Path) -> None:
    st.subheader("Run History")
    runs_df = _load_history_runs(repo_root)
    if runs_df.empty:
        st.info("No saved runs yet. Execute a run first; it will be stored automatically.")
        return
    selected_run_from_table: int | None = None
    table_event = st.dataframe(
        runs_df,
        width="content",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="history_runs_table_select",
    )
    try:
        sel_rows = table_event.get("selection", {}).get("rows", [])
        if sel_rows:
            row_idx = int(sel_rows[0])
            if 0 <= row_idx < len(runs_df):
                selected_run_from_table = int(runs_df.iloc[row_idx]["run_id"])
                st.session_state["history_modal_run_id"] = selected_run_from_table
    except Exception:
        selected_run_from_table = None
    picked = st.multiselect("Select run IDs to compare", options=runs_df["run_id"].tolist(), default=runs_df["run_id"].head(2).tolist())
    picked_ids = [int(x) for x in picked]
    rows_df = _load_history_rows_with_fallback(repo_root, picked_ids)
    if rows_df.empty:
        picked_meta = runs_df[runs_df["run_id"].isin(picked_ids)]
        failed = int((picked_meta["return_code"] != 0).sum()) if not picked_meta.empty else 0
        st.info(
            "No per-budget rows found for selected runs. "
            f"Failed runs in selection: {failed}. "
            "Try selecting successful runs, or check output_csv paths."
        )
        return
    st.markdown("**Selected run rows**")
    st.dataframe(rows_df, width="content", hide_index=True)
    if {"sample_budget", "mode", "total_time_s", "run_id"}.issubset(rows_df.columns):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for (run_id, mode), grp in rows_df.dropna(subset=["total_time_s"]).groupby(["run_id", "mode"], dropna=False):
            g = grp.sort_values("sample_budget")
            ax.plot(g["sample_budget"], g["total_time_s"], marker="o", label=f"run {run_id} - {mode}")
        ax.set_title("Runtime comparison across saved runs")
        ax.set_xlabel("Sample budget")
        ax.set_ylabel("Total runtime (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)

    st.markdown("---")
    st.markdown("### Inspect Single Run")
    inspect_run_id = st.selectbox(
        "Run ID",
        options=runs_df["run_id"].tolist(),
        index=0,
        help="Open a focused popup with this run's details and graphs.",
        key="history_inspect_run_id",
    )
    if "history_modal_run_id" not in st.session_state:
        st.session_state["history_modal_run_id"] = None
    if st.button("Open run report popup"):
        st.session_state["history_modal_run_id"] = int(inspect_run_id)

    run_id_for_modal = st.session_state.get("history_modal_run_id")
    if run_id_for_modal is not None:
        run_match = runs_df[runs_df["run_id"] == int(run_id_for_modal)]
        output_csv = None if run_match.empty else str(run_match.iloc[0].get("output_csv", "") or "").strip()
        modal_title = f"Run {run_id_for_modal} report"
        if hasattr(st, "dialog"):
            @st.dialog(modal_title, width="large")
            def _show_history_run_dialog() -> None:
                if run_match.empty:
                    st.warning("Run metadata not found.")
                else:
                    st.dataframe(run_match, width="content", hide_index=True)
                    _render_results_analysis(repo_root, output_csv or None)
                if st.button("Close popup"):
                    st.session_state["history_modal_run_id"] = None
                    st.rerun()

            _show_history_run_dialog()
        else:
            # Fallback for Streamlit versions without dialog support.
            st.info(f"{modal_title} (inline fallback)")
            if run_match.empty:
                st.warning("Run metadata not found.")
            else:
                st.dataframe(run_match, width="content", hide_index=True)
                _render_results_analysis(repo_root, output_csv or None)
            if st.button("Close run report"):
                st.session_state["history_modal_run_id"] = None
                st.rerun()


def _render_analysis_tab(repo_root: Path) -> None:
    st.subheader("Cross-Run Analysis")
    runs_df = _load_history_runs(repo_root)
    if runs_df.empty:
        st.info("No saved runs yet. Run experiments first, then compare them here.")
        return
    ok_only = st.toggle("Only successful runs", value=True)
    if ok_only:
        runs_df = runs_df[runs_df["return_code"] == 0].copy()
    if runs_df.empty:
        st.info("No successful runs found.")
        return

    run_ids = runs_df["run_id"].tolist()
    selected_run_ids = st.multiselect("Runs to include", options=run_ids, default=run_ids[: min(20, len(run_ids))])
    if not selected_run_ids:
        st.info("Select at least one run.")
        return

    selected_ids = [int(x) for x in selected_run_ids]
    rows_df = _load_history_rows_with_fallback(repo_root, selected_ids)
    if rows_df.empty:
        selected_meta = runs_df[runs_df["run_id"].isin(selected_ids)]
        failed = int((selected_meta["return_code"] != 0).sum()) if not selected_meta.empty else 0
        st.info(
            "No per-budget rows available for the selected runs. "
            f"Failed runs in selection: {failed}. "
            "Try selecting successful runs or rerun with explicit models."
        )
        return

    meta_cols = [
        "run_id",
        "seed",
        "strata_composition",
        "stratify_features",
        "train_selection",
        "train_skew",
        "run_by",
    ]
    meta = runs_df.loc[:, meta_cols].drop_duplicates(subset=["run_id"])
    df = rows_df.merge(meta, on="run_id", how="left")
    if "plan_quality_source" in df.columns:
        sources = sorted(str(x) for x in df["plan_quality_source"].dropna().unique().tolist() if str(x).strip())
        if sources:
            selected_sources = st.multiselect(
                "Plan quality source filter",
                options=sources,
                default=sources,
                help="Use this to keep only true final-plan quality rows (e.g. final_plan) or include fallback rows.",
            )
            if selected_sources:
                df = df[df["plan_quality_source"].astype(str).isin(selected_sources)].copy()
            if df.empty:
                st.info("No rows remain after plan quality source filtering.")
                return
    # Rich config label for clearer paper-facing comparisons.
    df["setting_label"] = df.apply(
        lambda r: (
            f"comp={r.get('strata_composition', 'na')} | "
            f"train={r.get('train_selection', 'na')} | "
            f"skew={r.get('train_skew', 'na')} | "
            f"seed={r.get('seed', 'na')} | "
            f"by={r.get('run_by', 'na')}"
        ),
        axis=1,
    )

    # Quality metric selection: plan quality preferred if present.
    has_plan = df["mean_plan_quality"].notna().any()
    has_sentinel = df["mean_sentinel_quality"].notna().any()
    metric_options = (
        ["mean_plan_quality", "mean_sentinel_quality"]
        if has_plan and has_sentinel
        else (["mean_plan_quality"] if has_plan else ["mean_sentinel_quality"])
    )
    metric_labels = {
        "mean_plan_quality": "Final-plan quality",
        "mean_sentinel_quality": "Sentinel-stage quality",
    }
    quality_metric = st.selectbox(
        "Quality metric",
        options=metric_options,
        format_func=lambda x: metric_labels.get(str(x), str(x)),
        help="Choose whether comparisons use final executed plan quality or sentinel-stage quality.",
    )
    if has_plan and has_sentinel:
        st.caption("Both quality types are available below: sentinel-stage and final-plan quality.")
    if "mean_plan_quality" in df.columns and not df["mean_plan_quality"].notna().any():
        st.warning(
            "Final-plan quality column exists but has no values. "
            "This usually means judge/model scoring failed during runs."
        )

    groupby_col = st.selectbox(
        "Group runs by",
        options=["setting_label", "strata_composition", "stratify_features", "train_selection", "train_skew", "seed", "run_by"],
        help="Groups are compared using stratified-vs-random deltas within each run and budget.",
    )

    # Paired random/stratified table.
    piv = df.pivot_table(
        index=["run_id", "sample_budget", groupby_col, "seed"],
        columns="mode",
        values=["total_time_s", "total_cost", quality_metric],
        aggfunc="mean",
    )
    needed = [("total_time_s", "random"), ("total_time_s", "stratified")]
    if any(c not in piv.columns for c in needed):
        st.info("Need both random and stratified rows for selected runs.")
        return
    piv = piv.dropna(subset=needed).copy()
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    paired = piv.reset_index()

    # Optional budget range filter.
    budgets = sorted(int(x) for x in paired["sample_budget"].dropna().unique().tolist())
    if budgets:
        bmin, bmax = budgets[0], budgets[-1]
        lo, hi = st.slider("Budget range", min_value=bmin, max_value=bmax, value=(bmin, bmax))
        paired = paired[(paired["sample_budget"] >= lo) & (paired["sample_budget"] <= hi)].copy()
    if paired.empty:
        st.info("No paired rows after filtering.")
        return

    # Deltas and win flags.
    paired["quality_delta"] = paired.get(f"{quality_metric}__stratified") - paired.get(f"{quality_metric}__random")
    if {"mean_sentinel_quality__random", "mean_sentinel_quality__stratified"}.issubset(paired.columns):
        paired["sentinel_quality_delta"] = paired["mean_sentinel_quality__stratified"] - paired["mean_sentinel_quality__random"]
    if {"mean_plan_quality__random", "mean_plan_quality__stratified"}.issubset(paired.columns):
        paired["plan_quality_delta"] = paired["mean_plan_quality__stratified"] - paired["mean_plan_quality__random"]
    paired["runtime_delta_s"] = paired["total_time_s__stratified"] - paired["total_time_s__random"]
    paired["cost_delta"] = paired.get("total_cost__stratified") - paired.get("total_cost__random")
    denom = paired["total_time_s__random"].replace(0, pd.NA)
    paired["runtime_speedup_pct"] = (paired["total_time_s__random"] - paired["total_time_s__stratified"]) / denom * 100.0
    paired["quality_win"] = paired["quality_delta"] > 0
    paired["runtime_win"] = paired["runtime_delta_s"] < 0

    # Early signal when selected runs are effectively flat/degenerate.
    near_zero_quality = paired["quality_delta"].fillna(0.0).abs().max() < 1e-9
    near_zero_cost = paired["cost_delta"].fillna(0.0).abs().max() < 1e-9
    if near_zero_quality and near_zero_cost:
        st.warning(
            "Selected runs are mostly flat (quality/cost deltas ~0). "
            "These are not strong evidence runs for your paper. "
            "Try larger eval_n, multiple seeds, and explicit model list."
        )

    # --- Tables ---
    st.markdown("---")
    st.markdown("### Leaderboard (Aggregated by Setting)")
    st.caption("What it shows: which settings most consistently beat random across budgets and runs.")
    leaderboard = (
        paired.groupby(groupby_col, dropna=False)
        .agg(
            mean_quality_delta=("quality_delta", "mean"),
            mean_runtime_speedup_pct=("runtime_speedup_pct", "mean"),
            mean_cost_delta=("cost_delta", "mean"),
            quality_win_rate=("quality_win", "mean"),
            runtime_win_rate=("runtime_win", "mean"),
            n_points=("run_id", "count"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    leaderboard["quality_win_rate"] = leaderboard["quality_win_rate"] * 100.0
    leaderboard["runtime_win_rate"] = leaderboard["runtime_win_rate"] * 100.0
    if "sentinel_quality_delta" in paired.columns:
        sentinel_mean = paired.groupby(groupby_col, dropna=False)["sentinel_quality_delta"].mean().rename("mean_sentinel_quality_delta")
        leaderboard = leaderboard.merge(sentinel_mean.reset_index(), on=groupby_col, how="left")
    if "plan_quality_delta" in paired.columns:
        plan_mean = paired.groupby(groupby_col, dropna=False)["plan_quality_delta"].mean().rename("mean_plan_quality_delta")
        leaderboard = leaderboard.merge(plan_mean.reset_index(), on=groupby_col, how="left")
    leaderboard = leaderboard.sort_values(["quality_win_rate", "mean_runtime_speedup_pct"], ascending=False)
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    st.markdown("### Per-Budget Comparison Table")
    st.caption("What it shows: exact paired values (random vs stratified) and deltas at each budget.")
    per_budget_cols = [
        "run_id",
        groupby_col,
        "sample_budget",
        "mean_sentinel_quality__random",
        "mean_sentinel_quality__stratified",
        "sentinel_quality_delta",
        "mean_plan_quality__random",
        "mean_plan_quality__stratified",
        "plan_quality_delta",
        f"{quality_metric}__random",
        f"{quality_metric}__stratified",
        "quality_delta",
        "total_time_s__random",
        "total_time_s__stratified",
        "runtime_delta_s",
        "runtime_speedup_pct",
        "total_cost__random",
        "total_cost__stratified",
        "cost_delta",
    ]
    # Avoid duplicate column names when the selected quality metric overlaps
    # with explicitly included sentinel/final quality columns.
    deduped_per_budget_cols = list(dict.fromkeys(per_budget_cols))
    available_cols = [c for c in deduped_per_budget_cols if c in paired.columns]
    st.dataframe(paired.loc[:, available_cols].sort_values(["run_id", "sample_budget"]), width="stretch", hide_index=True)

    st.markdown("### Robustness Table (Across Seeds)")
    st.caption("What it shows: variability across runs/seeds, not just average effect size.")
    robust = (
        paired.groupby(groupby_col, dropna=False)
        .agg(
            quality_delta_mean=("quality_delta", "mean"),
            quality_delta_std=("quality_delta", "std"),
            runtime_speedup_mean=("runtime_speedup_pct", "mean"),
            runtime_speedup_std=("runtime_speedup_pct", "std"),
            cost_delta_mean=("cost_delta", "mean"),
            cost_delta_std=("cost_delta", "std"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
        .sort_values("runtime_speedup_mean", ascending=False)
    )
    if "sentinel_quality_delta" in paired.columns:
        s_stats = paired.groupby(groupby_col, dropna=False).agg(
            sentinel_quality_delta_mean=("sentinel_quality_delta", "mean"),
            sentinel_quality_delta_std=("sentinel_quality_delta", "std"),
        )
        robust = robust.merge(s_stats.reset_index(), on=groupby_col, how="left")
    if "plan_quality_delta" in paired.columns:
        p_stats = paired.groupby(groupby_col, dropna=False).agg(
            plan_quality_delta_mean=("plan_quality_delta", "mean"),
            plan_quality_delta_std=("plan_quality_delta", "std"),
        )
        robust = robust.merge(p_stats.reset_index(), on=groupby_col, how="left")
    st.dataframe(robust, width="stretch", hide_index=True)

    # --- Charts ---
    st.markdown("---")
    st.markdown("### Comparative Charts")
    st.caption("Each chart compares stratified against random baseline across the selected runs.")
    selected_groups = st.multiselect(
        f"Groups to plot ({groupby_col})",
        options=sorted(str(x) for x in paired[groupby_col].dropna().unique().tolist()),
        default=sorted(str(x) for x in paired[groupby_col].dropna().unique().tolist())[:4],
    )
    plot_df = paired[paired[groupby_col].astype(str).isin(selected_groups)].copy() if selected_groups else paired.copy()
    if plot_df.empty:
        st.info("No rows for selected groups.")
        return

    # Complexity-focused sample-efficiency error view (Flesch-Kincaid proxy).
    st.markdown("### Sample-Efficiency Error Curves (Flesch-Kincaid Complexity)")
    complexity_df = df.copy()
    if "stratify_features" in complexity_df.columns:
        complexity_mask = complexity_df["stratify_features"].fillna("").str.contains("complexity_score", case=False, regex=False)
        if complexity_mask.any():
            complexity_df = complexity_df[complexity_mask].copy()
    # Build a per-run reference at the maximum budget across both modes.
    ref_cols = ["run_id", "sample_budget", "mode", quality_metric, "total_time_s"]
    ref_source = complexity_df.loc[:, [c for c in ref_cols if c in complexity_df.columns]].copy()
    if not ref_source.empty and {"run_id", "sample_budget", "mode", quality_metric, "total_time_s"}.issubset(ref_source.columns):
        max_budget = ref_source.groupby("run_id", as_index=False)["sample_budget"].max().rename(columns={"sample_budget": "max_budget"})
        ref_source = ref_source.merge(max_budget, on="run_id", how="left")
        ref_at_max = ref_source[ref_source["sample_budget"] == ref_source["max_budget"]].copy()
        ref_by_run = (
            ref_at_max.groupby("run_id", as_index=False)
            .agg(
                true_quality_proxy=(quality_metric, "mean"),
                true_runtime_proxy=("total_time_s", "mean"),
            )
        )
        err_df = ref_source.merge(ref_by_run, on="run_id", how="left")
        err_df["quality_error"] = (err_df[quality_metric] - err_df["true_quality_proxy"]).abs()
        err_df["runtime_error_s"] = (err_df["total_time_s"] - err_df["true_runtime_proxy"]).abs()
        err_df = err_df[err_df["mode"].astype(str).str.lower().isin(["random", "stratified"])].copy()
        if not err_df.empty:
            c_err1, c_err2 = st.columns(2)
            with c_err1:
                _graph_heading(
                    f"{metric_labels.get(quality_metric, quality_metric)} Error vs Number of Samples",
                    f"What it shows: absolute {metric_labels.get(quality_metric, quality_metric).lower()} error at each budget, using each run's highest-budget value as the true proxy. How to use it: a line that drops faster is more sample-efficient.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
                qagg = err_df.groupby(["mode", "sample_budget"], dropna=False)["quality_error"].mean().reset_index()
                for mode, grp in qagg.groupby("mode", dropna=False):
                    style = {"linestyle": "--", "marker": "o"} if str(mode).lower() == "random" else {"linestyle": "-", "marker": "s"}
                    ax.plot(grp["sample_budget"], grp["quality_error"], linewidth=2, label=str(mode), **style)
                ax.set_xlabel("Number of samples (budget)")
                ax.set_ylabel("Quality error (absolute)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                st.pyplot(fig, width="content")
                plt.close(fig)
            with c_err2:
                _graph_heading(
                    "Runtime Error vs Number of Samples",
                    "What it shows: absolute runtime error at each budget, using each run's highest-budget runtime as the true-runtime proxy. How to use it: lower and faster-decreasing lines indicate better runtime estimation.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
                tagg = err_df.groupby(["mode", "sample_budget"], dropna=False)["runtime_error_s"].mean().reset_index()
                for mode, grp in tagg.groupby("mode", dropna=False):
                    style = {"linestyle": "--", "marker": "o"} if str(mode).lower() == "random" else {"linestyle": "-", "marker": "s"}
                    ax.plot(grp["sample_budget"], grp["runtime_error_s"], linewidth=2, label=str(mode), **style)
                ax.set_xlabel("Number of samples (budget)")
                ax.set_ylabel("Runtime error (seconds, absolute)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                st.pyplot(fig, width="content")
                plt.close(fig)

    # 1) Delta vs Budget line (quality + runtime)
    c1, c2 = st.columns(2)
    with c1:
        _graph_heading(
            "Quality Delta vs Budget (line)",
            "What it shows: average quality difference (stratified minus random) across selected runs, split by budget. How to use it: points above zero favor stratified quality.",
        )
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        agg_q = plot_df.groupby([groupby_col, "sample_budget"], dropna=False)["quality_delta"].mean().reset_index()
        for g, grp in agg_q.groupby(groupby_col, dropna=False):
            ax.plot(grp["sample_budget"], grp["quality_delta"], marker="o", linewidth=2, label=str(g))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Sample budget")
        ax.set_ylabel("Quality delta (stratified - random)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)
    with c2:
        _graph_heading(
            "Runtime Delta vs Budget (line)",
            "What it shows: average runtime difference in seconds (stratified minus random) across selected runs. How to use it: points below zero mean stratified is faster.",
        )
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        agg_t = plot_df.groupby([groupby_col, "sample_budget"], dropna=False)["runtime_delta_s"].mean().reset_index()
        for g, grp in agg_t.groupby(groupby_col, dropna=False):
            ax.plot(grp["sample_budget"], grp["runtime_delta_s"], marker="o", linewidth=2, label=str(g))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Sample budget")
        ax.set_ylabel("Runtime delta seconds")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)

    # 2) Speedup vs Budget
    c3, c4 = st.columns(2)
    with c3:
        _graph_heading(
            "Runtime Speedup vs Budget (line)",
            "What it shows: average runtime percent improvement versus random across selected runs. How to use it: values above zero mean stratified saved time.",
        )
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        agg_s = plot_df.groupby([groupby_col, "sample_budget"], dropna=False)["runtime_speedup_pct"].mean().reset_index()
        for g, grp in agg_s.groupby(groupby_col, dropna=False):
            ax.plot(grp["sample_budget"], grp["runtime_speedup_pct"], marker="o", linewidth=2, label=str(g))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Sample budget")
        ax.set_ylabel("Speedup (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)
    with c4:
        _graph_heading(
            "Win Rate by Setting (bar)",
            "What it shows: for each setting group, the percent of comparisons where stratified wins. How to use it: taller bars indicate more reliable gains.",
        )
        wr = (
            plot_df.groupby(groupby_col, dropna=False)
            .agg(quality_win_rate=("quality_win", "mean"), runtime_win_rate=("runtime_win", "mean"))
            .reset_index()
        )
        wr["quality_win_rate"] *= 100
        wr["runtime_win_rate"] *= 100
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        x = range(len(wr))
        ax.bar([i - 0.2 for i in x], wr["quality_win_rate"], width=0.4, label="quality win rate")
        ax.bar([i + 0.2 for i in x], wr["runtime_win_rate"], width=0.4, label="runtime win rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in wr[groupby_col]], rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Win rate (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)

    # 3) Effect size distribution + Pareto
    c5, c6 = st.columns(2)
    with c5:
        _graph_heading(
            "Effect Size Distribution (boxplot)",
            "What it shows: spread of quality deltas for each setting group. How to use it: boxes mostly above zero indicate positive gains; tighter boxes indicate more stable results.",
        )
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        groups = []
        labels = []
        for g, grp in plot_df.groupby(groupby_col, dropna=False):
            groups.append(grp["quality_delta"].dropna().values)
            labels.append(str(g))
        if groups:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_ylabel("Quality delta")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig, width="content")
        plt.close(fig)
    with c6:
        _graph_heading(
            "Pareto View (quality delta vs runtime delta)",
            "What it shows: each point is one comparison with runtime change on x and quality change on y. How to use it: upper-left area is best (better quality and faster runtime).",
        )
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        for g, grp in plot_df.groupby(groupby_col, dropna=False):
            ax.scatter(grp["runtime_delta_s"], grp["quality_delta"], alpha=0.7, label=str(g))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Runtime delta (s)")
        ax.set_ylabel("Quality delta")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig, width="content")
        plt.close(fig)


@functools.lru_cache(maxsize=1)
def _load_stratifier_stress_module():
    path = Path(__file__).resolve().parent / "generate_stratifier_stress_dataset.py"
    name = "_pz_stratifier_stress_ds"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stratifier stress generator from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required before exec_module: @dataclass looks up cls.__module__ in sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stress_dataset_domain_options() -> list[str]:
    return ["bio_medical", "cs", "math", "physics"]


def _render_stress_feature_preview(df_feat: pd.DataFrame) -> None:
    st.markdown("#### Latest feature preview")
    st.dataframe(df_feat.head(16), width="stretch", hide_index=True)
    h1, h2 = st.columns(2)
    with h1:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        bins_n = min(24, max(8, len(df_feat) // 3))
        df_feat["word_count"].plot.hist(ax=ax, bins=bins_n, color="#1f77b4", alpha=0.75)
        ax.set_xlabel("word_count (synthetic)")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig, width="content")
        plt.close(fig)
    with h2:
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE)
        vc = df_feat["domain"].astype(str).value_counts()
        vc.plot(kind="bar", ax=ax, color="#ff7f0e", alpha=0.8)
        ax.set_xlabel("domain")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35)
        st.pyplot(fig, width="content")
        plt.close(fig)


def _render_stress_dataset_builder(repo_root: Path, *, widget_key_prefix: str) -> None:
    try:
        mod = _load_stratifier_stress_module()
    except Exception as exc:
        st.error(f"Could not load generator module: {exc}")
        return

    StressDatasetParams = mod.StressDatasetParams
    merge_sp = mod.merge_stress_params

    if "strat_bench_last_summary" not in st.session_state:
        st.session_state["strat_bench_last_summary"] = None

    with st.expander("What each synthetic parameter means", expanded=False):
        st.markdown(STRESS_POOL_PARAM_GUIDE)

    preset_opts = list(mod.AVAILABLE_PRESETS)
    preset_idx = preset_opts.index("rare_numeric_tail") if "rare_numeric_tail" in preset_opts else 0

    with st.form(f"stress_ds_builder_{widget_key_prefix}"):
        preset = st.selectbox(
            "Preset (recipe)",
            options=preset_opts,
            index=preset_idx,
            format_func=lambda p: SYNTHESIS_PRESET_LABELS.get(str(p), str(p)),
            help=(
                "Pick **which fiction** synthetic rows embody: tails on word-count, categorical domain rarity, "
                "or splits on section-count. Sliders mutate only the highlighted recipe."
            ),
        )
        st.caption(mod.PRESET_DESCRIPTIONS[preset])

        out_dir = st.text_input(
            "Output folder (inside repo)",
            value="sampling-project/results/sampling-results/pools/stratifier_stress_seed42",
            help="Creates this directory (relative to repo root), drops `paper_features.csv` plus `doc_*.pdf`; "
            "old `doc_*.pdf` in that folder disappear on regenerate.",
        )
        sb_seed = st.number_input(
            "Pool random seed",
            min_value=0,
            value=42,
            step=1,
            help=(
                "Controls stochastic sampling **while fabricating rows**—which corpus PDF gets copied and which "
                "random numbers populate synthetic features. Separate from Experiment **Seed**, which configures "
                "the sentinel A/B sampler once you enqueue a job."
            ),
        )
        n_docs = st.number_input(
            "Pool size (# PDFs)",
            min_value=4,
            value=48,
            step=1,
            help=(
                "Number of duplicated/symlinked PDFs (= rows written to `paper_features.csv`). Larger pools give "
                "the stratifier more bins to balance but lengthen each experiment."
            ),
        )
        source_papers = st.text_input(
            "PDF donor folder",
            value="papers",
            help="Recursive search root for `.pdf` files to recycle into the synthetic folder; originals untouched.",
        )
        link_mode = st.selectbox(
            "How to attach PDF bytes",
            options=["copy", "symlink"],
            index=0,
            format_func=lambda m: {"copy": "Copy files (standalone folder)", "symlink": "Symlink originals (fast)"}.get(
                str(m), str(m)
            ),
            help="Copy duplicates disk but relocatable; symlink is instant yet depends on originals staying put.",
        )
        sync_eval_n = st.checkbox(
            "Match Eval N to pool size afterward",
            value=True,
            help="When checked, sets Run-form Eval N equal to Pool size so sentinel never scans beyond synthesized rows.",
        )

        base = StressDatasetParams.defaults()
        params = base
        dom_opts = _stress_dataset_domain_options()

        if preset == "rare_numeric_tail":
            st.markdown("**Rare numeric tail controls**")
            tf = st.slider(
                "Tail fraction",
                0.02,
                0.50,
                float(base.tail_fraction),
                step=0.01,
                key=f"{widget_key_prefix}_tf",
                help=(
                    "Fraction of synthesized rows pretending to belong to an extreme-length regime. Higher values "
                    "make outliers less rare statistically (gentler imbalance)."
                ),
            )
            tail_dom = st.selectbox(
                "`domain` tag for minority tail rows",
                options=dom_opts,
                index=dom_opts.index(base.tail_domain) if base.tail_domain in dom_opts else 0,
                key=f"{widget_key_prefix}_taildom",
                help="Synthetic label applied only inside the minority slice; categorical stratifiers can pick it up.",
            )
            bulk_pick = st.multiselect(
                "`domain` tags rotating through bulk regime",
                options=dom_opts,
                default=list(base.bulk_domains),
                key=f"{widget_key_prefix}_bulkdom",
                help="Each bulk row draws uniformly from chosen labels.",
            )
            c1, c2 = st.columns(2)
            with c1:
                bwl = st.number_input(
                    "Bulk synthetic word_min",
                    value=int(base.bulk_word_low),
                    step=100,
                    key=f"{widget_key_prefix}_bwl",
                    help="Minimum `word_count` for majority rows.",
                )
                bwh = st.number_input(
                    "Bulk synthetic word_max",
                    value=int(base.bulk_word_high),
                    step=100,
                    key=f"{widget_key_prefix}_bwh",
                    help="Maximum `word_count` for majority rows.",
                )
            with c2:
                twl = st.number_input(
                    "Tail synthetic word_min",
                    value=int(base.tail_word_low),
                    step=500,
                    key=f"{widget_key_prefix}_twl",
                    help="Lower bound for unusually long synthetic papers.",
                )
                twh = st.number_input(
                    "Tail synthetic word_max",
                    value=int(base.tail_word_high),
                    step=500,
                    key=f"{widget_key_prefix}_twh",
                    help="Upper bound for unusually long synthetic papers.",
                )
            bd = tuple(bulk_pick) if bulk_pick else base.bulk_domains
            params = replace(
                base,
                tail_fraction=tf,
                tail_domain=tail_dom,
                bulk_domains=bd,
                bulk_word_low=bwl,
                bulk_word_high=bwh,
                tail_word_low=twl,
                tail_word_high=twh,
            )

        elif preset == "domain_heavy_skew":
            st.markdown("**Domain-heavy skew controls**")
            mf = st.slider(
                "Dominant-domain share",
                0.50,
                0.98,
                float(base.majority_fraction),
                step=0.01,
                key=f"{widget_key_prefix}_mf",
                help=(
                    "Share of synthesized rows pretending to hail from Majority domain—the rest fall into Minority."
                ),
            )
            maj = st.selectbox(
                "Majority-domain label",
                options=dom_opts,
                index=dom_opts.index(base.majority_domain) if base.majority_domain in dom_opts else 0,
                key=f"{widget_key_prefix}_maj",
                help="Synthetic categorical value that appears most frequently.",
            )
            mino_idx = dom_opts.index(base.minority_domain) if base.minority_domain in dom_opts else min(2, len(dom_opts) - 1)
            mino = st.selectbox(
                "Rare-domain label",
                options=dom_opts,
                index=mino_idx,
                key=f"{widget_key_prefix}_mino",
                help="Synthetic label that should stay scarce under the Majority share slider.",
            )
            params = replace(base, majority_fraction=mf, majority_domain=maj, minority_domain=mino)

        elif preset == "bimodal_sections":
            st.markdown("**Bimodal sections controls**")
            sf = st.slider(
                "Bulk “short-structure” fraction",
                0.50,
                0.98,
                float(base.short_section_fraction),
                step=0.01,
                key=f"{widget_key_prefix}_sf",
                help="Share of synthesized rows occupying the low `section_count` band versus the elongated band.",
            )
            c1, c2 = st.columns(2)
            with c1:
                ssl = st.number_input(
                    "Short regime section_min",
                    value=int(base.short_section_low),
                    key=f"{widget_key_prefix}_ssl",
                    help="Minimum `section_count` for dominant rows.",
                )
                ssh = st.number_input(
                    "Short regime section_max",
                    value=int(base.short_section_high),
                    key=f"{widget_key_prefix}_ssh",
                    help="Maximum `section_count` for dominant rows.",
                )
            with c2:
                lsl = st.number_input(
                    "Stretch regime section_min",
                    value=int(base.long_section_low),
                    key=f"{widget_key_prefix}_lsl",
                    help="Minimum `section_count` for minority long-structure outliers.",
                )
                lsh = st.number_input(
                    "Stretch regime section_max",
                    value=int(base.long_section_high),
                    key=f"{widget_key_prefix}_lsh",
                    help="Maximum `section_count` for minority long outliers.",
                )
            bdom_pick = st.multiselect(
                "Domain picks for sampling",
                options=dom_opts,
                default=list(base.bimodal_domains),
                key=f"{widget_key_prefix}_bidom",
                help="Synthetic `domain` label choices applied across both regimes.",
            )
            domt = tuple(bdom_pick) if bdom_pick else base.bimodal_domains
            params = replace(
                base,
                short_section_fraction=sf,
                short_section_low=ssl,
                short_section_high=ssh,
                long_section_low=lsl,
                long_section_high=lsh,
                bimodal_domains=domt,
            )

        elif preset == "ultra_rare_domain_tail":
            st.markdown("**Ultra-rare tail controls (worst-case)**")
            rf = st.slider(
                "Ultra-rare fraction",
                0.01,
                0.12,
                float(base.ultra_rare_fraction),
                step=0.005,
                key=f"{widget_key_prefix}_urf",
                help="How tiny the outlier minority is; lower values are harsher low-budget stress tests.",
            )
            ur_dom = st.selectbox(
                "Ultra-rare domain",
                options=dom_opts,
                index=dom_opts.index(base.ultra_rare_domain) if base.ultra_rare_domain in dom_opts else 0,
                key=f"{widget_key_prefix}_urdom",
            )
            ubulk_pick = st.multiselect(
                "Ultra bulk domains",
                options=dom_opts,
                default=list(base.ultra_bulk_domains),
                key=f"{widget_key_prefix}_ubdom",
            )
            c1, c2 = st.columns(2)
            with c1:
                ubw_l = st.number_input(
                    "Ultra bulk word_min",
                    value=int(base.ultra_bulk_word_low),
                    step=200,
                    key=f"{widget_key_prefix}_ubw_l",
                )
                ubw_h = st.number_input(
                    "Ultra bulk word_max",
                    value=int(base.ultra_bulk_word_high),
                    step=200,
                    key=f"{widget_key_prefix}_ubw_h",
                )
                urs_l = st.number_input(
                    "Ultra-rare section_min",
                    value=int(base.ultra_rare_section_low),
                    step=5,
                    key=f"{widget_key_prefix}_urs_l",
                )
                urs_h = st.number_input(
                    "Ultra-rare section_max",
                    value=int(base.ultra_rare_section_high),
                    step=5,
                    key=f"{widget_key_prefix}_urs_h",
                )
            with c2:
                urw_l = st.number_input(
                    "Ultra-rare word_min",
                    value=int(base.ultra_rare_word_low),
                    step=500,
                    key=f"{widget_key_prefix}_urw_l",
                )
                urw_h = st.number_input(
                    "Ultra-rare word_max",
                    value=int(base.ultra_rare_word_high),
                    step=500,
                    key=f"{widget_key_prefix}_urw_h",
                )
            ubd = tuple(ubulk_pick) if ubulk_pick else base.ultra_bulk_domains
            params = replace(
                base,
                ultra_rare_fraction=rf,
                ultra_rare_domain=ur_dom,
                ultra_bulk_domains=ubd,
                ultra_bulk_word_low=ubw_l,
                ultra_bulk_word_high=ubw_h,
                ultra_rare_word_low=urw_l,
                ultra_rare_word_high=urw_h,
                ultra_rare_section_low=urs_l,
                ultra_rare_section_high=urs_h,
            )

        else:
            st.markdown("**Adversarial conflict controls (worst-case)**")
            cf = st.slider(
                "Conflict cohort-A fraction",
                0.10,
                0.90,
                float(base.conflict_fraction),
                step=0.01,
                key=f"{widget_key_prefix}_cf",
                help="Balance between two opposing synthetic regimes.",
            )
            cdom_a = st.selectbox(
                "Cohort-A domain",
                options=dom_opts,
                index=dom_opts.index(base.conflict_domain_a) if base.conflict_domain_a in dom_opts else 0,
                key=f"{widget_key_prefix}_cda",
            )
            cdom_b = st.selectbox(
                "Cohort-B domain",
                options=dom_opts,
                index=dom_opts.index(base.conflict_domain_b) if base.conflict_domain_b in dom_opts else 0,
                key=f"{widget_key_prefix}_cdb",
            )
            c1, c2 = st.columns(2)
            with c1:
                caw_l = st.number_input(
                    "Cohort-A word_min",
                    value=int(base.conflict_a_word_low),
                    step=200,
                    key=f"{widget_key_prefix}_caw_l",
                )
                caw_h = st.number_input(
                    "Cohort-A word_max",
                    value=int(base.conflict_a_word_high),
                    step=200,
                    key=f"{widget_key_prefix}_caw_h",
                )
            with c2:
                cbw_l = st.number_input(
                    "Cohort-B word_min",
                    value=int(base.conflict_b_word_low),
                    step=500,
                    key=f"{widget_key_prefix}_cbw_l",
                )
                cbw_h = st.number_input(
                    "Cohort-B word_max",
                    value=int(base.conflict_b_word_high),
                    step=500,
                    key=f"{widget_key_prefix}_cbw_h",
                )
            params = replace(
                base,
                conflict_fraction=cf,
                conflict_domain_a=cdom_a,
                conflict_domain_b=cdom_b,
                conflict_a_word_low=caw_l,
                conflict_a_word_high=caw_h,
                conflict_b_word_low=cbw_l,
                conflict_b_word_high=cbw_h,
            )

        adv_json = st.text_area(
            "Advanced JSON knobs (merged on top)",
            value="",
            height=110,
            placeholder='{\"tail_fraction\": 0.18}',
            help="Optional JSON overlay on `StressDatasetParams` declared in scripts/generate_stratifier_stress_dataset.py.",
        )

        st.caption(
            "**Generate & apply to Run** rewrites Papers directory / Features CSV (and optionally Eval N) beneath this page. "
            "Then enqueue **Start experiment (enqueue)** to run sentinel A/B plots."
        )
        gen_go = st.form_submit_button("Generate & apply to Run")

    if gen_go:
        params_final = params
        try:
            if adv_json.strip():
                overrides = json.loads(adv_json)
                if not isinstance(overrides, dict):
                    raise ValueError("JSON overrides must be an object")
                params_final = merge_sp(params, overrides)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                summary = mod.generate_stratifier_stress_dataset(
                    repo_root,
                    output_dir=out_dir.strip(),
                    seed=int(sb_seed),
                    n_docs=int(n_docs),
                    preset=str(preset),
                    source_papers=source_papers.strip(),
                    link_mode=str(link_mode),
                    stress_params=params_final,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["strat_bench_last_summary"] = summary
                st.session_state["stress_eval_last_summary"] = summary
                st.session_state["papers"] = summary["papers_dir_rel"]
                st.session_state["features_csv"] = summary["features_csv_rel"]
                st.session_state["output_csv"] = "sampling-project/results/sampling-results/runs/ab_results_strat_bench.csv"
                if sync_eval_n:
                    st.session_state["eval_n_raw"] = str(int(n_docs))
                st.success(
                    f"Wrote `{summary['papers_dir_rel']}` + `{summary['features_csv_rel']}`. "
                    f"Paths now appear in **Papers directory** / **Features CSV** below. "
                    "Queue **Start experiment**, then inspect plots beneath or under **Analysis** / **Synthetic pools**."
                )

    summ = st.session_state.get("strat_bench_last_summary")
    if isinstance(summ, dict):
        feat_path = repo_root / summ["features_csv_rel"]
        if feat_path.is_file():
            df_feat = pd.read_csv(feat_path)
            _render_stress_feature_preview(df_feat)
            st.download_button(
                "Download paper_features.csv",
                data=feat_path.read_bytes(),
                file_name="paper_features.csv",
                mime="text/csv",
                key=f"{widget_key_prefix}_dl_feat",
            )


@st.dialog("Synthetic pool builder")
def _stress_eval_dialog(repo_root: Path) -> None:
    st.caption(
        "Skew edits only ``paper_features.csv``. PDF paths are reused from your corpus via copy or symlink."
    )
    _render_stress_dataset_builder(repo_root, widget_key_prefix="dlg")


def _render_stratifier_bench_tab(repo_root: Path) -> None:
    st.subheader(SYNTHETIC_POOLS_TAB)
    st.caption(
        "Full-page synthetic pool workspace—identical knobs to Run → Eval corpus → **Open synthetic pool builder**. "
        "Use small sample budgets afterward to amplify random vs stratified differences."
    )
    with st.expander("Preset descriptions", expanded=False):
        try:
            mod = _load_stratifier_stress_module()
            for name in mod.AVAILABLE_PRESETS:
                st.markdown(f"**{name}** — {mod.PRESET_DESCRIPTIONS[name]}")
        except Exception as exc:
            st.warning(str(exc))
    st.code(
        "python scripts/generate_stratifier_stress_dataset.py --output-dir sampling-project/results/sampling-results/pools/stratifier_stress_s42 "
        "--seed 42 --n-docs 60 --preset rare_numeric_tail --stress-params-json params.json",
        language="bash",
    )
    _render_stress_dataset_builder(repo_root, widget_key_prefix="bench")


def build_command(
    *,
    papers: str,
    features_csv: str,
    train_n: int,
    eval_n: int | None,
    budgets: list[int],
    available_models: list[str],
    seed: int,
    strata: int,
    k: int,
    j: int,
    max_workers: int | None,
    stratify_features: list[str],
    strata_composition: str,
    train_selection: str,
    train_selection_strata: int,
    train_selection_features: list[str],
    train_skew: str,
    train_skew_focus_domain: str | None,
    train_skew_domain_ratios: str | None,
    random_only: bool,
    stratified_only: bool,
    no_progress: bool,
    output_csv: str | None,
    fields_json: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_sentinel_sampling_ab.py",
        "--papers",
        papers,
        "--features-csv",
        features_csv,
        "--train-n",
        str(train_n),
        "--budgets",
        *[str(b) for b in budgets],
        "--seed",
        str(seed),
        "--strata",
        str(strata),
        "--k",
        str(k),
        "--j",
        str(j),
        "--strata-composition",
        strata_composition,
        "--stratify-features",
        *stratify_features,
        "--train-selection",
        train_selection,
        "--train-selection-strata",
        str(train_selection_strata),
        "--train-selection-features",
        *train_selection_features,
        "--train-skew",
        train_skew,
    ]
    if train_skew_focus_domain:
        cmd.extend(["--train-skew-focus-domain", train_skew_focus_domain])
    if train_skew_domain_ratios:
        cmd.extend(["--train-skew-domain-ratios", train_skew_domain_ratios])
    if eval_n is not None:
        cmd.extend(["--eval-n", str(eval_n)])
    if available_models:
        cmd.extend(["--available-models", *available_models])
    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])
    if random_only:
        cmd.append("--random-only")
    if stratified_only:
        cmd.append("--stratified-only")
    if no_progress:
        cmd.append("--no-progress")
    if output_csv:
        cmd.extend(["--output-csv", output_csv])
    if fields_json:
        cmd.extend(["--fields-json", fields_json])
    return cmd


def main() -> None:
    st.set_page_config(page_title="Palimpzest Experiment Runner", layout="wide")
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure .env variables (including PALIMPZEST_HISTORY_DB_URL) are loaded for UI sessions.
    load_dotenv(repo_root / ".env", override=False)
    _init_session_defaults()
    migrated_runs = 0
    migrated_rows = 0
    try:
        migrated_runs, migrated_rows = _maybe_migrate_local_sqlite_to_remote(repo_root)
    except Exception as exc:
        st.warning(f"History migration check failed: {exc}")
    imported_batch_runs, pending_batch_runs = _sync_completed_batch_runs(repo_root)
    queue_max_parallel = int(st.session_state.get("queue_max_parallel", 1))
    started_jobs_n, started_msgs = _maybe_start_queued_jobs(repo_root, max_parallel=queue_max_parallel)
    st.title("Palimpzest Sentinel A/B Runner")
    st.caption("Run random vs stratified sampling experiments with configurable feature strata.")
    backend = "remote" if _history_db_url() else "local sqlite"
    st.caption(f"History backend: {backend}")
    if migrated_runs:
        st.success(f"Migrated {migrated_runs} local runs ({migrated_rows} rows) to shared history DB.")
    if imported_batch_runs:
        st.success(f"Imported {imported_batch_runs} completed batch run(s) into History.")
    for msg in started_msgs:
        st.info(msg)
    if pending_batch_runs:
        st.caption(f"{pending_batch_runs} batch run(s) still running.")
    if st.button("Use quick graph-test settings"):
        _apply_quick_graph_test_defaults()
        st.success("Applied quick graph-test settings (for plot testing only).")
    st.caption("Quick graph-test settings are for smoke-testing plot generation only, not final experiments.")
    view = st.segmented_control(
        "View",
        options=["Run", "History", "Analysis", SYNTHETIC_POOLS_TAB],
        default="Run",
    )
    if view == "History":
        _render_history_tab(repo_root)
        return
    if view == "Analysis":
        _render_analysis_tab(repo_root)
        return
    if view == SYNTHETIC_POOLS_TAB:
        _render_stratifier_bench_tab(repo_root)
        return

    # Keep conditional controls outside the form so they live-rerender.
    train_selection = st.selectbox(
        "Train set selection",
        options=TRAIN_SELECTION_OPTIONS,
        key="train_selection",
        help=(
            "How train docs are chosen from eval docs: prefix (first N), "
            "random, or stratified for diversity."
        ),
    )
    if train_selection == "stratified":
        train_selection_strata = st.number_input(
            "Train selection strata",
            min_value=1,
            step=1,
            key="train_selection_strata",
            help="Only used when train set selection is stratified.",
        )
        train_selection_features = st.multiselect(
            "Train selection features",
            options=STRAT_FEATURE_COLUMNS,
            key="train_selection_features",
            help="Features used to diversify the selected training subset.",
        )
    else:
        train_selection_strata = 8
        train_selection_features = STRAT_FEATURE_COLUMNS.copy()

    train_skew = st.selectbox(
        "Train skew policy",
        options=TRAIN_SKEW_OPTIONS,
        key="train_skew",
        help="Target domain mix in training set.",
    )
    if train_skew == "focus_domain":
        train_skew_focus_domain = st.text_input(
            "Train skew focus domain",
            key="train_skew_focus_domain",
            help="Example: cs",
        )
    else:
        train_skew_focus_domain = ""
    if train_skew == "custom_domain_ratios":
        train_skew_domain_ratios = st.text_input(
            "Train skew domain ratios",
            key="train_skew_domain_ratios",
            help="Example: cs=0.5,biomedical=0.2,math=0.2,physics=0.1",
        )
    else:
        train_skew_domain_ratios = ""

    st.markdown("#### Eval corpus")
    st.markdown(EVAL_CORPUS_INTRO_MARKDOWN)
    ls_ev = st.session_state.get("stress_eval_last_summary")
    if isinstance(ls_ev, dict):
        st.success(
            f"**Last synthetic pool loaded into the form:** `{ls_ev.get('papers_dir_rel')}` — "
            f"preset `{ls_ev.get('preset')}`, **n**={ls_ev.get('n_docs')}, generation seed `{ls_ev.get('seed')}`. "
            "Open **Synthetic pools** tab or the builder button to regenerate."
        )
        sp_ev = ls_ev.get("stress_params")
        if isinstance(sp_ev, dict) and sp_ev:
            with st.expander("Last synthetic pool knobs (JSON)", expanded=False):
                st.json(sp_ev)

    open_builder = st.button(
        "Open synthetic pool builder",
        key="open_stress_builder_dialog",
        type="primary",
        use_container_width=True,
        help=(
            "Build a deliberate skew-only feature table plus copied PDFs; "
            "after Generate, Papers directory and Features CSV are filled automatically."
        ),
    )
    if open_builder:
        _stress_eval_dialog(repo_root)

    st.markdown("**Extraction fields** — define what to extract from each PDF. Each entry needs `name`, `type` (`str`, `bool`, `int`, `float`), and `desc`.")
    selected_fields_preset = st.selectbox(
        "Fields preset",
        options=list(FIELD_PRESETS.keys()),
        key="fields_preset_name",
        help="Switching preset auto-populates Fields JSON below.",
    )
    if st.session_state.get("fields_preset_last_applied") != selected_fields_preset:
        st.session_state["fields_json_raw"] = FIELD_PRESETS[selected_fields_preset]
        st.session_state["fields_preset_last_applied"] = selected_fields_preset

    with st.form("runner"):
        c1, c2, c3 = st.columns(3)
        with c1:
            papers = st.text_input(
                "Papers directory",
                key="papers",
                help="PDF root for experiments. Paste a corpus path—or let **Open synthetic pool builder** populate it.",
            )
            features_csv = st.text_input(
                "Features CSV",
                key="features_csv",
                help="Stratifier input; usually from extract_features.py. Synthetic skew tables land here automatically after Generate.",
            )
            output_csv = st.text_input(
                "Output CSV (optional)",
                key="output_csv",
                help="Where to save per-run metrics for experiment tracking.",
            )
        with c2:
            train_n = st.number_input(
                "Train N",
                min_value=1,
                step=1,
                key="train_n",
                help="Number of docs available to sentinel/MAB during optimization.",
            )
            eval_n_raw = st.text_input(
                "Eval N (blank = all)",
                key="eval_n_raw",
                help="Cap on evaluation docs; blank means evaluate all docs under papers.",
            )
            budgets_raw = st.text_input(
                "Budgets (space/comma separated)",
                key="budgets_raw",
                help="Sample budgets to sweep; each budget runs random + stratified comparison.",
            )
        with c3:
            seed = st.number_input(
                "Seed",
                min_value=0,
                step=1,
                key="seed",
                help="Random seed for reproducibility of sampling/order.",
            )
            strata = st.number_input(
                "Strata",
                min_value=1,
                step=1,
                key="strata",
                help="Number of bins used in stratified ordering.",
            )
            max_workers_raw = st.text_input(
                "Max workers (blank = auto)",
                key="max_workers_raw",
                help="Parallel worker count for model calls; blank lets Palimpzest choose.",
            )

        c4, c5 = st.columns(2)
        with c4:
            k = st.number_input(
                "MAB k",
                min_value=1,
                step=1,
                key="k",
                help="Initial number of candidate operators on the MAB frontier.",
            )
            j = st.number_input(
                "MAB j",
                min_value=1,
                step=1,
                key="j",
                help="Minimum samples per operator before MAB pruning.",
            )
            models_raw = st.text_input(
                "Available models override (optional)",
                key="models_raw",
                help="Space/comma list. If set, passes --available-models exactly as provided.",
            )
        with c5:
            strata_composition = st.selectbox(
                "Strata composition",
                options=["cartesian", "composite", "exclusive"],
                key="strata_composition",
                help=(
                    "cartesian: stratify by the Cartesian product of all selected features. "
                    "composite: combine selected features into one stratifier. "
                    "exclusive: run one stratified pass per feature (mutually exclusive)."
                ),
            )
            stratify_features = st.multiselect(
                "Stratification features",
                options=STRAT_FEATURE_COLUMNS,
                key="stratify_features",
                help="Feature columns used by the stratifier for composite/exclusive modes.",
            )
            random_only = st.checkbox(
                "Random only",
                key="random_only",
                help="Skip stratified runs and execute baseline random ordering only.",
            )
            stratified_only = st.checkbox(
                "Stratified only",
                key="stratified_only",
                help="Skip random baseline and execute stratified runs only.",
            )
            no_progress = st.checkbox(
                "No progress bars",
                key="no_progress",
                help="Disable progress bars in script output.",
            )
            output_append_timestamp = st.checkbox(
                "Append timestamp to output filename",
                key="output_append_timestamp",
                help="Recommended: avoids overwriting previous runs with the same seed/config.",
            )
        fields_json_raw = st.text_area(
            "Fields JSON",
            height=220,
            key="fields_json_raw",
            help="JSON array of fields passed to sem_map. Edit to change what the LLM extracts.",
        )
        st.markdown("---")
        run_submitted = st.form_submit_button("Start experiment (enqueue)")

    jobs = _load_batch_manifest(repo_root)
    st.markdown("### Jobs")
    st.number_input(
        "Max parallel queue workers",
        min_value=1,
        step=1,
        key="queue_max_parallel",
        help="How many queued jobs can run at the same time.",
    )
    if jobs:
        queued_n, running_n, _ = _queue_state_counts(repo_root, jobs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Queued", str(queued_n))
        c2.metric("Running", str(running_n))
        c3.metric("Active queue jobs", str(queued_n + running_n))
        if running_n > 0:
            st.markdown("**Queue worker:** ⏳ running")
        elif queued_n > 0:
            st.markdown("**Queue worker:** ⏸ waiting to start next queued job")
        else:
            st.markdown("**Queue worker:** idle")
        st.caption(f"Queue summary: {queued_n} queued, {running_n} running (max parallel={int(st.session_state.get('queue_max_parallel', 1))})")
        active_rows = []
        for job in jobs[-80:]:
            status = _job_state(repo_root, job)
            rec = {
                "status": status,
                "return_code": job.get("return_code"),
                "job_id": job.get("job_id"),
                "pid": job.get("pid"),
                "started_at": _job_started_at_text(job),
                "runtime_min": round(_job_runtime_minutes(job) or 0.0, 2),
                "seed": job.get("seed"),
                "composition": job.get("strata_composition"),
                "train_selection": job.get("train_selection"),
                "train_skew": job.get("train_skew"),
                "train_n": job.get("train_n"),
                "eval_n": job.get("eval_n"),
                "budgets": " ".join(str(x) for x in (job.get("budgets") or [])),
                "output_csv": job.get("output_csv"),
                "command": job.get("command"),
                "log_file": job.get("log_file"),
                "status_file": job.get("status_file"),
            }
            if status in {"queued", "running"}:
                active_rows.append(rec)
        st.markdown("**Queued / Running**")
        if active_rows:
            active_df = pd.DataFrame(active_rows)
            st.dataframe(
                active_df[
                    [
                        "status",
                        "job_id",
                        "pid",
                        "started_at",
                        "runtime_min",
                        "seed",
                        "composition",
                        "train_selection",
                        "train_skew",
                        "train_n",
                        "eval_n",
                        "budgets",
                        "output_csv",
                    ]
                ],
                width="content",
                hide_index=True,
            )
            running_for_progress = [j for j in jobs if _job_state(repo_root, j) == "running"]
            if running_for_progress:
                st.markdown("**Estimated progress (from log file)**")
                st.caption(
                    "Each A/B phase runs sentinel optimization then the final plan, so the log gets **two** "
                    "finish lines per phase (`Total opt. time:` then `Total time:`). The bar counts those, "
                    "plus Rich `%` from the log tail for the current sub-step. "
                    "With **No progress bars**, only `=== Run A/B ===` headers are used (coarser). "
                    "Auto-refresh updates every few seconds."
                )
                for job in running_for_progress:
                    ratio, detail = _job_progress_from_log(repo_root, job)
                    label = (
                        f"Job {job.get('job_id')} · seed {job.get('seed')} · "
                        f"{job.get('strata_composition')} / {job.get('train_selection')}"
                    )
                    st.markdown(f"**{label}** — about **{ratio * 100:.0f}%** complete")
                    st.progress(min(1.0, max(0.0, ratio)))
                    st.caption(detail)
            inspect_id = st.selectbox(
                "Inspect job details",
                options=[str(x) for x in active_df["job_id"].tolist()],
                help="Choose a queued/running job to view full config details.",
            )
            sel = active_df[active_df["job_id"].astype(str) == str(inspect_id)].to_dict(orient="records")
            if sel:
                with st.expander("Selected job details", expanded=True):
                    st.json(sel[0], expanded=False)
        else:
            st.caption("No queued or running jobs right now.")
        runs_df_preview = _load_history_runs(repo_root)
        with st.expander("Show recent completed runs (history DB)", expanded=True):
            if runs_df_preview.empty:
                st.caption("No imported history rows yet.")
            else:
                st.dataframe(runs_df_preview.head(20), width="content", hide_index=True)
    else:
        st.caption("No jobs yet.")

    if not run_submitted:
        # Optional FIFO auto-polling while queue has pending/running jobs.
        auto_poll = st.checkbox(
            "Auto-refresh queue status (every 5s)",
            value=True,
            key="queue_auto_poll",
            help="Keeps FIFO queue moving while this page is open.",
        )
        if auto_poll:
            queued_n, running_n, _ = _queue_state_counts(repo_root, jobs)
            if queued_n > 0 or running_n > 0:
                time.sleep(5)
                st.rerun()
        return

    try:
        budgets = parse_int_list(budgets_raw)
        eval_n = int(eval_n_raw) if eval_n_raw.strip() else None
        max_workers = int(max_workers_raw) if max_workers_raw.strip() else None
        available_models = parse_text_list(models_raw)
        if not budgets:
            st.error("Provide at least one budget.")
            return
        if not stratify_features:
            st.error("Select at least one stratification feature.")
            return
        if train_selection == "stratified" and not train_selection_features:
            st.error("Select at least one train selection feature.")
            return
        if train_skew == "focus_domain" and not train_skew_focus_domain.strip():
            st.error("Provide focus domain when train skew policy is focus_domain.")
            return
        if train_skew == "custom_domain_ratios" and not train_skew_domain_ratios.strip():
            st.error("Provide domain ratios when train skew policy is custom_domain_ratios.")
            return
        try:
            parsed = json.loads(fields_json_raw)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Must be a non-empty JSON array.")
            for f in parsed:
                if not all(k in f for k in ("name", "type", "desc")):
                    raise ValueError(f"Each field needs 'name', 'type', and 'desc'. Got: {f}")
                if f["type"] not in ("str", "bool", "int", "float"):
                    raise ValueError(f"Invalid type {f['type']!r}. Use str, bool, int, or float.")
            fields_json = fields_json_raw.strip()
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid Fields JSON: {exc}")
            return
    except ValueError as exc:
        st.error(f"Invalid numeric input: {exc}")
        return

    base_kwargs = dict(
        papers=papers,
        features_csv=features_csv,
        train_n=int(train_n),
        eval_n=eval_n,
        budgets=budgets,
        available_models=available_models,
        seed=int(seed),
        strata=int(strata),
        k=int(k),
        j=int(j),
        max_workers=max_workers,
        stratify_features=stratify_features,
        strata_composition=strata_composition,
        train_selection=train_selection,
        train_selection_strata=int(train_selection_strata),
        train_selection_features=train_selection_features,
        train_skew=train_skew,
        train_skew_focus_domain=train_skew_focus_domain.strip() or None,
        train_skew_domain_ratios=train_skew_domain_ratios.strip() or None,
        random_only=random_only,
        stratified_only=stratified_only,
        no_progress=no_progress,
        output_csv=output_csv.strip() or None,
        fields_json=fields_json,
    )
    cmd = build_command(**base_kwargs)

    st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")

    manifest_rows = _load_batch_manifest(repo_root)
    base_output = (output_csv.strip() or "sampling-project/results/sampling-results/manual_compare/queued_run.csv")
    out_path = Path(base_output)
    suffix = f"_seed{int(seed)}_{strata_composition}_{train_selection}_{train_skew}"
    if output_append_timestamp:
        suffix += "_" + time.strftime("%Y%m%d_%H%M%S")
    if out_path.suffix.lower() == ".csv":
        out_csv = str(out_path.with_name(f"{out_path.stem}{suffix}.csv"))
    else:
        out_csv = f"{base_output}{suffix}.csv"
    cfg = {
        **base_kwargs,
        "seed": int(seed),
        "strata_composition": strata_composition,
        "train_selection": train_selection,
        "train_skew": train_skew,
        "output_csv": out_csv,
    }
    run_cmd = build_command(**cfg)
    run_cmd_str = " ".join(shlex.quote(part) for part in run_cmd)
    manifest_rows.append(
        {
            "job_id": _new_job_id(),
            "command": run_cmd_str,
            "output_csv": out_csv,
            "budgets": budgets,
            "train_n": int(train_n),
            "eval_n": eval_n,
            "strata": int(strata),
            "k": int(k),
            "j": int(j),
            "seed": int(seed),
            "strata_composition": strata_composition,
            "stratify_features": stratify_features,
            "train_selection": train_selection,
            "train_skew": train_skew,
            "random_only": random_only,
            "stratified_only": stratified_only,
            "no_progress": no_progress,
            "history_imported": False,
            "state": "queued",
        }
    )
    _save_batch_manifest(repo_root, manifest_rows)
    started_n, started_msgs = _maybe_start_queued_jobs(
        repo_root,
        max_parallel=int(st.session_state.get("queue_max_parallel", 1)),
    )
    st.success("Queued experiment config.")
    if started_n > 0:
        for msg in started_msgs:
            st.info(msg)
    else:
        st.info("Another job is active. This config will auto-start when current job finishes.")
    return


if __name__ == "__main__":
    main()
