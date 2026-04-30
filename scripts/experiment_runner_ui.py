#!/usr/bin/env python3
"""
Simple Streamlit UI for running sentinel A/B sampling experiments.

Run:
    streamlit run scripts/experiment_runner_ui.py
"""

from __future__ import annotations

import io
import json
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


STRAT_FEATURE_COLUMNS = [
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
    "domain",
]


def parse_int_list(raw: str) -> list[int]:
    values = [v.strip() for v in raw.replace(",", " ").split() if v.strip()]
    return [int(v) for v in values]


def parse_text_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.replace(",", " ").split() if v.strip()]


DEFAULT_FIELDS_JSON = json.dumps([
    {"name": "primary_contribution", "type": "str", "desc": "The single most important technical contribution of the paper in one sentence."},
    {"name": "methodology", "type": "str", "desc": "The core method or approach used (e.g. algorithm name, experimental design, proof technique)."},
    {"name": "domain", "type": "str", "desc": "The research domain: one of 'cs', 'biomedical', 'math', or 'physics'."},
    {"name": "uses_experiments", "type": "bool", "desc": "True if the paper includes empirical experiments or evaluations, False if purely theoretical."},
], indent=2)


QUICK_GRAPH_FIELDS_JSON = json.dumps([
    {"name": "primary_contribution", "type": "str", "desc": "Main contribution in one sentence."},
], indent=2)

FIGSIZE_LINE = (4.8, 2.8)
FIGSIZE_BAR = (4.8, 2.8)
FIGSIZE_WIN = (4.8, 2.8)


def _apply_quick_graph_test_defaults() -> None:
    """Set lightweight defaults intended only for fast plot smoke-tests."""
    st.session_state["train_selection"] = "prefix"
    st.session_state["train_skew"] = "natural"
    st.session_state["papers"] = "papers"
    st.session_state["features_csv"] = "papers/paper_features.csv"
    st.session_state["output_csv"] = "papers/ab_results_smoke.csv"
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
    st.session_state["models_raw"] = ""
    st.session_state["random_only"] = False
    st.session_state["stratified_only"] = False
    st.session_state["no_progress"] = False
    st.session_state["fields_json_raw"] = QUICK_GRAPH_FIELDS_JSON


def _apply_worst_case_stress_defaults() -> None:
    """
    Configure a low-budget, higher-heterogeneity stress setup where random
    sampling is more likely to be unstable than stratified sampling.
    """
    st.session_state["train_selection"] = "prefix"
    st.session_state["train_skew"] = "natural"
    st.session_state["papers"] = "papers"
    st.session_state["features_csv"] = "papers/paper_features.csv"
    st.session_state["output_csv"] = "papers/ab_results_worst_case.csv"
    st.session_state["train_n"] = 20
    st.session_state["eval_n_raw"] = "80"
    st.session_state["budgets_raw"] = "3 5 7 10"
    st.session_state["seed"] = 42
    st.session_state["strata"] = 8
    st.session_state["max_workers_raw"] = "8"
    st.session_state["k"] = 6
    st.session_state["j"] = 4
    st.session_state["strata_composition"] = "cartesian"
    st.session_state["stratify_features"] = STRAT_FEATURE_COLUMNS.copy()
    st.session_state["models_raw"] = ""
    st.session_state["random_only"] = False
    st.session_state["stratified_only"] = False
    st.session_state["no_progress"] = False
    st.session_state["fields_json_raw"] = DEFAULT_FIELDS_JSON


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
        "output_csv": "papers/ab_results.csv",
        "train_n": 20,
        "eval_n_raw": "20",
        "budgets_raw": "5 10 15 20",
        "seed": 42,
        "strata": 8,
        "max_workers_raw": "64",
        "k": 6,
        "j": 4,
        "models_raw": "",
        "strata_composition": "cartesian",
        "stratify_features": STRAT_FEATURE_COLUMNS.copy(),
        "random_only": False,
        "stratified_only": False,
        "no_progress": False,
        "fields_json_raw": DEFAULT_FIELDS_JSON,
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


def _graph_heading(title: str, why_it_matters: str) -> None:
    safe_help = why_it_matters.replace('"', "&quot;")
    st.markdown(f'**{title}** <abbr title="{safe_help}">ⓘ</abbr>', unsafe_allow_html=True)


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

    # Select whichever quality metric is actually populated.
    quality_col, quality_label = _choose_quality_metric(df)
    time_delta = _paired_budget_deltas(df.dropna(subset=["total_time_s"]), "total_time_s") if {"sample_budget", "mode", "total_time_s"}.issubset(df.columns) else pd.DataFrame()

    # Compact baseline-vs-stratified summary first.
    if not time_delta.empty:
        mean_speedup = float(((time_delta["random"] - time_delta["stratified"]) / time_delta["random"] * 100.0).mean())
        faster_budgets = int((time_delta["delta"] < 0).sum())
        total_budgets = int(len(time_delta))
        c1, c2 = st.columns(2)
        c1.metric("Stratified faster budgets", f"{faster_budgets}/{total_budgets}")
        c2.metric("Avg runtime speedup", f"{mean_speedup:.1f}%")
    if quality_col and {"sample_budget", "mode", quality_col}.issubset(df.columns):
        qd = _paired_budget_deltas(df.dropna(subset=[quality_col]), quality_col)
        if not qd.empty:
            q_wins = int((qd["delta"] > 0).sum())
            q_total = int(len(qd))
            st.metric(f"Stratified quality wins ({quality_label})", f"{q_wins}/{q_total}")

    left_col, right_col = st.columns(2)
    panel = 0

    def _next_col():
        nonlocal panel
        col = left_col if panel % 2 == 0 else right_col
        panel += 1
        return col

    # Plot 1: quality vs budget
    if quality_col and {"sample_budget", "mode", quality_col}.issubset(df.columns):
        qdf = df.dropna(subset=[quality_col]).copy()
        if not qdf.empty:
            with _next_col():
                _graph_heading(
                    f"{quality_label} vs sample budget",
                    "Shows sample efficiency: whether stratified sampling reaches equal or higher quality with smaller budgets than random baseline.",
                )
                p1 = _plot_line_by_mode(
                    qdf,
                    quality_col,
                    f"{quality_label} vs sample budget",
                    quality_label,
                )
                st.download_button(
                    "Download quality plot (PNG)",
                    data=p1,
                    file_name="quality_vs_budget.png",
                    mime="image/png",
                )

    # Plot 2: absolute quality error vs budget (relative to max budget per mode)
    if quality_col and {"sample_budget", "mode", quality_col}.issubset(df.columns):
        tmp = df.dropna(subset=[quality_col]).copy()
        if not tmp.empty:
            ref = (
                tmp.sort_values("sample_budget")
                .groupby("mode", as_index=False)
                .tail(1)
                .loc[:, ["mode", quality_col]]
                .rename(columns={quality_col: "ref_quality"})
            )
            err_df = tmp.merge(ref, on="mode", how="left")
            err_df["abs_quality_error"] = (err_df[quality_col] - err_df["ref_quality"]).abs()
            with _next_col():
                _graph_heading(
                    "Quality estimation error vs sample budget",
                    "Lower error means your sampled estimate is closer to high-budget behavior, which supports better optimizer decisions.",
                )
                p2 = _plot_line_by_mode(
                    err_df,
                    "abs_quality_error",
                    "Absolute quality error vs sample budget",
                    "|quality - quality@max_budget|",
                )
                st.download_button(
                    "Download error plot (PNG)",
                    data=p2,
                    file_name="quality_error_vs_budget.png",
                    mime="image/png",
                )

    # Plot 3: stratified quality lift over random per budget (most persuasive).
    if quality_col and {"sample_budget", "mode", quality_col}.issubset(df.columns):
        ddf = _paired_budget_deltas(df.dropna(subset=[quality_col]), quality_col)
        if not ddf.empty:
            with _next_col():
                _graph_heading(
                    "Stratified quality lift vs random (by budget)",
                    "Direct A/B delta plot: positive bars indicate stratified outperforms the random baseline at that budget.",
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
                ax.set_ylabel(f"Delta {quality_label} (stratified - random)")
                ax.set_title("Quality lift by budget")
                ax.grid(True, axis="y", alpha=0.3)
                if (ddf["delta"] == 0).all():
                    ax.set_ylim(-0.05, 0.05)
                    ax.text(0.02, 0.95, "No quality difference vs random in this run.", transform=ax.transAxes, va="top", fontsize=8)
                _render_and_download(fig, "Download quality lift plot (PNG)", "quality_lift_vs_random.png")

    # Plot 4: total cost vs budget (random baseline vs stratified).
    if {"sample_budget", "mode", "total_cost"}.issubset(df.columns):
        cst = df.dropna(subset=["total_cost"]).copy()
        if not cst.empty:
            with _next_col():
                _graph_heading(
                    "Total cost vs sample budget (baseline comparison)",
                    "Compares spend trajectories. A lower stratified line indicates cheaper optimization/execution than random.",
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
                    "Negative bars mean stratified is cheaper than random at the same sample budget.",
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
                    "Compares end-to-end latency; lower stratified runtime supports practical efficiency gains.",
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
            time_delta["speedup_pct"] = (time_delta["random"] - time_delta["stratified"]) / time_delta["random"] * 100.0
            with _next_col():
                _graph_heading(
                    "Stratified runtime speedup vs random (by budget)",
                    "Percent speedup summary. Positive values mean stratified finishes faster than random baseline.",
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

    # Plot 8: head-to-head win rate as bar chart.
    if quality_col and {"sample_budget", "mode", quality_col, "total_time_s"}.issubset(df.columns):
        qd = _paired_budget_deltas(df.dropna(subset=[quality_col]), quality_col)
        td = _paired_budget_deltas(df.dropna(subset=["total_time_s"]), "total_time_s")
        if not qd.empty and not td.empty:
            q_win_rate = float((qd["delta"] > 0).mean() * 100.0)
            t_win_rate = float((td["delta"] < 0).mean() * 100.0)  # lower runtime is better
            with _next_col():
                _graph_heading(
                    "Stratified win rate across budgets",
                    "Aggregate view of how often stratified beats random across tested budgets for quality and runtime.",
                )
                fig, ax = plt.subplots(figsize=FIGSIZE_WIN)
                labels = [f"{quality_label} win rate", "Runtime win rate"]
                vals = [q_win_rate, t_win_rate]
                bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e"], alpha=0.9)
                for b, v in zip(bars, vals):
                    y = v - 3 if v >= 10 else v + 1.5
                    va = "top" if v >= 10 else "bottom"
                    ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.1f}%", ha="center", va=va, fontsize=9)
                ax.set_ylim(0, 110)
                ax.set_ylabel("Win rate (%) vs random baseline")
                ax.set_title("Stratified win rate across budgets", pad=8)
                ax.grid(True, axis="y", alpha=0.3)
                _render_and_download(fig, "Download win-rate plot (PNG)", "win_rate_vs_random.png")


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


def _init_history_db(repo_root: Path) -> Path:
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
    return db_path


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
) -> int:
    db_path = _init_history_db(repo_root)
    csv_df = _read_results_csv(repo_root, output_csv)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs(command, output_csv, return_code, budgets, train_n, eval_n, strata, k, j)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command,
                output_csv or "",
                int(return_code),
                " ".join(str(b) for b in budgets),
                int(train_n),
                None if eval_n is None else int(eval_n),
                int(strata),
                int(k),
                int(j),
            ),
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
                    )
                )
            conn.executemany(
                """
                INSERT INTO run_rows(run_id, sample_budget, mode, total_time_s, total_cost, mean_sentinel_quality, mean_plan_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return run_id


def _load_history_runs(repo_root: Path) -> pd.DataFrame:
    db_path = _init_history_db(repo_root)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT run_id, created_at, return_code, budgets, train_n, eval_n, strata, k, j, output_csv
            FROM runs
            ORDER BY run_id DESC
            LIMIT 200
            """,
            conn,
        )


def _load_history_rows(repo_root: Path, run_ids: list[int]) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()
    db_path = _init_history_db(repo_root)
    placeholders = ",".join("?" for _ in run_ids)
    query = f"""
        SELECT rr.run_id, rr.sample_budget, rr.mode, rr.total_time_s, rr.total_cost, rr.mean_sentinel_quality, rr.mean_plan_quality
        FROM run_rows rr
        WHERE rr.run_id IN ({placeholders})
        ORDER BY rr.run_id, rr.sample_budget, rr.mode
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=run_ids)


def _render_history_tab(repo_root: Path) -> None:
    st.subheader("Run History")
    runs_df = _load_history_runs(repo_root)
    if runs_df.empty:
        st.info("No saved runs yet. Execute a run first; it will be stored automatically.")
        return
    st.dataframe(runs_df, width="stretch", hide_index=True)
    picked = st.multiselect("Select run IDs to compare", options=runs_df["run_id"].tolist(), default=runs_df["run_id"].head(2).tolist())
    rows_df = _load_history_rows(repo_root, [int(x) for x in picked])
    if rows_df.empty:
        return
    st.markdown("**Selected run rows**")
    st.dataframe(rows_df, width="stretch", hide_index=True)
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
        "python3",
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
    _init_session_defaults()
    repo_root = Path(__file__).resolve().parents[1]
    st.title("Palimpzest Sentinel A/B Runner")
    st.caption("Run random vs stratified sampling experiments with configurable feature strata.")
    if st.button("Use quick graph-test settings"):
        _apply_quick_graph_test_defaults()
        st.success("Applied quick graph-test settings (for plot testing only).")
    if st.button("Use worst-case baseline stress settings"):
        _apply_worst_case_stress_defaults()
        st.success("Applied worst-case baseline stress settings (targeting low-budget heterogeneity).")
    st.caption("Quick graph-test settings are for smoke-testing plot generation only, not final experiments.")
    view = st.segmented_control("View", options=["Run", "History"], default="Run")
    if view == "History":
        _render_history_tab(repo_root)
        return

    # Keep conditional controls outside the form so they live-rerender.
    train_selection = st.selectbox(
        "Train set selection",
        options=["prefix", "random", "stratified"],
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
        options=[
            "natural",
            "balanced_domain",
            "min_one_per_domain",
            "focus_domain",
            "custom_domain_ratios",
        ],
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

    with st.form("runner"):
        c1, c2, c3 = st.columns(3)
        with c1:
            papers = st.text_input(
                "Papers directory",
                key="papers",
                help="Directory containing PDFs used for training/evaluation.",
            )
            features_csv = st.text_input(
                "Features CSV",
                key="features_csv",
                help="Precomputed feature table from extract_features.py --scan.",
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
                "Available models (space/comma separated, optional)",
                key="models_raw",
                help="Optional model allow-list; leave blank to use default auto-detected set.",
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
        st.markdown("---")
        st.markdown("**Extraction fields** — define what to extract from each PDF. Each entry needs `name`, `type` (`str`, `bool`, `int`, `float`), and `desc`.")
        fields_json_raw = st.text_area(
            "Fields JSON",
            height=220,
            key="fields_json_raw",
            help="JSON array of fields passed to sem_map. Edit to change what the LLM extracts.",
        )
        submitted = st.form_submit_button("Run experiment")

    if not submitted:
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

    cmd = build_command(
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

    st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")

    with st.spinner("Running experiment..."):
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )

    st.subheader("Exit status")
    if proc.returncode == 0:
        st.success("Completed successfully.")
    else:
        st.error(f"Run failed with exit code {proc.returncode}")

    st.subheader("Stdout")
    cleaned_stdout, removed_stdout_noise = _clean_stdout(proc.stdout or "")
    if removed_stdout_noise:
        st.caption(f"Suppressed {removed_stdout_noise} noisy stdout lines (progress/LiteLLM helper spam).")
    st.code(_summarize_stdout(cleaned_stdout or ""), language="text")
    with st.expander("Show raw stdout"):
        st.text(proc.stdout or "(empty)")
    st.subheader("Stderr")
    cleaned_stderr, removed_noise = _clean_stderr(proc.stderr or "")
    if removed_noise:
        st.caption(f"Suppressed {removed_noise} repeated LiteLLM helper lines in stderr.")
    st.text(cleaned_stderr or "(empty)")
    with st.expander("Show raw stderr"):
        st.text(proc.stderr or "(empty)")
    saved_run_id = _save_run_history(
        repo_root,
        command=" ".join(shlex.quote(part) for part in cmd),
        output_csv=output_csv.strip() or None,
        return_code=int(proc.returncode),
        budgets=budgets,
        train_n=int(train_n),
        eval_n=eval_n,
        strata=int(strata),
        k=int(k),
        j=int(j),
    )
    st.caption(f"Saved run to history DB as run_id={saved_run_id}. Switch View -> History to compare.")
    _render_results_analysis(repo_root, output_csv.strip() or None)


if __name__ == "__main__":
    main()
