#!/usr/bin/env python3
"""
Simple Streamlit UI for running sentinel A/B sampling experiments.

Run:
    streamlit run scripts/experiment_runner_ui.py
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import streamlit as st


STRAT_FEATURE_COLUMNS = [
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
]


def parse_int_list(raw: str) -> list[int]:
    values = [v.strip() for v in raw.replace(",", " ").split() if v.strip()]
    return [int(v) for v in values]


def parse_text_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.replace(",", " ").split() if v.strip()]


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
    random_only: bool,
    stratified_only: bool,
    no_progress: bool,
    output_csv: str | None,
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
    ]
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
    return cmd


def main() -> None:
    st.set_page_config(page_title="Palimpzest Experiment Runner", layout="wide")
    st.title("Palimpzest Sentinel A/B Runner")
    st.caption("Run random vs stratified sampling experiments with configurable feature strata.")

    with st.form("runner"):
        c1, c2, c3 = st.columns(3)
        with c1:
            papers = st.text_input(
                "Papers directory",
                value="papers",
                help="Directory containing PDFs used for training/evaluation.",
            )
            features_csv = st.text_input(
                "Features CSV",
                value="papers/paper_features.csv",
                help="Precomputed feature table from extract_features.py --scan.",
            )
            output_csv = st.text_input(
                "Output CSV (optional)",
                value="papers/ab_results.csv",
                help="Where to save per-run metrics for experiment tracking.",
            )
        with c2:
            train_n = st.number_input(
                "Train N",
                min_value=1,
                value=20,
                step=1,
                help="Number of docs available to sentinel/MAB during optimization.",
            )
            eval_n_raw = st.text_input(
                "Eval N (blank = all)",
                value="20",
                help="Cap on evaluation docs; blank means evaluate all docs under papers.",
            )
            budgets_raw = st.text_input(
                "Budgets (space/comma separated)",
                value="5 10 15 20",
                help="Sample budgets to sweep; each budget runs random + stratified comparison.",
            )
        with c3:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=42,
                step=1,
                help="Random seed for reproducibility of sampling/order.",
            )
            strata = st.number_input(
                "Strata",
                min_value=1,
                value=8,
                step=1,
                help="Number of bins used in stratified ordering.",
            )
            max_workers_raw = st.text_input(
                "Max workers (blank = auto)",
                value="64",
                help="Parallel worker count for model calls; blank lets Palimpzest choose.",
            )

        c4, c5 = st.columns(2)
        with c4:
            k = st.number_input(
                "MAB k",
                min_value=1,
                value=6,
                step=1,
                help="Initial number of candidate operators on the MAB frontier.",
            )
            j = st.number_input(
                "MAB j",
                min_value=1,
                value=4,
                step=1,
                help="Minimum samples per operator before MAB pruning.",
            )
            models_raw = st.text_input(
                "Available models (space/comma separated, optional)",
                value="",
                help="Optional model allow-list; leave blank to use default auto-detected set.",
            )
        with c5:
            strata_composition = st.selectbox(
                "Strata composition",
                options=["composite", "exclusive"],
                help=(
                    "composite: combine selected features into one stratifier. "
                    "exclusive: run one stratified pass per feature (mutually exclusive)."
                ),
            )
            stratify_features = st.multiselect(
                "Stratification features",
                options=STRAT_FEATURE_COLUMNS,
                default=STRAT_FEATURE_COLUMNS,
                help="Feature columns used by the stratifier for composite/exclusive modes.",
            )
            random_only = st.checkbox(
                "Random only",
                help="Skip stratified runs and execute baseline random ordering only.",
            )
            stratified_only = st.checkbox(
                "Stratified only",
                help="Skip random baseline and execute stratified runs only.",
            )
            no_progress = st.checkbox(
                "No progress bars",
                value=False,
                help="Disable progress bars in script output.",
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
        random_only=random_only,
        stratified_only=stratified_only,
        no_progress=no_progress,
        output_csv=output_csv.strip() or None,
    )

    st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")

    repo_root = Path(__file__).resolve().parents[1]
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
    st.text(proc.stdout or "(empty)")
    st.subheader("Stderr")
    st.text(proc.stderr or "(empty)")


if __name__ == "__main__":
    main()
