#!/usr/bin/env python3
"""
Simple Streamlit UI for running sentinel A/B sampling experiments.

Run:
    streamlit run scripts/experiment_runner_ui.py
"""

from __future__ import annotations

import json
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

        c4, c5, c6 = st.columns(3)
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
        with c6:
            train_selection = st.selectbox(
                "Train set selection",
                options=["prefix", "random", "stratified"],
                help=(
                    "How train docs are chosen from eval docs: prefix (first N), "
                    "random, or stratified for diversity."
                ),
            )
            train_selection_strata = st.number_input(
                "Train selection strata",
                min_value=1,
                value=8,
                step=1,
                help="Only used when train set selection is stratified.",
            )
            train_selection_features = st.multiselect(
                "Train selection features",
                options=STRAT_FEATURE_COLUMNS,
                default=STRAT_FEATURE_COLUMNS,
                help="Features used to diversify the selected training subset.",
            )
            train_skew = st.selectbox(
                "Train skew policy",
                options=[
                    "natural",
                    "balanced_domain",
                    "min_one_per_domain",
                    "focus_domain",
                    "custom_domain_ratios",
                ],
                help="Target domain mix in training set.",
            )
            train_skew_focus_domain = st.text_input(
                "Train skew focus domain (for focus_domain)",
                value="",
                help="Example: cs",
            )
            train_skew_domain_ratios = st.text_input(
                "Train skew domain ratios (for custom_domain_ratios)",
                value="",
                help="Example: cs=0.5,biomedical=0.2,math=0.2,physics=0.1",
            )

        st.markdown("---")
        st.markdown("**Extraction fields** — define what to extract from each PDF. Each entry needs `name`, `type` (`str`, `bool`, `int`, `float`), and `desc`.")
        fields_json_raw = st.text_area(
            "Fields JSON",
            value=DEFAULT_FIELDS_JSON,
            height=220,
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
        if not train_selection_features:
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
