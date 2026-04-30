# Stratified Sampling Experiment Guide

This guide is specific to the Palimpzest sampling experiments in this repo (Project 3: importance/diversity sampling for query optimization).

It covers:
- running feature extraction
- running experiments from CLI and Streamlit UI
- understanding key parameters
- reading outputs and plots
- running long jobs asynchronously on a remote machine

---

## 1) What this experiment does

The experiment compares document ordering/sampling strategies for sentinel optimization:
- `random` baseline
- `stratified` ordering using document features

At each sample budget, the script runs one or more methods, then records:
- quality estimates (`mean_sentinel_quality`, `mean_plan_quality`)
- optimization/execution cost
- optimization/execution runtime

Main script:
- `scripts/run_sentinel_sampling_ab.py`

Main UI:
- `scripts/experiment_runner_ui.py`

---

## 2) Setup (known-good commands)

From repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-experiments.txt
pip install -e .
```

> Important: `palimpzest` currently requires Python `>=3.12,<3.14`.
> If your default `python3` is `3.14.x`, installs will fail.

If you use `uv`, adapt as needed (`uv pip install ...`).

### If you got `requires a different Python: 3.14.x not in '<3.14,>=3.12'`

Use one of these:

```bash
# Option A (recommended): use Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-experiments.txt
pip install -e .
```

```bash
# Option B (macOS Homebrew): install Python 3.12 first
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-experiments.txt
pip install -e .
```

If `streamlit` is not found, use:

```bash
python -m streamlit run scripts/experiment_runner_ui.py
```

---

## 3) Build feature table first

The stratified sampler depends on a features CSV for each PDF.

Important: build this CSV using an **absolute scan path** to avoid path mismatches.

Known-good command:

```bash
python scripts/extract_features.py --scan "/Users/sabrinado/PalimpzestSampling/papers" -o "papers/paper_features.csv"
```

Expected feature columns include:
- `word_count`
- `section_count`
- `avg_sentence_length`
- `figure_count`
- `table_count`
- `complexity_score`
- `domain`

---

## 4) Run from CLI

Example run:

```bash
python3 scripts/run_sentinel_sampling_ab.py \
  --papers papers \
  --features-csv papers/paper_features.csv \
  --train-n 20 \
  --eval-n 20 \
  --budgets 5 10 15 20 \
  --strata 8 \
  --k 6 \
  --j 4 \
  --strata-composition cartesian \
  --train-selection stratified \
  --train-selection-strata 8 \
  --train-selection-features word_count section_count avg_sentence_length figure_count table_count complexity_score domain \
  --train-skew natural \
  --output-csv papers/ab_results.csv
```

### `--strata-composition` modes

- `cartesian`: stratify by Cartesian product of selected features
- `composite`: combine selected features into one composite stratifier
- `exclusive`: run one stratified pass per single feature

---

## 5) Run from Streamlit UI

Start UI:

```bash
python -m streamlit run scripts/experiment_runner_ui.py
```

The UI lets you configure:
- dataset paths and output path
- budgets, seed, strata, MAB `k/j`
- stratification features and composition mode
- train selection strategy and train skew policy
- extraction fields JSON (`name`, `type`, `desc`)

After each completed run, the UI shows:
- command preview
- stdout/stderr
- plots generated from the output CSV
- download buttons for CSV and plots

---

## 6) Plot outputs in UI (paper-friendly)

The UI reads your output CSV and renders these plots:
- **Plan quality vs sample budget**
- **Absolute quality error vs sample budget** (relative to max-budget quality per mode)
- **Cost-quality tradeoff**
- **Total runtime vs sample budget**

Download buttons are available for:
- results CSV
- each PNG figure

These are useful for your report sections on:
- sample efficiency
- estimation accuracy
- tradeoff analysis (quality vs cost/runtime)

---

## 7) Output CSV schema (core fields)

Saved by `--output-csv`:
- `sample_budget`
- `mode`
- `optimization_cost`
- `optimization_time_s`
- `plan_execution_cost`
- `plan_execution_time_s`
- `total_cost`
- `total_time_s`
- `mean_sentinel_quality`
- `mean_plan_quality`
- `sentinel_plan_count`
- `final_plan_count`
- `stratify_features`
- `strata_composition`
- `strata_feature`
- `train_selection`
- `train_selection_features`
- `train_skew`
- `train_skew_focus_domain`
- `train_skew_domain_ratios`

---

## 8) Asynchronous runs on a remote machine

For long experiments, run in background:

```bash
nohup python3 scripts/run_sentinel_sampling_ab.py \
  --papers papers \
  --features-csv papers/paper_features.csv \
  --train-n 20 \
  --eval-n 20 \
  --budgets 5 10 15 20 25 30 \
  --strata 8 \
  --k 6 \
  --j 4 \
  --output-csv runs/ab_results.csv \
  > runs/run.log 2>&1 &
```

Check status:

```bash
ps -ef | rg run_sentinel_sampling_ab.py
tail -f runs/run.log
```

Why this is async:
- `&` backgrounds the process
- `nohup` keeps it running after SSH disconnects

---

## 9) Suggested baseline experiment matrix

To structure your paper results, sweep:
- budgets: `5 10 15 20 25 30`
- `strata`: `4, 8, 12`
- feature sets:
  - all features
  - each feature alone (ablation)
  - with/without `domain`
- train selection:
  - `prefix`, `random`, `stratified`
- train skew:
  - `natural`, `balanced_domain`

Track each run in one sheet with:
- run id / date
- command
- output csv path
- key result deltas vs random baseline

---

## 10) Troubleshooting

- **Feature CSV not found**: run `extract_features.py --scan ... -o ...` first.
- **Missing columns**: verify selected feature names match the CSV headers.
- **Very slow runs**: reduce `eval_n`, budgets, and `max_workers`; run overnight in background.
- **Empty/invalid plot section in UI**: confirm `--output-csv` path exists and has non-empty rows.
- **Quality fields null**: check validator/model configuration and script stderr.

