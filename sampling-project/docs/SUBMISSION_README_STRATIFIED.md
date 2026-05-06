# Submission README (Stratified Sampling Project)

This README describes the exact code layout, run commands, UI workflow, and cleanup steps used for:

**Sampling for Robust Query Optimization in Heterogeneous Document Collections**  
Sabrina Do, Katherine Li, Jenny Yu

---

## 1) Current project organization

### Core Palimpzest + experiment entrypoints
- `scripts/run_sentinel_sampling_ab.py` (main paired A/B runner)
- `scripts/run_stress_test_sweep.py` (automated stress sweep runner)
- `scripts/generate_stratifier_stress_dataset.py` (stress pool generator)
- `scripts/experiment_runner_ui.py` (Streamlit UI)
- `src/palimpzest/query/execution/feature_stratified_sampling.py` (sampling implementation)

### Analysis utilities
- `scripts/sampling-analysis/analyze_evidence_stage_results.py`
- `scripts/sampling-analysis/merge_stress_sweep_shards.py`
- `scripts/sampling-analysis/compare_stratify_feature_runs.py`
- `sampling-project/code/sampling-analysis/plot_presentation_summary_figures.py` 

### Results/artifacts root
- `results/sampling-results/stress_test_dataset_analysis/`

This reorganization keeps core Palimpzest functionality untouched while grouping project-specific analysis code and outputs.

---

## 2) Environment setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-experiments.txt
pip install -e .
```

If needed, generate feature table first:

```bash
python scripts/extract_features.py --scan "$(pwd)/papers" -o papers/paper_features.csv
```

---

## 3) General sanity/test command before running experiments

Use this to verify key scripts still import/parse correctly:

```bash
python -m py_compile \
  scripts/run_sentinel_sampling_ab.py \
  scripts/run_stress_test_sweep.py \
  scripts/generate_stratifier_stress_dataset.py \
  scripts/experiment_runner_ui.py \
  scripts/sampling-analysis/analyze_evidence_stage_results.py \
  scripts/sampling-analysis/merge_stress_sweep_shards.py \
  scripts/sampling-analysis/compare_stratify_feature_runs.py
```

---

## 4) Main CLI experiments used in report

### 4.1 Budget-resolved paired run (Table 1 / Figure 1 context)

```bash
python scripts/run_sentinel_sampling_ab.py \
  --papers papers \
  --features-csv papers/paper_features.csv \
  --train-n 20 \
  --eval-n 20 \
  --budgets 5 10 15 20 \
  --seed 42 \
  --strata 4 \
  --k 6 \
  --j 4 \
  --strata-composition cartesian \
  --stratify-features word_count section_count avg_sentence_length figure_count table_count complexity_score domain \
  --train-selection prefix \
  --train-selection-strata 4 \
  --train-selection-features word_count section_count avg_sentence_length figure_count table_count complexity_score domain \
  --train-skew natural \
  --output-csv papers/ab_results.csv
```

### 4.2 Stress-pool sweeps

```bash
python scripts/run_stress_test_sweep.py \
  --repo-root . \
  --out-dir results/sampling-results/stress_test_dataset_analysis \
  --max-cases 8
```

### 4.3 Representative-scale anchor

```bash
python scripts/run_stress_test_sweep.py \
  --repo-root . \
  --out-dir results/sampling-results/stress_test_dataset_analysis/paper_anchor_20260501 \
  --dataset-ids dhs_maj095 ultra_tail03 adv_conflict65 \
  --n-docs 100 \
  --train-n 24 \
  --eval-n 48 \
  --budgets 12 24 \
  --exp-seeds 101 102 \
  --strata 8 \
  --k 4 \
  --j 2 \
  --max-workers 2 \
  --no-progress
```

### 4.4 Post-processing / significance summaries

```bash
python scripts/sampling-analysis/analyze_evidence_stage_results.py \
  --analysis-dir results/sampling-results/stress_test_dataset_analysis/paper_anchor_20260501
```

---

## 5) UI workflow (feature ablation and interactive runs)

Launch UI:

```bash
python -m streamlit run scripts/experiment_runner_ui.py
```

### What was done in UI for feature ablation
Feature-ablation runs were performed by repeatedly running `run_sentinel_sampling_ab` settings through UI/CLI with:
- same `train_n`, `eval_n`, budgets, seed, and optimizer knobs
- only `stratify_features` changed (domain, word_count, section_count, complexity_score, avg_sentence_length, figure_count, table_count)

Then results were aggregated using:

```bash
python scripts/sampling-analysis/compare_stratify_feature_runs.py --glob-dir <folder_with_ablation_csvs>
```

---

## 6) Key report artifacts and where they live

- `results/sampling-results/stress_test_dataset_analysis/table_budget_indexed_experiment_summary.png`
- `results/sampling-results/stress_test_dataset_analysis/feature_ablation/feature_ablation_4_28_table.png`
- `results/sampling-results/stress_test_dataset_analysis/feature_ablation/feature_ablation_4_28_table.csv`
- `results/sampling-results/stress_test_dataset_analysis/appendix_table_D_pvalue_provenance.png`
- `results/sampling-results/stress_test_dataset_analysis/appendix_table_D_pvalue_provenance.csv`

---

## 7) What to delete safely before pushing (non-core cleanup)

Delete generated or temporary artifacts that are not needed for the report:

- `parse-answer-errors/`
- `results/batch_logs/queue_*.status`
- old intermediate run logs/csvs not cited in report figures/tables
- duplicated scratch images/tables not referenced in paper

Do **not** delete:
- core scripts under `scripts/` that are used by report experiments
- `src/palimpzest/...` stratified sampling implementation
- final artifacts explicitly cited in the report

Optional local cleanup command (review before running):

```bash
rm -rf parse-answer-errors
rm -f results/batch_logs/queue_*.status
```

---

## 8) Notes for reviewers

- All experiments are paired random-vs-stratified under matched settings.
- P-values are two-sided sign tests over paired deltas (see appendix p-value method note).
- Feature-ablation quality error is computed as `1 - mean_sentinel_quality` when explicit quality error is absent.
