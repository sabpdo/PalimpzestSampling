# Experiment 4: Feature Ablation

## Purpose
Measure how resource/quality deltas change when stratification is performed on one feature axis at a time.

## How runs were produced
Runs are repeated paired A/B runs using `run_sentinel_sampling_ab.py` with all settings fixed except:
- `--stratify-features` (single feature per run)

Primary cohort used in paper:
- date: 4/28
- `train_n=20`, `eval_n=20`
- budgets `{5,10,15,20}`
- seed `42`
- `strata=4`, `k=6`, `j=4`
- `strata_composition=cartesian`

## Aggregation script
- `scripts/sampling-analysis/compare_stratify_feature_runs.py`

Example:
```bash
python scripts/sampling-analysis/compare_stratify_feature_runs.py \
  --glob-dir /path/to/feature-ablation-csvs
```

## Report artifacts
- `results/sampling-results/stress_test_dataset_analysis/feature_ablation/feature_ablation_4_28_table.csv`
- `results/sampling-results/stress_test_dataset_analysis/feature_ablation/feature_ablation_4_28_table.png`

## Presentation figures
Use the combined summary plotting script to generate slide-ready visuals from the
feature-ablation and stress-significance summary CSVs:

```bash
python sampling-project/code/sampling-analysis/plot_presentation_summary_figures.py
```

This writes:
- `results/sampling-results/stress_test_dataset_analysis/figures/feature_ablation_summary_bars.png`
- `results/sampling-results/stress_test_dataset_analysis/figures/feature_ablation_summary_bars.pdf`
- `results/sampling-results/stress_test_dataset_analysis/figures/stress_significance_stacked_wlt.png`
- `results/sampling-results/stress_test_dataset_analysis/figures/stress_significance_stacked_wlt.pdf`

## Metric note
If explicit `quality_error` is missing in CSVs, use:
- `quality_error := 1 - mean_sentinel_quality`
