# Experiment 2: Stress-Pool Sweeps

## Purpose
Test robustness under controlled heterogeneity/skew and identify saturation regimes.

## Scripts
- `scripts/run_stress_test_sweep.py`
- `scripts/generate_stratifier_stress_dataset.py`

## Core command
```bash
python scripts/run_stress_test_sweep.py \
  --repo-root . \
  --out-dir results/sampling-results/stress_test_dataset_analysis \
  --max-cases 8
```

## Important parameters
- `--max-cases`: cap number of stress presets.
- `--budgets`: if omitted, script default applies; can override.
- `--exp-seeds`: one or more seeds for paired robustness.
- `--strata`, `--k`, `--j`: optimizer/sampling knobs.
- `--dataset-ids` (optional): run selected presets only.
- `--n-docs` (optional): override pool size.

## Outputs
- `stress_ab_all_runs.csv`
- `stress_significance_summary.csv`
- `stress_summary_stratified_minus_random.csv`
- `stress_dataset_manifest.csv`
- `figures/*.png`

These are consumed by analysis scripts and report tables.
