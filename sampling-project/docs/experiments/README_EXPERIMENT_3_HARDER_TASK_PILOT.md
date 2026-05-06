# Experiment 3: Harder-Task Pilot

## Purpose
Break quality saturation and test whether efficiency signals remain under a harder extraction task.

## How it was run
This pilot was executed via repeated paired runs (random vs stratified) at tight budgets and aggregated under:
- `results/sampling-results/stress_test_dataset_analysis/signif_search/`

The pilot uses:
- harder extraction fields JSON than the default sweep
- small train/eval settings
- paired deltas summarized with a two-sided sign test

## Recompute significance
Use this script to recompute the pilot significance outputs directly from paired deltas:

- Script: `sampling-project/code/sampling-analysis/summarize_harder_pilot_significance.py`
- Input: `sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/paired_deltas_running.csv`

Run:

`/Users/sabrinado/PalimpzestSampling/.venv/bin/python sampling-project/code/sampling-analysis/summarize_harder_pilot_significance.py`

This regenerates:
- `sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/significance_snapshot.csv`
- `sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/per_budget_snapshot.csv`
- `sampling-project/results/sampling-results/stress_test_dataset_analysis/signif_search/evidence_snapshot.md`

## Key output files
- `signif_search/paired_deltas_running.csv`
- `signif_search/significance_snapshot.csv`
- `signif_search/per_budget_snapshot.csv`
- `signif_search/evidence_snapshot.md`

## Interpretation notes
- `dcost < 0` means stratified cheaper than random.
- `dtime < 0` means stratified faster.
- quality deltas depend on the score source and can still be mixed.
