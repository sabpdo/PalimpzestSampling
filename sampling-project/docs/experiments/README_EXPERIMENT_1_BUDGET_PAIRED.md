# Experiment 1: Budget-Resolved Paired Run

## Purpose
Produce the headline budget-indexed comparison for random vs stratified.

## Script
- `scripts/run_sentinel_sampling_ab.py`

## Core command (known-good)
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

## Important parameters
- `--budgets`: sample budgets compared in the paper.
- `--seed`: keep fixed (`42`) for direct reproducibility.
- `--strata-composition`: `cartesian` for joint-feature strata.
- `--stratify-features`: all seven features in headline setting.
- `--train-n` / `--eval-n`: split used for this experiment.

## Expected output
- CSV with random + stratified rows per budget.
- Used to generate:
  - cost/time table by budget
  - quality-error vs budget figure
