# UI Workflow (Streamlit)

## Prerequisite: `.env` for runs/history

Before launching the UI, create a `.env` file in the repo root with the API keys/env vars used by your model providers and experiment logging setup.

Why this matters:
- the UI can load prior run history/state from experiment outputs,
- new runs may fail or return empty results if provider keys are missing,
- run diagnostics/history panels are most useful when the same `.env` is used consistently across sessions.

## Launch
```bash
python -m streamlit run scripts/experiment_runner_ui.py
```

## What the UI is used for
- configuring paired random vs stratified runs
- adjusting budgets/seeds/sampling knobs
- selecting stratification composition/features
- configuring train selection and train skew controls
- setting extraction fields JSON
- generating and previewing stress pools

## Typical flow
1. Set papers/features paths.
2. Configure run parameters (train/eval, budgets, seed, strata, `k`, `j`).
3. Choose sampling configuration (`strata_composition`, `stratify_features`).
4. Run experiment.
5. Review generated plots and CSV output paths.

## Feature ablation via UI
Repeat the same run while changing only `stratify_features`:
- domain
- word_count
- section_count
- complexity_score
- avg_sentence_length
- figure_count
- table_count

Export each run CSV, then aggregate with:
```bash
python scripts/sampling-analysis/compare_stratify_feature_runs.py --glob-dir <ablation_csv_dir>
```

## UI script
- `scripts/experiment_runner_ui.py`
