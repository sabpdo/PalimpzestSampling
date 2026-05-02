# Appendix tables (generated from repo CSVs)

Files (paths relative to `results/stress_test_dataset_analysis/`):

| File | Contents |
|------|----------|
| `appendix_table_A_stress_sweep_raw.csv` | One row per **dataset × seed × budget × mode** for the bundled stress sweep (`stress_ab_all_runs.csv`). Key metrics + stratifier settings columns. |
| `appendix_table_B_paired_deltas_stress_sweep.csv` | Same sweep collapsed to **paired deltas** (stratified − random) per dataset, seed, budget. |
| `appendix_table_C_signif_search_pilot_paired_deltas.csv` | Paired deltas from the harder-fields pilot under `signif_search/` (seeds 80–89, budgets 2 & 4). |

**Source sweep settings** for Table A/B are also in `sweep_run_meta.json` (train/eval, budgets, seeds, `fields_json`).

**Paper anchor** (`paper_anchor_20260501/`): when `stress_ab_all_runs.csv` appears there, regenerate appendix rows or add a second appendix table from that directory.

To rebuild Tables A–C after updating `stress_ab_all_runs.csv`, reuse the same pandas logic that produced them (groupby `dataset_id`, `exp_seed`, `sample_budget`; split rows by `mode`; subtract stratified − random).
