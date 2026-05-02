# Paper anchor run (`paper_anchor_20260501`)

This folder holds a **representative-scale** stress evaluation relative to the faster “many presets / smaller pools” sweeps.

## Why this exists

- **Fast sweep**: Many synthetic presets with smaller `n_docs` and moderate train/eval splits are useful as **controlled stress tests**, but they are weaker evidence for “behavior at corpus-like scale” because the evaluation subset is small relative to what practitioners imagine.

- **Anchor run**: One coordinated sweep with **larger pools** (`n_docs` ≈ 100, bounded by available PDFs under `papers/` ~299 files), **train_n=24**, **eval_n=48**, and **budgets 12 and 24**. Random vs stratified remain matched on budgets and pipeline settings.

## Presets included (this launch)

Because the fast shard merge was not finished yet, selection uses **the strongest heterogeneity regimes** in `DEFAULT_CASES` rather than post-hoc ranking:

| `dataset_id`     | Rationale |
|------------------|-----------|
| `dhs_maj095`     | Strong domain skew (95/5-style regime). |
| `ultra_tail03`   | Ultra-rare “needle” tail vs bulk majority. |
| `adv_conflict65` | Adversarial mismatch between feature regimes (hard stratification case). |

If your merged fast sweep later shows another preset dominating deltas, re-run anchor with `--dataset-ids …` for those IDs using the same scale knobs.

## Limitations (use verbatim ideas in your paper)

1. **Synthetic skew**: Pools are **fabricated** feature distributions over real PDFs; they stress the stratifier but are not a guarantee about arbitrary production corpora.

2. **Still bounded \(N\)**: Even at `n_docs≈100`, this is not “millions of documents”; claims should stay scoped to **heterogeneous finite pools** under your extraction task (`SWEEP_FIELDS_JSON`).

3. **Sign test scope**: Aggregated significance pools paired comparisons across seeds/budgets; interpret alongside **per-scenario** CSVs and plots.

4. **Cost/time variance**: Remaining LLM/API jitter means wide runtime tails; prefer **direction + effect sizes + paired deltas** over single-run headlines.

## Outputs to cite

- `sweep_run_meta.json` — exact CLI-equivalent settings.
- `stress_dataset_manifest.csv` — pool provenance and stress params.
- `runs/*_ab.csv` — raw A/B rows per preset × seed.
- `stress_ab_all_runs.csv` — pooled table.
- `stress_significance_summary.csv` — sign-test style summary on pooled pairs.
- `figures/` — delta plots from the sweep script.
