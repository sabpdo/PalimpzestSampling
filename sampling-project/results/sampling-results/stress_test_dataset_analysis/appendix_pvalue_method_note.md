# P-value calculation note

All reported p-values are two-sided sign-test p-values over paired deltas.

For each metric:
- Define a paired delta per (dataset, seed, budget): Δ = metric_stratified - metric_random.
- Count wins/losses according to metric direction:
  - quality metrics (higher better): wins = #Δ>0, losses=#Δ<0
  - time/cost metrics (lower better): wins = #Δ<0, losses=#Δ>0
- Ties (#Δ=0) are excluded from the binomial count n = wins + losses.
- Two-sided p-value uses exact Binomial(n, 0.5):
  p = 2 * sum_{i=0}^{min(wins,losses)} C(n,i) * 0.5^n

Implementation source: scripts/run_stress_test_sweep.py (`_two_sided_binom_p`, `_significance_summary`) and signif_search snapshot script.
