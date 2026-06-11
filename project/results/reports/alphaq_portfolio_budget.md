# AlphaQ Guarded Portfolio Selection Under A Materialization Budget

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_details.csv`.
Dedupe audit CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_dedupe_audit.csv`.
Figure: `/Users/caio/Deep-Learning/project/results/figures/alphaq_portfolio_budget.png`.

The evaluation portfolio is the six deployed AlphaQ objectives: `factor_count`, `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`, `depth_guarded_mixed_pair`, `t_preserving_frontier_pair`.

Policies `top-m` materialize the m candidates ranked best by the leave-one-target-out cheap-feature selector. Policies `guarded_top-m` always include the `factor_count` baseline candidate in the budget, so for m >= 2 they are never worse than the baseline pipeline in T-count by construction. `scope=groups` treats each (split, target) pair as one observation; `scope=targets` deduplicates to one group per target.

| scope | policy | budget | groups | oracle-T recovered | T wins | T losses | sign-test p | median T ratio | 95% CI | median QASM ratio | random recovery | perm. p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| groups | top1 | 1 | 54 | 53 | 21 | 1 | 5.48363e-06 | 1 | [0.984615, 1] | 1 | 31.672 | 0 |
| groups | top2 | 2 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 | 47.414 | 0.001 |
| groups | top3 | 3 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | top4 | 4 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | top5 | 5 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | top6 | 6 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | guarded_top1 | 1 | 54 | 33 | 0 | 0 |  | 1 | [1, 1] | 1 | 33.000 | 1 |
| groups | guarded_top2 | 2 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 1 | 46.631 | 0 |
| groups | guarded_top3 | 3 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | guarded_top4 | 4 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | guarded_top5 | 5 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | guarded_top6 | 6 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| groups | oracle_full_portfolio | 6 | 54 | 54 | 21 | 0 | 4.76837e-07 | 1 | [0.984615, 1] | 0.997561 |  |  |
| targets | top1 | 1 | 24 | 23 | 13 | 0 | 0.00012207 | 0.975066 | [0.914754, 1] | 0.997561 | 13.405 | 0 |
| targets | top2 | 2 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 | 19.849 | 0.0035 |
| targets | top3 | 3 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | top4 | 4 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | top5 | 5 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | top6 | 6 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | guarded_top1 | 1 | 24 | 10 | 0 | 0 |  | 1 | [1, 1] | 1 | 10.000 | 1 |
| targets | guarded_top2 | 2 | 24 | 23 | 13 | 0 | 0.00012207 | 0.975066 | [0.914754, 1] | 0.997561 | 18.122 | 0.0025 |
| targets | guarded_top3 | 3 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | guarded_top4 | 4 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | guarded_top5 | 5 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | guarded_top6 | 6 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |
| targets | oracle_full_portfolio | 6 | 24 | 24 | 14 | 0 | 6.10352e-05 | 0.964643 | [0.906593, 1] | 0.991781 |  |  |

## Dedupe Sensitivity

Default target-level dedupe keeps the most complete group, preferring external splits. The audit CSV lists every repeated target and compares that rule with an alternate best-oracle-T dedupe rule.

| dedupe policy | budget | targets | oracle-T recovered | T wins | T losses |
|---|---:|---:|---:|---:|---:|
| current_dedupe | 2 | 24 | 23 | 13 | 0 |
| current_dedupe | 3 | 24 | 24 | 14 | 0 |
| best_oracle_t_dedupe | 2 | 24 | 24 | 11 | 0 |
| best_oracle_t_dedupe | 3 | 24 | 24 | 11 | 0 |
