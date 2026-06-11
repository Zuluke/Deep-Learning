# AlphaQ Guarded Portfolio Selection Under A Materialization Budget

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_details.csv`.
Dedupe audit CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_dedupe_audit.csv`.
Figure: `/Users/caio/Deep-Learning/project/results/figures/alphaq_portfolio_budget.png`.

The portfolio is the four deployed AlphaQ objectives: `factor_count`, `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`.

Policies `top-m` materialize the m candidates ranked best by the leave-one-target-out cheap-feature selector. Policies `guarded_top-m` always include the `factor_count` baseline candidate in the budget, so for m >= 2 they are never worse than the baseline pipeline in T-count by construction. `scope=groups` treats each (split, target) pair as one observation; `scope=targets` deduplicates to one group per target.

| scope | policy | budget | groups | oracle-T recovered | T wins | T losses | sign-test p | median T ratio | 95% CI | median QASM ratio | random recovery | perm. p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| groups | top1 | 1 | 33 | 32 | 12 | 1 | 0.00170898 | 1 | [0.965517, 1] | 1 | 18.761 | 0 |
| groups | top2 | 2 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 | 27.849 | 0.001 |
| groups | top3 | 3 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | top4 | 4 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | guarded_top1 | 1 | 33 | 21 | 0 | 0 |  | 1 | [1, 1] | 1 | 21.000 | 1 |
| groups | guarded_top2 | 2 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 | 27.658 | 0.0005 |
| groups | guarded_top3 | 3 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.964912, 1] | 1 |  |  |
| groups | guarded_top4 | 4 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | oracle_full_portfolio | 4 | 33 | 33 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| targets | top1 | 1 | 18 | 17 | 7 | 0 | 0.0078125 | 1 | [0.942857, 1] | 1 | 10.842 | 0 |
| targets | top2 | 2 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 1 | 15.132 | 0.0175 |
| targets | top3 | 3 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 0.990854 |  |  |
| targets | top4 | 4 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 0.990854 |  |  |
| targets | guarded_top1 | 1 | 18 | 10 | 0 | 0 |  | 1 | [1, 1] | 1 | 10.000 | 1 |
| targets | guarded_top2 | 2 | 18 | 17 | 7 | 0 | 0.0078125 | 1 | [0.942857, 1] | 1 | 14.198 | 0.032 |
| targets | guarded_top3 | 3 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 0.990854 |  |  |
| targets | guarded_top4 | 4 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 0.990854 |  |  |
| targets | oracle_full_portfolio | 4 | 18 | 18 | 8 | 0 | 0.00390625 | 1 | [0.9, 1] | 0.990854 |  |  |

## Dedupe Sensitivity

Default target-level dedupe keeps the most complete group, preferring external splits. The audit CSV lists every repeated target and compares that rule with an alternate best-oracle-T dedupe rule.

| dedupe policy | budget | targets | oracle-T recovered | T wins | T losses |
|---|---:|---:|---:|---:|---:|
| current_dedupe | 2 | 18 | 17 | 7 | 0 |
| current_dedupe | 3 | 18 | 18 | 8 | 0 |
| best_oracle_t_dedupe | 2 | 18 | 18 | 5 | 0 |
| best_oracle_t_dedupe | 3 | 18 | 18 | 5 | 0 |
