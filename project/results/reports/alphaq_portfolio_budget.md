# AlphaQ Guarded Portfolio Selection Under A Materialization Budget

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_details.csv`.
Figure: `/Users/caio/Deep-Learning/project/results/figures/alphaq_portfolio_budget.png`.

The portfolio is the four deployed AlphaQ objectives: `factor_count`, `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`.

Policies `top-m` materialize the m candidates ranked best by the leave-one-target-out cheap-feature selector. Policies `guarded_top-m` always include the `factor_count` baseline candidate in the budget, so for m >= 2 they are never worse than the baseline pipeline in T-count by construction. `scope=groups` treats each (split, target) pair as one observation; `scope=targets` deduplicates to one group per target.

| scope | policy | budget | groups | oracle-T recovered | T wins | T losses | sign-test p | median T ratio | 95% CI | median QASM ratio | random recovery | perm. p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| groups | top1 | 1 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 | 15.547 | 0 |
| groups | top2 | 2 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 | 23.686 | 0.0025 |
| groups | top3 | 3 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 |  |  |
| groups | top4 | 4 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 |  |  |
| groups | guarded_top1 | 1 | 29 | 19 | 0 | 0 |  | 1 | [1, 1] | 1 | 19.000 | 1 |
| groups | guarded_top2 | 2 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 | 24.019 | 0.001 |
| groups | guarded_top3 | 3 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 |  |  |
| groups | guarded_top4 | 4 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 |  |  |
| groups | oracle_full_portfolio | 4 | 29 | 29 | 10 | 0 | 0.000976562 | 1 | [0.965517, 1] | 1 |  |  |
| targets | top1 | 1 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 1 | 8.694 | 0 |
| targets | top2 | 2 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 1 | 12.970 | 0.019 |
| targets | top3 | 3 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 0.992248 |  |  |
| targets | top4 | 4 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 0.992248 |  |  |
| targets | guarded_top1 | 1 | 16 | 9 | 0 | 0 |  | 1 | [1, 1] | 1 | 9.000 | 1 |
| targets | guarded_top2 | 2 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 1 | 12.489 | 0.005 |
| targets | guarded_top3 | 3 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 0.992248 |  |  |
| targets | guarded_top4 | 4 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 0.992248 |  |  |
| targets | oracle_full_portfolio | 4 | 16 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 0.992248 |  |  |
