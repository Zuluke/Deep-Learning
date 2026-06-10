# AlphaQ Guarded Portfolio Selection Under A Materialization Budget

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_portfolio_budget_details.csv`.
Figure: `/Users/caio/Deep-Learning/project/results/figures/alphaq_portfolio_budget.png`.

The portfolio is the four deployed AlphaQ objectives: `factor_count`, `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`.

Policies `top-m` materialize the m candidates ranked best by the leave-one-target-out cheap-feature selector. Policies `guarded_top-m` always include the `factor_count` baseline candidate in the budget, so for m >= 2 they are never worse than the baseline pipeline in T-count by construction. `scope=groups` treats each (split, target) pair as one observation; `scope=targets` deduplicates to one group per target.

| scope | policy | budget | groups | oracle-T recovered | T wins | T losses | sign-test p | median T ratio | 95% CI | median QASM ratio | random recovery | perm. p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| groups | top1 | 1 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 | 17.420 | 0 |
| groups | top2 | 2 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 | 26.526 | 0.001 |
| groups | top3 | 3 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | top4 | 4 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | guarded_top1 | 1 | 32 | 20 | 0 | 0 |  | 1 | [1, 1] | 1 | 20.000 | 1 |
| groups | guarded_top2 | 2 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 | 26.593 | 0 |
| groups | guarded_top3 | 3 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | guarded_top4 | 4 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| groups | oracle_full_portfolio | 4 | 32 | 32 | 12 | 0 | 0.000244141 | 1 | [0.965517, 1] | 1 |  |  |
| targets | top1 | 1 | 17 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 1 | 9.870 | 0 |
| targets | top2 | 2 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 1 | 14.161 | 0.0295 |
| targets | top3 | 3 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 0.984496 |  |  |
| targets | top4 | 4 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 0.984496 |  |  |
| targets | guarded_top1 | 1 | 17 | 9 | 0 | 0 |  | 1 | [1, 1] | 1 | 9.000 | 1 |
| targets | guarded_top2 | 2 | 17 | 16 | 7 | 0 | 0.0078125 | 1 | [0.9, 1] | 1 | 13.134 | 0.031 |
| targets | guarded_top3 | 3 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 0.984496 |  |  |
| targets | guarded_top4 | 4 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 0.984496 |  |  |
| targets | oracle_full_portfolio | 4 | 17 | 17 | 8 | 0 | 0.00390625 | 1 | [0.864865, 1] | 0.984496 |  |  |
