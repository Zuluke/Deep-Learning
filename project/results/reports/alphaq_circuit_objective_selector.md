# AlphaQ circuit-conditioned objective selector

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_details.csv`.

This selector uses only pre-run AlphaQ/circuit tensor features. ZX and materialized metrics are labels/evaluation targets only.

| policy | groups | exact oracle | T non-worse | T wins | QASM wins | runtime wins | median T ratio | median QASM ratio | selected objectives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oracle_posthoc | 29 | 29 | 29 | 10 | 13 | 0 | 1 | 1 | factor_count=13, factor_count_pair_cap=9, frontier_pair=2, mixed_pair=5 |
| baseline_factor_count | 29 | 13 | 29 | 0 | 0 | 0 | 1 | 1 | factor_count=29 |
| best_fixed_loto | 29 | 13 | 29 | 0 | 0 | 0 | 1 | 1 | factor_count=29 |
| circuit_softmax | 27 | 10 | 26 | 1 | 4 | 0 | 1 | 1 | factor_count=18, factor_count_pair_cap=8, frontier_pair=1 |
| circuit_knn3 | 29 | 9 | 23 | 0 | 3 | 0 | 1 | 1 | factor_count=21, factor_count_pair_cap=1, mixed_pair=7 |
| circuit_1nn | 26 | 5 | 18 | 3 | 7 | 0 | 1 | 1 | factor_count=6, factor_count_pair_cap=9, frontier_pair=1, mixed_pair=10 |
| circuit_nearest_centroid | 24 | 5 | 17 | 1 | 5 | 0 | 1 | 1 | factor_count=12, factor_count_pair_cap=2, frontier_pair=3, mixed_pair=7 |
| shuffled_centroid_control | 28 | 4 | 19 | 3 | 8 | 0 | 1 | 1 | factor_count=10, factor_count_pair_cap=7, frontier_pair=3, mixed_pair=8 |
