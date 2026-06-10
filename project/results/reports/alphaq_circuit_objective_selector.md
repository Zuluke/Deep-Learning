# AlphaQ circuit-conditioned objective selector

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_details.csv`.

This selector uses only pre-run AlphaQ/circuit tensor features. ZX and materialized metrics are labels/evaluation targets only.

| policy | groups | exact oracle | T non-worse | T wins | QASM wins | runtime wins | median T ratio | median QASM ratio | selected objectives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oracle_posthoc | 32 | 32 | 32 | 12 | 15 | 2 | 1 | 1 | factor_count=14, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=6 |
| baseline_factor_count | 32 | 14 | 32 | 0 | 0 | 0 | 1 | 1 | factor_count=32 |
| best_fixed_loto | 32 | 14 | 32 | 0 | 0 | 0 | 1 | 1 | factor_count=32 |
| circuit_knn3 | 32 | 12 | 29 | 0 | 3 | 0 | 1 | 1 | factor_count=27, factor_count_pair_cap=1, mixed_pair=4 |
| circuit_1nn | 30 | 9 | 24 | 4 | 8 | 0 | 1 | 1 | factor_count=11, factor_count_pair_cap=12, mixed_pair=7 |
| circuit_softmax | 30 | 9 | 21 | 4 | 6 | 1 | 1 | 1 | factor_count=14, factor_count_pair_cap=8, frontier_pair=2, mixed_pair=6 |
| shuffled_centroid_control | 27 | 6 | 19 | 5 | 9 | 2 | 1 | 1 | factor_count=7, factor_count_pair_cap=10, frontier_pair=5, mixed_pair=5 |
| circuit_nearest_centroid | 26 | 5 | 17 | 2 | 7 | 1 | 1 | 1 | factor_count=10, factor_count_pair_cap=4, frontier_pair=5, mixed_pair=7 |
