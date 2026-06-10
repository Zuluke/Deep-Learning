# AlphaQ circuit-conditioned objective selector

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_details.csv`.

This selector uses only pre-run AlphaQ/circuit tensor features. ZX and materialized metrics are labels/evaluation targets only.

| policy | groups | exact oracle | T non-worse | T wins | QASM wins | runtime wins | median T ratio | median QASM ratio | selected objectives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oracle_posthoc | 33 | 33 | 33 | 12 | 16 | 2 | 1 | 1 | factor_count=14, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=7 |
| baseline_factor_count | 33 | 14 | 33 | 0 | 0 | 0 | 1 | 1 | factor_count=33 |
| best_fixed_loto | 33 | 14 | 33 | 0 | 0 | 0 | 1 | 1 | factor_count=33 |
| circuit_knn3 | 33 | 12 | 30 | 0 | 3 | 0 | 1 | 1 | factor_count=28, factor_count_pair_cap=1, mixed_pair=4 |
| circuit_1nn | 31 | 10 | 25 | 4 | 8 | 0 | 1 | 1 | factor_count=12, factor_count_pair_cap=12, mixed_pair=7 |
| circuit_softmax | 30 | 10 | 24 | 1 | 5 | 0 | 1 | 1 | factor_count=18, factor_count_pair_cap=10, mixed_pair=2 |
| shuffled_centroid_control | 25 | 6 | 19 | 2 | 5 | 0 | 1 | 1 | factor_count=10, factor_count_pair_cap=8, frontier_pair=2, mixed_pair=5 |
| circuit_nearest_centroid | 27 | 3 | 16 | 3 | 9 | 1 | 1 | 1 | factor_count=6, factor_count_pair_cap=4, frontier_pair=6, mixed_pair=11 |
