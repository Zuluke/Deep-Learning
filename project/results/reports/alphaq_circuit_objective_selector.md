# AlphaQ circuit-conditioned objective selector

Summary CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_summary.csv`.
Detail CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_circuit_objective_selector_details.csv`.

This selector uses only pre-run AlphaQ/circuit tensor features. ZX and materialized metrics are labels/evaluation targets only.

| policy | groups | exact oracle | T non-worse | T wins | QASM wins | runtime wins | median T ratio | median QASM ratio | selected objectives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oracle_posthoc | 54 | 54 | 54 | 21 | 27 | 15 | 1 | 0.998 | depth_guarded_mixed_pair=6, factor_count=21, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=7, t_preserving_frontier_pair=8 |
| circuit_knn3 | 51 | 25 | 49 | 7 | 6 | 5 | 1 | 1 | factor_count=40, factor_count_pair_cap=3, mixed_pair=2, t_preserving_frontier_pair=6 |
| circuit_softmax | 52 | 25 | 52 | 6 | 4 | 5 | 1 | 1 | factor_count=45, factor_count_pair_cap=1, t_preserving_frontier_pair=6 |
| baseline_factor_count | 54 | 21 | 54 | 0 | 0 | 0 | 1 | 1 | factor_count=54 |
| best_fixed_loto | 54 | 21 | 54 | 0 | 0 | 0 | 1 | 1 | factor_count=54 |
| circuit_1nn | 40 | 16 | 34 | 10 | 11 | 4 | 1 | 1 | factor_count=15, factor_count_pair_cap=12, mixed_pair=8, t_preserving_frontier_pair=5 |
| circuit_nearest_centroid | 31 | 13 | 21 | 7 | 10 | 8 | 1 | 1 | depth_guarded_mixed_pair=4, factor_count=13, factor_count_pair_cap=2, frontier_pair=2, mixed_pair=4, t_preserving_frontier_pair=6 |
| shuffled_centroid_control | 27 | 9 | 19 | 3 | 9 | 6 | 1 | 1 | depth_guarded_mixed_pair=2, factor_count=9, factor_count_pair_cap=4, frontier_pair=3, mixed_pair=6, t_preserving_frontier_pair=3 |
