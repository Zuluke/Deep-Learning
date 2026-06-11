# Verification Status Of Portfolio Selection Wins

Policy: `guarded_top2` (scope `targets`). CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_verification_impact.csv`.

T-count wins vs baseline: 13; wins whose selected candidate is fully proven: 6.

| target | split | selected objective | T ratio | outcome | selected proof | baseline proof |
|---|---|---|---:|---|---|---|
| barenco_tof_3 | external_journal_k6_backfill | t_preserving_frontier_pair | 1.0 | tie | equal | equal |
| cuccaro_adder_n3 | external_journal_k6_backfill | t_preserving_frontier_pair | 1.0 | tie | equal-block-exact | equal-block-exact |
| cuccaro_adder_n4 | external_article_extended | mixed_pair | 1.0 | tie | equal-block-exact | equal-block-exact |
| gf_2pow2_mult | external_article_core | factor_count | 1.0 | tie | equal | equal |
| gf_2pow5_mult | external_journal_full_gf_2pow5_mult_long | factor_count | 1.0 | tie | unverified | unverified |
| hamming_weight_n4 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| hamming_weight_n5 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| hamming_weight_n6 | external_article_extended | frontier_pair | 1.0 | tie | equal | equal |
| mod_5_4 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| nc_tof_3 | external_journal_k6_backfill | factor_count | 1.0 | tie | equal | equal |
| nc_tof_5 | external_journal_full_nc_tof_5_long | factor_count | 1.0 | tie | equal | equal |
| barenco_tof_4 | external_night_long | mixed_pair | 0.65625 | win | equal | equal |
| barenco_tof_5 | external_journal_k6_frontier | frontier_pair | 0.9478260869565217 | win | inconclusive | inconclusive |
| csla_mux_3 | external_journal_k6_frontier | t_preserving_frontier_pair | 0.8303030303030303 | win | inconclusive | inconclusive |
| cuccaro_adder_n6 | external_journal_k6_frontier | t_preserving_frontier_pair | 0.9846153846153847 | win | inconclusive | inconclusive |
| gf_2pow3_mult | internal | factor_count_pair_cap | 0.9428571428571428 | win | unverified | equal |
| gf_2pow4_mult | external_article_extended | factor_count_pair_cap | 0.9655172413793104 | win | equal-block-exact | equal-block-exact |
| gf_2pow7_mult | external_journal_k6_frontier | t_preserving_frontier_pair | 0.9147540983606557 | win | inconclusive | inconclusive |
| hamming_weight_n7 | external_article_extended | frontier_pair | 0.8235294117647058 | win | equal | equal |
| hamming_weight_n8 | external_journal_k6_frontier | t_preserving_frontier_pair | 0.9065934065934066 | win | inconclusive | inconclusive |
| mod_mult_55 | external_article_extended | mixed_pair | 0.9 | win | equal-numeric | equal |
| nc_tof_4 | external_article_core | factor_count_pair_cap | 0.8648648648648649 | win | equal | equal |
| unary_iteration_n3 | external_journal_k6_frontier | t_preserving_frontier_pair | 0.9637681159420289 | win | inconclusive | inconclusive |
| vbe_adder_3 | external_article_repair2_vbe | factor_count_pair_cap | 0.7846153846153846 | win | equal | equal |
