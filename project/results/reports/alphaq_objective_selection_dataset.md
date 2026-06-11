# AlphaQ Objective-Selection Dataset

CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_objective_selection_dataset.csv`.

This dataset reframes the current evidence as supervised objective selection: each target/run contains one row per AlphaQ objective and an oracle label derived from the best materialized beam candidate.

## Bottom Line

Decision: `prototype-ready`.

Train-ready target groups: 54/68.
External train-ready target groups: 46.
Oracle objective distribution: depth_guarded_mixed_pair=6, factor_count=21, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=7, t_preserving_frontier_pair=8.

We can proceed with a prototype selector and leave-one-target validation. The dataset is still too small for a high-capacity deep model or a broad journal-level generalization claim.

## Target Labels

| split | target | train ready | materialized objectives | oracle objective |
|---|---|---:|---:|---|
| external_article_core | barenco_tof_4 | False | 1 | frontier_pair |
| external_article_core | gf_2pow2_mult | True | 4 | factor_count |
| external_article_core | hamming_weight_n4 | True | 4 | factor_count |
| external_article_core | hamming_weight_n5 | True | 4 | factor_count |
| external_article_core | mod_5_4 | True | 4 | factor_count |
| external_article_core | nc_tof_4 | True | 4 | frontier_pair |
| external_article_core | vbe_adder_3 | True | 3 | factor_count |
| external_article_extended | cuccaro_adder_n4 | True | 4 | factor_count_pair_cap |
| external_article_extended | cuccaro_adder_n5 | False | 0 | - |
| external_article_extended | gf_2pow4_mult | True | 4 | factor_count_pair_cap |
| external_article_extended | gf_2pow5_mult | False | 0 | - |
| external_article_extended | hamming_weight_n6 | True | 4 | factor_count_pair_cap |
| external_article_extended | hamming_weight_n7 | True | 4 | frontier_pair |
| external_article_extended | mod_mult_55 | True | 4 | mixed_pair |
| external_article_extended | nc_tof_5 | False | 0 | - |
| external_article_repair2_barenco | barenco_tof_4 | True | 2 | mixed_pair |
| external_article_repair2_vbe | vbe_adder_3 | True | 4 | factor_count_pair_cap |
| external_journal_full_cuccaro_adder_n4 | cuccaro_adder_n4 | True | 3 | factor_count |
| external_journal_full_cuccaro_adder_n5 | cuccaro_adder_n5 | False | 1 | factor_count |
| external_journal_full_gf_2pow4_mult | gf_2pow4_mult | True | 3 | factor_count_pair_cap |
| external_journal_full_gf_2pow5_mult | gf_2pow5_mult | False | 0 | - |
| external_journal_full_gf_2pow5_mult_long | gf_2pow5_mult | True | 2 | factor_count |
| external_journal_full_hamming_weight_n6 | hamming_weight_n6 | True | 3 | factor_count_pair_cap |
| external_journal_full_hamming_weight_n7 | hamming_weight_n7 | True | 3 | factor_count |
| external_journal_full_mod_mult_55 | mod_mult_55 | True | 3 | mixed_pair |
| external_journal_full_nc_tof_5 | nc_tof_5 | False | 0 | - |
| external_journal_full_nc_tof_5_long | nc_tof_5 | True | 2 | factor_count |
| external_journal_k6_backfill | barenco_tof_3 | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_backfill | barenco_tof_4 | True | 2 | factor_count |
| external_journal_k6_backfill | cuccaro_adder_n3 | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_backfill | cuccaro_adder_n4 | True | 2 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | gf_2pow2_mult | True | 3 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | gf_2pow3_mult | True | 2 | factor_count |
| external_journal_k6_backfill | gf_2pow4_mult | True | 2 | factor_count |
| external_journal_k6_backfill | gf_2pow5_mult | False | 0 | - |
| external_journal_k6_backfill | hamming_weight_n4 | True | 3 | factor_count |
| external_journal_k6_backfill | hamming_weight_n5 | True | 3 | factor_count |
| external_journal_k6_backfill | hamming_weight_n6 | True | 2 | factor_count |
| external_journal_k6_backfill | hamming_weight_n7 | True | 2 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | mod_5_4 | True | 3 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | mod_mult_55 | True | 2 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | nc_tof_3 | True | 3 | factor_count |
| external_journal_k6_backfill | nc_tof_4 | True | 2 | depth_guarded_mixed_pair |
| external_journal_k6_backfill | nc_tof_5 | False | 0 | - |
| external_journal_k6_backfill | vbe_adder_3 | False | 1 | factor_count |
| external_journal_k6_frontier | barenco_tof_5 | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_frontier | csla_mux_3 | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_frontier | cuccaro_adder_n5 | False | 0 | - |
| external_journal_k6_frontier | cuccaro_adder_n6 | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_frontier | gf_2pow6_mult | False | 0 | - |
| external_journal_k6_frontier | gf_2pow7_mult | True | 3 | t_preserving_frontier_pair |
| external_journal_k6_frontier | hamming_weight_n8 | True | 2 | t_preserving_frontier_pair |
| external_journal_k6_frontier | unary_iteration_n3 | True | 3 | t_preserving_frontier_pair |
| external_journal_repair_paircap | barenco_tof_4 | False | 1 | factor_count_pair_cap |
| external_night_long | barenco_tof_4 | True | 2 | mixed_pair |
| external_night_long | nc_tof_4 | True | 3 | mixed_pair |
| external_night_long | vbe_adder_3 | True | 3 | factor_count |
| external_standard | barenco_tof_4 | False | 0 | - |
| external_standard | nc_tof_4 | True | 3 | factor_count |
| external_standard | vbe_adder_3 | True | 2 | factor_count |
| internal | barenco_tof_3 | True | 3 | factor_count |
| internal | cuccaro_adder_n3 | True | 3 | factor_count_pair_cap |
| internal | gf_2pow2_mult | True | 3 | mixed_pair |
| internal | gf_2pow3_mult | True | 3 | factor_count_pair_cap |
| internal | hamming_weight_n4 | True | 3 | mixed_pair |
| internal | hamming_weight_n5 | True | 3 | factor_count_pair_cap |
| internal | mod_5_4 | True | 3 | factor_count |
| internal | nc_tof_3 | True | 3 | factor_count_pair_cap |

## Objective Runtime And Coverage

| objective | rows | ok | beam rows | failures | oracle count | median runtime sec | mean runtime sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| factor_count | 68 | 56 | 56 | 12 | 23 | 3.01e+03 | 2.48e+03 |
| factor_count_pair_cap | 68 | 30 | 30 | 38 | 11 | 2.7e+03 | 2.7e+03 |
| mixed_pair | 68 | 29 | 29 | 39 | 7 | 2.7e+03 | 2.7e+03 |
| frontier_pair | 68 | 20 | 20 | 48 | 3 | 3.05e+03 | 3.07e+03 |
| depth_guarded_mixed_pair | 68 | 14 | 14 | 54 | 6 | 125 | 1.21e+03 |
| t_preserving_frontier_pair | 68 | 14 | 14 | 54 | 8 | 3e+03 | 1.96e+03 |
