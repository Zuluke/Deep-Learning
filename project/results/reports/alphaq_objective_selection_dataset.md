# AlphaQ Objective-Selection Dataset

CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_objective_selection_dataset.csv`.

This dataset reframes the current evidence as supervised objective selection: each target/run contains one row per AlphaQ objective and an oracle label derived from the best materialized beam candidate.

## Bottom Line

Decision: `prototype-ready`.

Train-ready target groups: 33/42.
External train-ready target groups: 25.
Oracle objective distribution: factor_count=14, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=7.

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
| factor_count | 42 | 34 | 34 | 8 | 15 | 3.04e+03 | 2.99e+03 |
| factor_count_pair_cap | 42 | 30 | 30 | 12 | 11 | 2.7e+03 | 2.7e+03 |
| mixed_pair | 42 | 29 | 29 | 13 | 7 | 2.7e+03 | 2.7e+03 |
| frontier_pair | 42 | 15 | 15 | 27 | 3 | 3.06e+03 | 2.97e+03 |
| depth_guarded_mixed_pair | 42 | 0 | 0 | 42 | 0 |  |  |
| t_preserving_frontier_pair | 42 | 0 | 0 | 42 | 0 |  |  |
