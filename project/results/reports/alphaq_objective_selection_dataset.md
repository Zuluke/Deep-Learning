# AlphaQ Objective-Selection Dataset

CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_objective_selection_dataset.csv`.

This dataset reframes the current evidence as supervised objective selection: each target/run contains one row per AlphaQ objective and an oracle label derived from the best materialized beam candidate.

## Bottom Line

Decision: `prototype-ready`.

Train-ready target groups: 18/22.
External train-ready target groups: 10.
Oracle objective distribution: factor_count=8, factor_count_pair_cap=7, mixed_pair=3.

We can proceed with a prototype selector and leave-one-target validation. The dataset is still too small for a high-capacity deep model or a broad journal-level generalization claim.

## Target Labels

| split | target | train ready | materialized objectives | oracle objective |
|---|---|---:|---:|---|
| external_journal_full_cuccaro_adder_n4 | cuccaro_adder_n4 | True | 3 | factor_count |
| external_journal_full_gf_2pow4_mult | gf_2pow4_mult | True | 3 | factor_count_pair_cap |
| external_journal_full_gf_2pow5_mult | gf_2pow5_mult | False | 0 | - |
| external_journal_full_hamming_weight_n6 | hamming_weight_n6 | True | 3 | factor_count_pair_cap |
| external_journal_full_hamming_weight_n7 | hamming_weight_n7 | True | 3 | factor_count |
| external_journal_full_mod_mult_55 | mod_mult_55 | True | 3 | mixed_pair |
| external_journal_full_nc_tof_5 | nc_tof_5 | False | 0 | - |
| external_journal_repair_paircap | barenco_tof_4 | False | 1 | factor_count_pair_cap |
| external_night_long | barenco_tof_4 | True | 2 | mixed_pair |
| external_night_long | nc_tof_4 | True | 3 | factor_count |
| external_night_long | vbe_adder_3 | True | 3 | factor_count |
| external_standard | barenco_tof_4 | False | 0 | - |
| external_standard | nc_tof_4 | True | 3 | factor_count |
| external_standard | vbe_adder_3 | True | 2 | factor_count |
| internal | barenco_tof_3 | True | 3 | factor_count |
| internal | cuccaro_adder_n3 | True | 3 | factor_count_pair_cap |
| internal | gf_2pow2_mult | True | 3 | factor_count_pair_cap |
| internal | gf_2pow3_mult | True | 3 | factor_count_pair_cap |
| internal | hamming_weight_n4 | True | 3 | mixed_pair |
| internal | hamming_weight_n5 | True | 3 | factor_count_pair_cap |
| internal | mod_5_4 | True | 3 | factor_count |
| internal | nc_tof_3 | True | 3 | factor_count_pair_cap |
