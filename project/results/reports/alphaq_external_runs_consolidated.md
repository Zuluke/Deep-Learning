# AlphaQ External Runs Consolidated

CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_external_runs_consolidated.csv`.
Runs: `standard,night_long,article_core,article_extended,journal_repair,journal_repair_paircap,journal_full_1_rerun,journal_full_1_rerun2,journal_full_2_rerun,journal_full_2_rerun2,journal_full_mod_mult_55,journal_full_cuccaro_adder_n4,journal_full_gf_2pow4_mult,journal_full_hamming_weight_n6,journal_full_hamming_weight_n7,journal_full_gf_2pow5_mult,journal_full_nc_tof_5,journal_full_cuccaro_adder_n5,article_repair2_barenco,article_repair2_vbe,journal_full_nc_tof_5_long,journal_full_gf_2pow5_mult_long`.

Complete targets: 12/15.

| target | objective | best run | status | T-count | best QASM | completed runs | failed runs |
|---|---|---|---|---:|---:|---|---|
| barenco_tof_4 | factor_count | article_repair2_barenco | ok | 64.0 | 298.0 | night_long,article_repair2_barenco | standard,article_core |
| barenco_tof_4 | factor_count_pair_cap | journal_repair_paircap | ok | 61.0 | 286.0 | journal_repair_paircap | standard,night_long,article_core,article_repair2_barenco |
| barenco_tof_4 | frontier_pair | article_core | ok | 64.0 | 298.0 | article_core | article_repair2_barenco |
| barenco_tof_4 | mixed_pair | article_repair2_barenco | ok | 42.0 | 266.0 | night_long,article_repair2_barenco | standard,article_core |
| cuccaro_adder_n4 | factor_count | journal_full_cuccaro_adder_n4 | ok | 23.0 | 241.0 | article_extended,journal_full_cuccaro_adder_n4 |  |
| cuccaro_adder_n4 | factor_count_pair_cap | journal_full_cuccaro_adder_n4 | ok | 23.0 | 251.0 | article_extended,journal_full_cuccaro_adder_n4 |  |
| cuccaro_adder_n4 | frontier_pair | article_extended | ok | 43.0 | 257.0 | article_extended |  |
| cuccaro_adder_n4 | mixed_pair | article_extended | ok | 32.0 | 251.0 | article_extended,journal_full_cuccaro_adder_n4 |  |
| cuccaro_adder_n5 | factor_count | journal_full_cuccaro_adder_n5 | ok | 590.0 | 806.0 | journal_full_cuccaro_adder_n5 | article_extended |
| cuccaro_adder_n5 | factor_count_pair_cap | article_extended | failed |  |  |  | article_extended,journal_full_cuccaro_adder_n5 |
| cuccaro_adder_n5 | frontier_pair | article_extended | failed |  |  |  | article_extended |
| cuccaro_adder_n5 | mixed_pair | article_extended | failed |  |  |  | article_extended,journal_full_cuccaro_adder_n5 |
| gf_2pow2_mult | factor_count | article_core | ok | 17.0 | 97.0 | article_core |  |
| gf_2pow2_mult | factor_count_pair_cap | article_core | ok | 17.0 | 99.0 | article_core |  |
| gf_2pow2_mult | frontier_pair | article_core | ok | 17.0 | 98.0 | article_core |  |
| gf_2pow2_mult | mixed_pair | article_core | ok | 17.0 | 98.0 | article_core |  |
| gf_2pow4_mult | factor_count | journal_full_gf_2pow4_mult | ok | 57.0 | 461.0 | article_extended,journal_full_gf_2pow4_mult |  |
| gf_2pow4_mult | factor_count_pair_cap | journal_full_gf_2pow4_mult | ok | 55.0 | 462.0 | article_extended,journal_full_gf_2pow4_mult |  |
| gf_2pow4_mult | frontier_pair | article_extended | ok | 64.0 | 465.0 | article_extended |  |
| gf_2pow4_mult | mixed_pair | article_extended | ok | 63.0 | 452.0 | article_extended,journal_full_gf_2pow4_mult |  |
| gf_2pow5_mult | factor_count | journal_full_gf_2pow5_mult_long | ok | 163.0 | 718.0 | journal_full_gf_2pow5_mult_long | article_extended,journal_full_gf_2pow5_mult |
| gf_2pow5_mult | factor_count_pair_cap | article_extended | failed |  |  |  | article_extended,journal_full_gf_2pow5_mult,journal_full_gf_2pow5_mult_long |
| gf_2pow5_mult | frontier_pair | journal_full_gf_2pow5_mult_long | ok | 163.0 | 718.0 | journal_full_gf_2pow5_mult_long | article_extended |
| gf_2pow5_mult | mixed_pair | article_extended | failed |  |  |  | article_extended,journal_full_gf_2pow5_mult,journal_full_gf_2pow5_mult_long |
| hamming_weight_n4 | factor_count | article_core | ok | 19.0 | 164.0 | article_core |  |
| hamming_weight_n4 | factor_count_pair_cap | article_core | ok | 19.0 | 167.0 | article_core |  |
| hamming_weight_n4 | frontier_pair | article_core | ok | 19.0 | 171.0 | article_core |  |
| hamming_weight_n4 | mixed_pair | article_core | ok | 20.0 | 170.0 | article_core |  |
| hamming_weight_n5 | factor_count | article_core | ok | 19.0 | 179.0 | article_core |  |
| hamming_weight_n5 | factor_count_pair_cap | article_core | ok | 20.0 | 179.0 | article_core |  |
| hamming_weight_n5 | frontier_pair | article_core | ok | 19.0 | 179.0 | article_core |  |
| hamming_weight_n5 | mixed_pair | article_core | ok | 20.0 | 180.0 | article_core |  |
| hamming_weight_n6 | factor_count | article_extended | ok | 25.0 | 172.0 | article_extended,journal_full_hamming_weight_n6 |  |
| hamming_weight_n6 | factor_count_pair_cap | article_extended | ok | 25.0 | 168.0 | article_extended,journal_full_hamming_weight_n6 |  |
| hamming_weight_n6 | frontier_pair | article_extended | ok | 25.0 | 172.0 | article_extended |  |
| hamming_weight_n6 | mixed_pair | journal_full_hamming_weight_n6 | ok | 26.0 | 172.0 | article_extended,journal_full_hamming_weight_n6 |  |
| hamming_weight_n7 | factor_count | journal_full_hamming_weight_n7 | ok | 26.0 | 202.0 | article_extended,journal_full_hamming_weight_n7 |  |
| hamming_weight_n7 | factor_count_pair_cap | journal_full_hamming_weight_n7 | ok | 28.0 | 188.0 | article_extended,journal_full_hamming_weight_n7 |  |
| hamming_weight_n7 | frontier_pair | article_extended | ok | 28.0 | 193.0 | article_extended |  |
| hamming_weight_n7 | mixed_pair | article_extended | ok | 36.0 | 210.0 | article_extended,journal_full_hamming_weight_n7 |  |
| mod_5_4 | factor_count | article_core | ok | 7.0 | 37.0 | article_core |  |
| mod_5_4 | factor_count_pair_cap | article_core | ok | 7.0 | 37.0 | article_core |  |
| mod_5_4 | frontier_pair | article_core | ok | 7.0 | 37.0 | article_core |  |
| mod_5_4 | mixed_pair | article_core | ok | 7.0 | 37.0 | article_core |  |
| mod_mult_55 | factor_count | journal_full_mod_mult_55 | ok | 18.0 | 268.0 | article_extended,journal_full_mod_mult_55 |  |
| mod_mult_55 | factor_count_pair_cap | article_extended | ok | 28.0 | 264.0 | article_extended,journal_full_mod_mult_55 |  |
| mod_mult_55 | frontier_pair | article_extended | ok | 29.0 | 264.0 | article_extended |  |
| mod_mult_55 | mixed_pair | article_extended | ok | 18.0 | 252.0 | article_extended,journal_full_mod_mult_55 |  |
| nc_tof_4 | factor_count | night_long | ok | 24.0 | 243.0 | standard,night_long,article_core |  |
| nc_tof_4 | factor_count_pair_cap | article_core | ok | 32.0 | 248.0 | standard,night_long,article_core |  |
| nc_tof_4 | frontier_pair | article_core | ok | 32.0 | 245.0 | article_core |  |
| nc_tof_4 | mixed_pair | night_long | ok | 24.0 | 240.0 | standard,night_long,article_core |  |
| nc_tof_5 | factor_count | journal_full_nc_tof_5_long | ok | 103.0 | 581.0 | journal_full_nc_tof_5_long | article_extended,journal_full_nc_tof_5 |
| nc_tof_5 | factor_count_pair_cap | article_extended | failed |  |  |  | article_extended,journal_full_nc_tof_5,journal_full_nc_tof_5_long |
| nc_tof_5 | frontier_pair | journal_full_nc_tof_5_long | ok | 103.0 | 581.0 | journal_full_nc_tof_5_long | article_extended |
| nc_tof_5 | mixed_pair | article_extended | failed |  |  |  | article_extended,journal_full_nc_tof_5,journal_full_nc_tof_5_long |
| vbe_adder_3 | factor_count | article_core | ok | 65.0 | 296.0 | standard,night_long,article_core,article_repair2_vbe |  |
| vbe_adder_3 | factor_count_pair_cap | article_repair2_vbe | ok | 51.0 | 278.0 | standard,night_long,article_core,article_repair2_vbe |  |
| vbe_adder_3 | frontier_pair | article_repair2_vbe | ok | 51.0 | 285.0 | article_core,article_repair2_vbe |  |
| vbe_adder_3 | mixed_pair | article_repair2_vbe | ok | 77.0 | 317.0 | night_long,article_repair2_vbe | standard,article_core |
