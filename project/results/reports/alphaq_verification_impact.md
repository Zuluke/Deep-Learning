# Verification Status Of Portfolio Selection Wins

Policy: `guarded_top2` (scope `targets`). CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_verification_impact.csv`.

T-count wins vs baseline: 7; wins whose selected candidate is fully proven: 3.

| target | split | selected objective | T ratio | outcome | selected proof | baseline proof |
|---|---|---|---:|---|---|---|
| barenco_tof_3 | internal | factor_count | 1.0 | tie | unverified | unverified |
| cuccaro_adder_n3 | internal | factor_count_pair_cap | 1.0 | tie | unverified | unverified |
| gf_2pow2_mult | external_article_core | factor_count | 1.0 | tie | equal | equal |
| hamming_weight_n4 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| hamming_weight_n5 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| hamming_weight_n6 | external_article_extended | frontier_pair | 1.0 | tie | equal | equal |
| mod_5_4 | external_article_core | factor_count | 1.0 | tie | equal | equal |
| nc_tof_3 | internal | factor_count_pair_cap | 1.0 | tie | unverified | unverified |
| vbe_adder_3 | external_night_long | factor_count | 1.0 | tie | nonclifford-correction | nonclifford-correction |
| barenco_tof_4 | external_night_long | mixed_pair | 0.65625 | win | equal | equal |
| cuccaro_adder_n4 | external_article_extended | factor_count_pair_cap | 0.75 | win | not-equal | not-equal |
| gf_2pow3_mult | internal | factor_count_pair_cap | 0.9428571428571428 | win | unverified | unverified |
| gf_2pow4_mult | external_article_extended | factor_count_pair_cap | 0.9655172413793104 | win | not-equal | not-equal |
| hamming_weight_n7 | external_article_extended | frontier_pair | 0.8235294117647058 | win | equal | equal |
| mod_mult_55 | external_article_extended | mixed_pair | 0.9 | win | equal-numeric | equal |
| nc_tof_4 | external_article_core | frontier_pair | 0.8648648648648649 | win | nonclifford-correction | nonclifford-correction |
