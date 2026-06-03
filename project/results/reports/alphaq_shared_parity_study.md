# Materialized candidate analysis

CSV: `results/csv/alphaq_shared_parity_study.csv`.

## Best candidates by target

| target | best T-count row | best structural row | interpretation |
|---|---:|---:|---|
| gf_2pow2_mult | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.8409) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.8409) | improves T-count and structural target |
| hamming_weight_n4 | milp_span_mixed_pair_pairo6 [shared_parity_network] greedy-cnot/max-row-weight (T=20, primary=0.973) | milp_span_mixed_pair_pairo6 [shared_parity_network] greedy-cnot/max-row-weight (T=20, primary=0.973) | improves T-count and structural target |
| hamming_weight_n5 | milp_span_mixed_pair_pairo6 [shared_parity_network] given/min-change (T=20, primary=0.8537) | milp_span_mixed_pair_pairo6 [shared_parity_network] given/min-change (T=20, primary=0.8537) | improves T-count and structural target |
| mod_5_4 | milp_span_factor_count_wfull [shared_parity_network] greedy-cnot/max-change (T=7, primary=0.2034) | milp_span_factor_count_wfull [shared_parity_network] greedy-cnot/max-change (T=7, primary=0.2034) | improves T-count and structural target |

## Candidate table

| target | kind | synthesis | factor order | target strategy | T-count | T-ratio | primary NC depth ratio | QASM depth ratio | structural cost |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| gf_2pow2_mult | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 17 | 0.6071 | 0.8409 | 2.619 | 0.9524 |
| hamming_weight_n4 | milp_span_mixed_pair_pairo6 | shared_parity_network | greedy-cnot | max-row-weight | 20 | 0.9524 | 0.973 | 5.353 | 6.353 |
| hamming_weight_n5 | milp_span_mixed_pair_pairo6 | shared_parity_network | given | min-change | 20 | 0.9524 | 0.8537 | 5.211 | 6.02 |
| mod_5_4 | milp_span_factor_count_wfull | shared_parity_network | greedy-cnot | max-change | 7 | 0.25 | 0.2034 | 0.678 | 0.6441 |
