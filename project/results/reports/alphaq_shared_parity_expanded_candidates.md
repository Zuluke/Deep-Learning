# Materialized candidate analysis

CSV: `results/csv/alphaq_shared_parity_expanded_candidates.csv`.

## Aggregate behavior

- Candidates analyzed: 8.
- Formal status: equal=7, inconclusive=1.
- T-count improves in 8/8 candidates.
- Primary NC depth ratio improves in 8/8 candidates.
- QASM depth inflates in 7/8 candidates.

The primary structural signal is favorable when the ratio is below one. QASM depth is reported separately because the current shared-parity realization can buy a smaller non-Clifford core by adding a large Clifford routing layer.

## Best candidates by target

| target | best T-count row | best structural row | interpretation |
|---|---:|---:|---|
| barenco_tof_3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=23, primary=0.8113) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=23, primary=0.8113) | improves T-count and structural target, with QASM depth inflation |
| cuccaro_adder_n3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=19, primary=0.7273) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=19, primary=0.7273) | improves T-count and structural target, with QASM depth inflation |
| gf_2pow2_mult | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.8409) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.8409) | improves T-count and structural target, with QASM depth inflation |
| gf_2pow3_mult | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=35, primary=0.8542) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=35, primary=0.8542) | improves T-count and structural target, with QASM depth inflation |
| hamming_weight_n4 | milp_span_mixed_pair_pairo6 [shared_parity_network] greedy-cnot/max-row-weight (T=20, primary=0.973) | milp_span_mixed_pair_pairo6 [shared_parity_network] greedy-cnot/max-row-weight (T=20, primary=0.973) | improves T-count and structural target, with QASM depth inflation |
| hamming_weight_n5 | milp_span_mixed_pair_pairo6 [shared_parity_network] given/min-change (T=20, primary=0.8537) | milp_span_mixed_pair_pairo6 [shared_parity_network] given/min-change (T=20, primary=0.8537) | improves T-count and structural target, with QASM depth inflation |
| mod_5_4 | milp_span_factor_count_wfull [shared_parity_network] greedy-cnot/max-change (T=7, primary=0.2034) | milp_span_factor_count_wfull [shared_parity_network] greedy-cnot/max-change (T=7, primary=0.2034) | improves T-count and structural target |
| nc_tof_3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.875) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.875) | improves T-count and structural target, with QASM depth inflation |

## Candidate table

| target | formal status | kind | synthesis | factor order | target strategy | T-count | T-ratio | primary NC depth ratio | QASM depth ratio | structural cost |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| barenco_tof_3 | equal | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 23 | 0.8214 | 0.8113 | 3.647 | 5.114 |
| cuccaro_adder_n3 | inconclusive | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 19 | 0.4524 | 0.7273 | 3.055 | 3.131 |
| gf_2pow2_mult | equal | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 17 | 0.6071 | 0.8409 | 2.619 | 0.9524 |
| gf_2pow3_mult | equal | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 35 | 0.5556 | 0.8542 | 2.901 | 2.506 |
| hamming_weight_n4 | equal | milp_span_mixed_pair_pairo6 | shared_parity_network | greedy-cnot | max-row-weight | 20 | 0.9524 | 0.973 | 5.353 | 6.353 |
| hamming_weight_n5 | equal | milp_span_mixed_pair_pairo6 | shared_parity_network | given | min-change | 20 | 0.9524 | 0.8537 | 5.211 | 6.02 |
| mod_5_4 | equal | milp_span_factor_count_wfull | shared_parity_network | greedy-cnot | max-change | 7 | 0.25 | 0.2034 | 0.678 | 0.6441 |
| nc_tof_3 | equal | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 17 | 0.8095 | 0.875 | 4.895 | 6.042 |
