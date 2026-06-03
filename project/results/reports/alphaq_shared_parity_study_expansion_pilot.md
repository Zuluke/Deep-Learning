# Materialized candidate analysis

CSV: `results/csv/alphaq_shared_parity_study_expansion_pilot.csv`.

## Best candidates by target

| target | best T-count row | best structural row | interpretation |
|---|---:|---:|---|
| barenco_tof_3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=23, primary=0.8113) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=23, primary=0.8113) | improves T-count and structural target |
| cuccaro_adder_n3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=19, primary=0.7273) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=19, primary=0.7273) | improves T-count and structural target |
| gf_2pow3_mult | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=35, primary=0.8542) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=35, primary=0.8542) | improves T-count and structural target |
| nc_tof_3 | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.875) | milp_span_mixed_pair_pairo6_wfull [shared_parity_network] greedy-cnot/max-change (T=17, primary=0.875) | improves T-count and structural target |

## Candidate table

| target | kind | synthesis | factor order | target strategy | T-count | T-ratio | primary NC depth ratio | QASM depth ratio | structural cost |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| barenco_tof_3 | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 23 | 0.8214 | 0.8113 | 3.647 | 5.114 |
| cuccaro_adder_n3 | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 19 | 0.4524 | 0.7273 | 3.055 | 3.131 |
| gf_2pow3_mult | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 35 | 0.5556 | 0.8542 | 2.901 | 2.506 |
| nc_tof_3 | milp_span_mixed_pair_pairo6_wfull | shared_parity_network | greedy-cnot | max-change | 17 | 0.8095 | 0.875 | 4.895 | 6.042 |
