# Numeric Resolution Of Inconclusive feynver Proofs

CSV: `results/verification/alphaq_external_numeric/verification_numeric.csv`.

Each pair is checked by exact computational-basis enumeration of the candidate isometry with gadget ancillas prepared in and postselected on |0...0>, compared against the original unitary up to one global constant. `equal-numeric` means the full functional behaviour matches with residual below 1e-08. `equal-up-to-clifford` means the candidate equals the original composed with an explicitly extracted Clifford relabeling (X frame + CNOT parity map + degree-<=2 phase polynomial); such corrections leave T-count unchanged.

- equal-block-exact: 8
- equal-numeric: 2

| target | objective | materializer | status | |c| | residual | side | X frame | parity changes | phase degree | defect signature | runtime (s) |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|
| nc_tof_4 | mixed_pair | selected-beam-shared-parity-w16 | equal-numeric | 0.25 | 2.11e-15 |  |  |  |  |  | 2.25 |
| cuccaro_adder_n4 | factor_count | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.01 |
| cuccaro_adder_n4 | factor_count_pair_cap | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.00 |
| cuccaro_adder_n4 | frontier_pair | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.00 |
| cuccaro_adder_n4 | mixed_pair | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.00 |
| gf_2pow4_mult | factor_count | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.01 |
| gf_2pow4_mult | factor_count_pair_cap | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.01 |
| gf_2pow4_mult | frontier_pair | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.01 |
| gf_2pow4_mult | mixed_pair | selected-beam-shared-parity-w32 | equal-block-exact |  |  |  |  |  |  |  | 0.01 |
| mod_mult_55 | mixed_pair | selected-beam-shared-parity-w4 | equal-numeric | 0.353553 | 4e-15 |  |  |  |  |  | 9.07 |
