# Numeric Resolution Of Inconclusive feynver Proofs

CSV: `results/verification/alphaq_external_numeric/verification_numeric.csv`.

Each pair is checked by exact computational-basis enumeration of the candidate isometry with gadget ancillas prepared in and postselected on |0...0>, compared against the original unitary up to one global constant. `equal-numeric` means the full functional behaviour matches with residual below 1e-08. `equal-up-to-clifford` means the candidate equals the original composed with an explicitly extracted Clifford relabeling (X frame + CNOT parity map + degree-<=2 phase polynomial); such corrections leave T-count unchanged.

- equal-numeric: 1
- nonclifford-correction: 13
- not-equal: 8

| target | objective | materializer | status | |c| | residual | side | X frame | parity changes | phase degree | defect signature | runtime (s) |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|
| nc_tof_4 | factor_count | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 2e-15 | input | 1 | 1 | 3 | fc446d84498e | 2.81 |
| nc_tof_4 | factor_count_pair_cap | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 1.89e-15 | input | 1 | 1 | 3 | fc446d84498e | 2.41 |
| nc_tof_4 | frontier_pair | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 1.97e-15 | input | 1 | 1 | 3 | fc446d84498e | 2.59 |
| nc_tof_4 | mixed_pair | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 1.97e-15 | input | 1 | 1 | 3 | fc446d84498e | 2.78 |
| vbe_adder_3 | factor_count | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 77.27 |
| vbe_adder_3 | factor_count_pair_cap | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.44e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 110.68 |
| vbe_adder_3 | frontier_pair | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.33e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 85.29 |
| cuccaro_adder_n4 | factor_count | selected-beam-shared-parity-w32 | not-equal | 4.58292e-18 | 0.125 |  |  |  |  |  | 11.38 |
| cuccaro_adder_n4 | factor_count_pair_cap | selected-beam-shared-parity-w32 | not-equal | 4.58292e-18 | 0.125 |  |  |  |  |  | 11.24 |
| cuccaro_adder_n4 | frontier_pair | selected-beam-shared-parity-w32 | not-equal | 4.58292e-18 | 0.125 |  |  |  |  |  | 11.13 |
| cuccaro_adder_n4 | mixed_pair | selected-beam-shared-parity-w32 | not-equal | 4.58292e-18 | 0.125 |  |  |  |  |  | 11.32 |
| gf_2pow4_mult | factor_count | selected-beam-shared-parity-w32 | not-equal | 0.0110485 | 0.258 |  |  |  |  |  | 292.54 |
| gf_2pow4_mult | factor_count_pair_cap | selected-beam-shared-parity-w32 | not-equal | 0.0110485 | 0.258 |  |  |  |  |  | 243.63 |
| gf_2pow4_mult | frontier_pair | selected-beam-shared-parity-w32 | not-equal | 0.0110485 | 0.258 |  |  |  |  |  | 416.14 |
| gf_2pow4_mult | mixed_pair | selected-beam-shared-parity-w32 | not-equal | 0.0110485 | 0.258 |  |  |  |  |  | 1355.84 |
| mod_mult_55 | mixed_pair | selected-beam-shared-parity-w4 | equal-numeric | 0.353553 | 4e-15 |  |  |  |  |  | 11.30 |
| nc_tof_4 | factor_count | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 1.92e-15 | input | 1 | 1 | 3 | fc446d84498e | 1.95 |
| nc_tof_4 | factor_count_pair_cap | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 1.89e-15 | input | 1 | 1 | 3 | fc446d84498e | 2.27 |
| nc_tof_4 | mixed_pair | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 1.97e-15 | input | 1 | 1 | 3 | fc446d84498e | 1.73 |
| vbe_adder_3 | factor_count | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 58.55 |
| vbe_adder_3 | factor_count_pair_cap | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 262.57 |
| vbe_adder_3 | mixed_pair | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.36e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 568.78 |
