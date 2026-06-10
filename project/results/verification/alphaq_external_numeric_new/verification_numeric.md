# Numeric Resolution Of Inconclusive feynver Proofs

CSV: `results/verification/alphaq_external_numeric_new/verification_numeric.csv`.

Each pair is checked by exact computational-basis enumeration of the candidate isometry with gadget ancillas prepared in and postselected on |0...0>, compared against the original unitary up to one global constant. `equal-numeric` means the full functional behaviour matches with residual below 1e-08. `equal-up-to-clifford` means the candidate equals the original composed with an explicitly extracted Clifford relabeling (X frame + CNOT parity map + degree-<=2 phase polynomial); such corrections leave T-count unchanged.

- nonclifford-correction: 6

| target | objective | materializer | status | |c| | residual | side | X frame | parity changes | phase degree | defect signature | runtime (s) |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|
| vbe_adder_3 | factor_count | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 58.52 |
| vbe_adder_3 | factor_count_pair_cap | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.36e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 50.14 |
| vbe_adder_3 | frontier_pair | selected-beam-shared-parity-w16 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 50.28 |
| vbe_adder_3 | mixed_pair | selected-beam-shared-parity-w32 | nonclifford-correction | 0.25 | 3.39e-15 | output | 1 | 0 | 3 | 19b1a1d255ca | 62.38 |
| nc_tof_5 | factor_count | selected-beam-shared-parity-w32 | nonclifford-correction | 0.125 | 1.33e-15 | output | 3 | 1 | 3 | 0b7bb41a615e | 86.88 |
| nc_tof_5 | frontier_pair | selected-beam-shared-parity-w32 | nonclifford-correction | 0.125 | 1.33e-15 | output | 3 | 1 | 3 | 0b7bb41a615e | 86.60 |
