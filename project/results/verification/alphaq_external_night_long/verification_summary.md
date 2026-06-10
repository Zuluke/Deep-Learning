# Beam Materializer Formal Verification

CSV: `results/verification/alphaq_external_night_long/verification_summary.csv`.
Total best-beam candidates checked: 8.
- equal: 2
- inconclusive: 6

| target | materializer | verification | primary ratio | QASM ratio | proof |
|---|---|---|---:|---:|---|
| barenco_tof_4 | selected-beam-shared-parity-w32 | equal |  | 3.01 | `results/verification/alphaq_external_night_long/proofs/barenco_tof_4/factor_count/selected-beam-shared-parity-w32.verify.txt` |
| barenco_tof_4 | selected-beam-shared-parity-w16 | equal |  | 2.69 | `results/verification/alphaq_external_night_long/proofs/barenco_tof_4/mixed_pair/selected-beam-shared-parity-w16.verify.txt` |
| nc_tof_4 | selected-beam-shared-parity-w16 | inconclusive |  | 3.92 | `results/verification/alphaq_external_night_long/proofs/nc_tof_4/factor_count/selected-beam-shared-parity-w16.verify.txt` |
| nc_tof_4 | selected-beam-shared-parity-w16 | inconclusive |  | 4 | `results/verification/alphaq_external_night_long/proofs/nc_tof_4/factor_count_pair_cap/selected-beam-shared-parity-w16.verify.txt` |
| nc_tof_4 | selected-beam-shared-parity-w16 | inconclusive |  | 3.98 | `results/verification/alphaq_external_night_long/proofs/nc_tof_4/mixed_pair/selected-beam-shared-parity-w16.verify.txt` |
| vbe_adder_3 | selected-beam-shared-parity-w32 | inconclusive |  | 3.05 | `results/verification/alphaq_external_night_long/proofs/vbe_adder_3/factor_count/selected-beam-shared-parity-w32.verify.txt` |
| vbe_adder_3 | selected-beam-shared-parity-w32 | inconclusive |  | 3.21 | `results/verification/alphaq_external_night_long/proofs/vbe_adder_3/factor_count_pair_cap/selected-beam-shared-parity-w32.verify.txt` |
| vbe_adder_3 | selected-beam-shared-parity-w32 | inconclusive |  | 3.28 | `results/verification/alphaq_external_night_long/proofs/vbe_adder_3/mixed_pair/selected-beam-shared-parity-w32.verify.txt` |
