# Formal Verification Summary

- Scope: `entrega1`.
- Total verification tasks: 34.
- Equal: 21.
- inconclusive: 5
- timeout: 8

## Non-equal or inconclusive cases

- `cuccaro_adder_n3` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.011s); proof: `results/verification/entrega1/proofs/cuccaro_adder_n3/public_resynth_gadgets.verify.txt`
- `hamming_15_low` / `compile_no_quizx`: inconclusive - Inconclusive (took 12.628s); proof: `results/verification/entrega1/proofs/hamming_15_low/compile_no_quizx.verify.txt`
- `hamming_15_low` / `public_resynth_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/public_resynth_gadgets.verify.txt`
- `hamming_15_low` / `public_resynth_no_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/public_resynth_no_gadgets.verify.txt`
- `hamming_15_low` / `pyzx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/pyzx.verify.txt`
- `qcla_mod_7` / `compile_no_quizx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/compile_no_quizx.verify.txt`
- `qcla_mod_7` / `compile_quizx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/compile_quizx.verify.txt`
- `qcla_mod_7` / `public_resynth_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/public_resynth_gadgets.verify.txt`
- `qcla_mod_7` / `public_resynth_no_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/public_resynth_no_gadgets.verify.txt`
- `qcla_mod_7` / `pyzx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/pyzx.verify.txt`
- `qft_4` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.570s); proof: `results/verification/entrega1/proofs/qft_4/public_resynth_gadgets.verify.txt`
- `vbe_adder_3` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.059s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_gadgets.verify.txt`
- `vbe_adder_3` / `public_resynth_no_gadgets`: inconclusive - Inconclusive (took 0.111s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_no_gadgets.verify.txt`

## Notes

- Verification is run through `circuit-to-tensor verify`, which calls `feynver -postselect-ancillas -ignore-global-phase`.
- QASM files are first normalized to the local Clifford+T comparison basis so PyZX `rz(k*pi/4)` output can be checked by the Feynman `.qc` backend.
- A status of `equal` means Feynman returned an equality proof for the normalized original/candidate pair.
