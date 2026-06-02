# Formal Verification Summary

- Scope: `entrega1`.
- Total verification tasks: 69.
- Equal: 52.
- inconclusive: 9
- timeout: 8

## Non-equal or inconclusive cases

- `cuccaro_adder_n3` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.011s); proof: `results/verification/entrega1/proofs/cuccaro_adder_n3/public_resynth_gadgets.verify.txt`
- `cuccaro_adder_n3` / `public_resynth_reranker`: inconclusive - Inconclusive (took 0.010s); proof: `results/verification/entrega1/proofs/cuccaro_adder_n3/public_resynth_reranker.verify.txt`
- `cuccaro_adder_n3` / `public_resynth_structural`: inconclusive - Inconclusive (took 0.011s); proof: `results/verification/entrega1/proofs/cuccaro_adder_n3/public_resynth_structural.verify.txt`
- `hamming_15_low` / `compile_no_quizx`: inconclusive - Inconclusive (took 12.498s); proof: `results/verification/entrega1/proofs/hamming_15_low/compile_no_quizx.verify.txt`
- `hamming_15_low` / `public_resynth_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/public_resynth_gadgets.verify.txt`
- `hamming_15_low` / `public_resynth_no_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/public_resynth_no_gadgets.verify.txt`
- `hamming_15_low` / `pyzx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/hamming_15_low/pyzx.verify.txt`
- `qcla_mod_7` / `compile_no_quizx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/compile_no_quizx.verify.txt`
- `qcla_mod_7` / `compile_quizx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/compile_quizx.verify.txt`
- `qcla_mod_7` / `public_resynth_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/public_resynth_gadgets.verify.txt`
- `qcla_mod_7` / `public_resynth_no_gadgets`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/public_resynth_no_gadgets.verify.txt`
- `qcla_mod_7` / `pyzx`: timeout - Timed out after 30s.; proof: `results/verification/entrega1/proofs/qcla_mod_7/pyzx.verify.txt`
- `qft_4` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.573s); proof: `results/verification/entrega1/proofs/qft_4/public_resynth_gadgets.verify.txt`
- `vbe_adder_3` / `public_resynth_gadgets`: inconclusive - Inconclusive (took 0.061s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_gadgets.verify.txt`
- `vbe_adder_3` / `public_resynth_no_gadgets`: inconclusive - Inconclusive (took 0.119s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_no_gadgets.verify.txt`
- `vbe_adder_3` / `public_resynth_reranker`: inconclusive - Inconclusive (took 0.063s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_reranker.verify.txt`
- `vbe_adder_3` / `public_resynth_structural`: inconclusive - Inconclusive (took 0.061s); proof: `results/verification/entrega1/proofs/vbe_adder_3/public_resynth_structural.verify.txt`

## Notes

- Verification is run through `circuit-to-tensor verify`, which calls `feynver -postselect-ancillas -ignore-global-phase`.
- QASM files are first normalized to the local Clifford+T comparison basis so PyZX `rz(k*pi/4)` output can be checked by the Feynman `.qc` backend.
- A status of `equal` means Feynman returned an equality proof for the normalized original/candidate pair.
