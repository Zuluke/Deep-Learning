# AlphaQuantum Journal Evidence Audit

Gate CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_journal_evidence_gates.csv`.
Next battery CSV: `/Users/caio/Deep-Learning/project/results/csv/alphaq_journal_next_battery.csv`.

## Decision

Decision: `not-yet-journal-ready`.

Note: journal-readiness gates are defined on the core-4 portfolio (`factor_count`, `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`); K=6 coverage is reported in the selection-evaluation layer.

passed gates=4/6; blocking gates=external_nonbaseline_effect, formal_verification_coverage.

The current evidence is strong enough to justify a prototype integration experiment, but it is not yet enough for a journal-level robustness claim. The blocking issues are external coverage, repeated non-baseline external wins, and complete formal verification over the promoted/expanded candidates.

## Evidence Gates

| gate | status | score | evidence | required next |
|---|---|---:|---|---|
| selector_loto | pass | 0.87 | best Split-Select `split_select_linear_alphaq_qasm` has oracle matches 47/54; baseline has 21/54; T-count wins over baseline 21/54. | Keep the trained selector as prototype policy; expand held-out targets before journal claims. |
| dataset_scale_and_label_diversity | pass | 1 | train-ready groups=54/68; external train-ready groups=46; oracle labels=depth_guarded_mixed_pair=6, factor_count=21, factor_count_pair_cap=10, frontier_pair=2, mixed_pair=7, t_preserving_frontier_pair=8. | Grow to at least 30 train-ready target/run groups with at least 10 external groups and all three objective labels represented. |
| external_generalization_coverage | pass | 1 | complete external targets=12 (barenco_tof_4, cuccaro_adder_n4, gf_2pow2_mult, gf_2pow4_mult, hamming_weight_n4, hamming_weight_n5, hamming_weight_n6, hamming_weight_n7, mod_5_4, mod_mult_55, nc_tof_4, vbe_adder_3); partial external targets=13 (barenco_tof_3, barenco_tof_5, csla_mux_3, cuccaro_adder_n3, cuccaro_adder_n5, cuccaro_adder_n6, gf_2pow3_mult, gf_2pow5_mult, gf_2pow7_mult, hamming_weight_n8, nc_tof_3, nc_tof_5, unary_iteration_n3); external families in dataset=6. | Run a broader external battery: at least 10 complete external targets spanning at least 3 families. |
| external_nonbaseline_effect | partial | 1 | non-baseline external improvements=14 (barenco_tof_3:t_preserving_frontier_pair, barenco_tof_4:mixed_pair, barenco_tof_5:t_preserving_frontier_pair, csla_mux_3:t_preserving_frontier_pair, cuccaro_adder_n3:t_preserving_frontier_pair, cuccaro_adder_n6:t_preserving_frontier_pair, gf_2pow4_mult:factor_count_pair_cap, gf_2pow7_mult:t_preserving_frontier_pair, hamming_weight_n6:factor_count_pair_cap, hamming_weight_n8:t_preserving_frontier_pair, mod_mult_55:depth_guarded_mixed_pair, nc_tof_4:depth_guarded_mixed_pair, unary_iteration_n3:t_preserving_frontier_pair, vbe_adder_3:factor_count_pair_cap); non-baseline regressions=1 (mod_5_4:depth_guarded_mixed_pair). | Find repeated external cases where Split-Select chooses a non-factor-count objective and improves T-count/depth. |
| formal_verification_coverage | partial | 0.833 | formal verification rows=102; proven=85 (equal=66, equal-numeric=2, equal-up-to-clifford=0); characterized assembly defects=0; inconclusive=17; unexplained failures=0. | Repair the assembly defect on targets with non-Clifford corrections, re-materialize, and re-verify; then all promoted candidates should be proven. |
| depth_control | pass | 1 | best Split-Select median QASM ratio=1; QASM non-worse=46/54. | Keep depth as a hard audit metric; avoid claiming T-count wins alone. |
| overall_journal_readiness | not-yet-journal-ready | 0.667 | passed gates=4/6; blocking gates=external_nonbaseline_effect, formal_verification_coverage. | Do not frame as journal-ready until blocking gates pass; use current results as prototype evidence. |
| next_decisive_battery | planned | 0.5 | recommended full-action expansion targets=0; tensor-v3 screening targets=3; restricted-action pilot targets=1. | Run tensor-v3 screening before repeating failed full-action targets; then implement the restricted-action pilot if the signal survives. |

## Recommended Next Battery

| target | family | tensor size | T original | stage | status | action |
|---|---|---:|---:|---|---|---|
| gf_2pow5_mult | arithmetic | 15 | 175 | tensor-v3-screen | needs-tensor-v3-screen | Run tensor-v3/profile screening before objective-grid expansion. |
| nc_tof_5 | arithmetic | 15 | 49 | tensor-v3-screen | needs-tensor-v3-screen | Run tensor-v3/profile screening before objective-grid expansion. |
| cuccaro_adder_n5 | applications | 16 | 70 | tensor-v3-screen | needs-tensor-v3-screen | Run tensor-v3/profile screening before objective-grid expansion. |
| hwb_6 | arithmetic | 27 | 105 | restricted-action-pilot | ready-restricted-action | Run guarded selector with restricted/beam action policy, not full action enumeration. |

## Practical Interpretation

The immediate journal path is not to change the reward again. It is to run the next external battery, keep failures explicit, and then rerun this audit. If the Split-Select policy keeps the Barenco-like gains while avoiding VBE-like regressions across a larger set, then the result starts looking like a journal claim rather than a promising engineering observation.
