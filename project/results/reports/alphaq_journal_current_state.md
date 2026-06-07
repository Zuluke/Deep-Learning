# AlphaQ Journal Current State

Figure: `/Users/caio/Deep-Learning/project/results/figures/alphaq_journal_current_state.png`.

Decision: `not-yet-journal-ready`.

## Main Readout

Split-Select currently matches the post-hoc oracle in 13/18 evaluated groups, versus 8/18 for the factor-count-only baseline.

The external battery has 8 complete targets, 1 partial target, and 2 failed targets.

## Best External Objective Relative to Factor-count

| target | best objective | T-count ratio | QASM-depth ratio |
|---|---|---:|---:|
| barenco_tof_4 | Mixed-pair | 0.656 | 0.893 |
| gf_2pow4_mult | Pair-cap | 0.965 | 1.002 |
| mod_mult_55 | Mixed-pair | 1.000 | 0.940 |
| hamming_weight_n6 | Pair-cap | 1.000 | 0.971 |
| cuccaro_adder_n4 | Factor-count | 1.000 | 1.000 |
| hamming_weight_n7 | Factor-count | 1.000 | 1.000 |
| nc_tof_4 | Factor-count | 1.000 | 1.000 |
| vbe_adder_3 | Factor-count | 1.000 | 1.000 |

## Gate Status

| gate | status | score |
|---|---|---:|
| selector_loto | pass | 0.556 |
| dataset_scale_and_label_diversity | partial | 0.600 |
| external_generalization_coverage | partial | 0.800 |
| external_nonbaseline_effect | pass | 1.000 |
| formal_verification_coverage | fail | 0.000 |
| depth_control | pass | 1.000 |
| overall_journal_readiness | not-yet-journal-ready | 0.500 |
| next_decisive_battery | planned | 0.500 |