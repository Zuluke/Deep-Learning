# AlphaQ Article Experiment Protocol

Protocol CSV: `results/csv/alphaq_article_experiment_protocol.csv`.
Apuana commands: `results/reports/alphaq_article_apuana_commands.md`.

This protocol freezes the article-facing comparison before the heavy jobs are submitted. The method sees only AlphaQ objectives; the paper-style ZX detector is used after materialization as an external audit.

## Fixed Objectives

- `factor_count`: original AlphaQuantum tensor objective.
- `factor_count_pair_cap`: original objective with pair-overlap cap.
- `mixed_pair`: previous split-aware AlphaQ-only objective.
- `frontier_pair`: article-inspired AlphaQ-only objective aligned with paper-style frontier auditing.

## Metrics

- Primary external metric: `paper_primary_nc_depth_ratio`.
- Secondary metrics: `tcount`, `qasm_depth_ratio`, `paper_zx_best_nonclifford_depth`.
- No ZX, PyZX, or feynver metric is used inside the AlphaQ objective.

## Target Coverage

Targets listed: 15.
Ready for direct run/control refresh: 7.
Require tensor/profile pre-screen first: 8.
Need review/readiness update: 0.

| batch | target | status | family | tensor size |
|---|---|---|---|---:|
| article_core | nc_tof_4 | ready-new-run | arithmetic | 11 |
| article_core | barenco_tof_4 | ready-new-run | arithmetic | 14 |
| article_core | vbe_adder_3 | ready-new-run | arithmetic | 14 |
| article_core | gf_2pow2_mult | ready-control-refresh | arithmetic | 6 |
| article_core | mod_5_4 | ready-control-refresh | arithmetic | 5 |
| article_core | hamming_weight_n4 | ready-control-refresh | applications | 9 |
| article_core | hamming_weight_n5 | ready-control-refresh | applications | 10 |
| article_extended | mod_mult_55 | pre-screen-required | arithmetic | 11 |
| article_extended | cuccaro_adder_n4 | pre-screen-required | applications | 12 |
| article_extended | gf_2pow4_mult | pre-screen-required | arithmetic | 12 |
| article_extended | hamming_weight_n6 | pre-screen-required | applications | 12 |
| article_extended | hamming_weight_n7 | pre-screen-required | applications | 13 |
| article_extended | nc_tof_5 | pre-screen-required | arithmetic | 15 |
| article_extended | gf_2pow5_mult | pre-screen-required | arithmetic | 15 |
| article_extended | cuccaro_adder_n5 | pre-screen-required | applications | 16 |
