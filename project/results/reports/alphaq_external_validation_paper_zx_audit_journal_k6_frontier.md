# ZX Border Detector Comparison

CSV: `results/csv/alphaq_external_validation_paper_zx_audit_journal_k6_frontier.csv`.

This compares the existing row-border ZX detector with a paper-style detector inspired by arXiv:2504.16004. The paper-style detector first converts the PyZX graph to graph-like form, applies Clifford simplification as an operational non-Clifford pushing stage, and then applies the same recursive crossing-gate closure used by the baseline detector.

Rows read: 51.
Rows with local QASM and both detectors OK: 51.
Rows missing local QASM: 0.

## Detector Delta

| relation | rows |
|---|---:|
| paper-smaller-core | 51 |
| tie | 0 |
| paper-larger-core | 0 |

## Objective-Level Comparison

Lower non-Clifford depth/ratio is better. Each row compares the best available candidate for `factor_count` against a split-aware objective on the same target.

| objective | legacy better/tie/worse | paper better/tie/worse | comparable targets |
|---|---:|---:|---:|
| frontier_pair | 3/1/1 | 2/1/2 | 5 |
| t_preserving_frontier_pair | 5/0/1 | 5/0/1 | 6 |

## Largest Paper-Style Core Reductions

| target | objective | materializer | legacy NC depth | paper NC depth |
|---|---|---|---:|---:|
| gf_2pow7_mult | t_preserving_frontier_pair | selected-shared-parity | 917 | 524 |
| gf_2pow7_mult | frontier_pair | selected-shared-parity | 953 | 562 |
| gf_2pow7_mult | factor_count | selected-shared-parity | 931 | 556 |
| hamming_weight_n8 | factor_count | selected-shared-parity | 697 | 338 |
| gf_2pow7_mult | factor_count | selected-beam-shared-parity-w4 | 809 | 492 |
| gf_2pow7_mult | frontier_pair | selected-beam-shared-parity-w4 | 819 | 516 |
| gf_2pow7_mult | frontier_pair | selected-beam-shared-parity-w16 | 796 | 509 |
| gf_2pow7_mult | factor_count | selected-beam-shared-parity-w16 | 787 | 502 |

## Interpretation

Use this report as an external audit, not as a replacement for the AlphaQ objective. If split-aware objectives improve under `paper_primary_nc_depth_ratio` or `paper_zx_best_nonclifford_depth`, then the internal AlphaQ-only objective is producing candidates that look better under a ZX-calculus splitting lens.
