# ZX Border Detector Comparison

CSV: `results/csv/alphaq_external_validation_paper_zx_audit_article_extended.csv`.

This compares the existing row-border ZX detector with a paper-style detector inspired by arXiv:2504.16004. The paper-style detector first converts the PyZX graph to graph-like form, applies Clifford simplification as an operational non-Clifford pushing stage, and then applies the same recursive crossing-gate closure used by the baseline detector.

Rows read: 80.
Rows with local QASM and both detectors OK: 80.
Rows missing local QASM: 0.

## Detector Delta

| relation | rows |
|---|---:|
| paper-smaller-core | 72 |
| tie | 1 |
| paper-larger-core | 7 |

## Objective-Level Comparison

Lower non-Clifford depth/ratio is better. Each row compares the best available candidate for `factor_count` against a split-aware objective on the same target.

| objective | legacy better/tie/worse | paper better/tie/worse | comparable targets |
|---|---:|---:|---:|
| factor_count_pair_cap | 2/0/3 | 2/1/2 | 5 |
| frontier_pair | 3/0/2 | 1/0/4 | 5 |
| mixed_pair | 3/0/2 | 3/0/2 | 5 |

## Largest Paper-Style Core Reductions

| target | objective | materializer | legacy NC depth | paper NC depth |
|---|---|---|---:|---:|
| gf_2pow4_mult | factor_count | selected-shared-parity | 165 | 113 |
| gf_2pow4_mult | frontier_pair | selected-shared-parity | 176 | 125 |
| gf_2pow4_mult | factor_count_pair_cap | selected-shared-parity | 162 | 113 |
| gf_2pow4_mult | mixed_pair | selected-shared-parity | 166 | 122 |
| gf_2pow4_mult | factor_count | selected-beam-shared-parity-w4 | 149 | 107 |
| gf_2pow4_mult | factor_count_pair_cap | selected-beam-shared-parity-w4 | 143 | 101 |
| gf_2pow4_mult | frontier_pair | selected-beam-shared-parity-w4 | 158 | 117 |
| hamming_weight_n7 | mixed_pair | selected-shared-parity | 111 | 71 |

## Interpretation

Use this report as an external audit, not as a replacement for the AlphaQ objective. If split-aware objectives improve under `paper_primary_nc_depth_ratio` or `paper_zx_best_nonclifford_depth`, then the internal AlphaQ-only objective is producing candidates that look better under a ZX-calculus splitting lens.
