# ZX Border Detector Comparison

CSV: `results/csv/alphaq_external_validation_paper_zx_audit_article_core.csv`.

This compares the existing row-border ZX detector with a paper-style detector inspired by arXiv:2504.16004. The paper-style detector first converts the PyZX graph to graph-like form, applies Clifford simplification as an operational non-Clifford pushing stage, and then applies the same recursive crossing-gate closure used by the baseline detector.

Rows read: 96.
Rows with local QASM and both detectors OK: 96.
Rows missing local QASM: 0.

## Detector Delta

| relation | rows |
|---|---:|
| paper-smaller-core | 58 |
| tie | 5 |
| paper-larger-core | 33 |

## Objective-Level Comparison

Lower non-Clifford depth/ratio is better. Each row compares the best available candidate for `factor_count` against a split-aware objective on the same target.

| objective | legacy better/tie/worse | paper better/tie/worse | comparable targets |
|---|---:|---:|---:|
| factor_count_pair_cap | 2/1/3 | 3/2/1 | 6 |
| frontier_pair | 3/2/1 | 2/2/2 | 6 |
| mixed_pair | 2/1/2 | 2/1/2 | 5 |

## Largest Paper-Style Core Reductions

| target | objective | materializer | legacy NC depth | paper NC depth |
|---|---|---|---:|---:|
| vbe_adder_3 | factor_count_pair_cap | selected-shared-parity | 266 | 190 |
| vbe_adder_3 | factor_count_pair_cap | selected-beam-shared-parity-w4 | 227 | 174 |
| vbe_adder_3 | frontier_pair | selected-shared-parity | 196 | 143 |
| vbe_adder_3 | factor_count | selected-shared-parity | 170 | 119 |
| vbe_adder_3 | factor_count_pair_cap | selected-beam-shared-parity-w16 | 218 | 168 |
| vbe_adder_3 | frontier_pair | selected-beam-shared-parity-w4 | 178 | 128 |
| vbe_adder_3 | frontier_pair | selected-beam-shared-parity-w16 | 167 | 125 |
| vbe_adder_3 | factor_count | selected-beam-shared-parity-w16 | 148 | 108 |

## Interpretation

Use this report as an external audit, not as a replacement for the AlphaQ objective. If split-aware objectives improve under `paper_primary_nc_depth_ratio` or `paper_zx_best_nonclifford_depth`, then the internal AlphaQ-only objective is producing candidates that look better under a ZX-calculus splitting lens.
