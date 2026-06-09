# ZX Border Detector Comparison

CSV: `results/csv/zx_border_detector_comparison.csv`.

This compares the existing row-border ZX detector with a paper-style detector inspired by arXiv:2504.16004. The paper-style detector first converts the PyZX graph to graph-like form, applies Clifford simplification as an operational non-Clifford pushing stage, and then applies the same recursive crossing-gate closure used by the baseline detector.

Rows read: 228.
Rows with local QASM and both detectors OK: 228.
Rows missing local QASM: 0.

## Detector Delta

| relation | rows |
|---|---:|
| paper-smaller-core | 221 |
| tie | 2 |
| paper-larger-core | 5 |

## Objective-Level Comparison

Lower non-Clifford depth/ratio is better. Each row compares the best available candidate for `factor_count` against a split-aware objective on the same target.

| objective | legacy better/tie/worse | paper better/tie/worse | comparable targets |
|---|---:|---:|---:|
| factor_count_pair_cap | 4/0/4 | 2/1/5 | 8 |
| mixed_pair | 2/0/6 | 2/0/6 | 8 |

## Largest Paper-Style Core Reductions

| target | objective | materializer | legacy NC depth | paper NC depth |
|---|---|---|---:|---:|
| hwb_6 | conservative | hwb_6:combo7 | 915 | 121 |
| hwb_6 | conservative | hwb_6:combo3 | 859 | 118 |
| hwb_6 | conservative | hwb_6:combo6 | 847 | 122 |
| hwb_6 | conservative | hwb_6:combo5 | 842 | 118 |
| hwb_6 | conservative | hwb_6:combo1 | 814 | 123 |
| hwb_6 | conservative | hwb_6:combo4 | 774 | 119 |
| hwb_6 | conservative | hwb_6:combo9 | 760 | 121 |
| hwb_6 | conservative | hwb_6:combo0 | 758 | 122 |

## Interpretation

Use this report as an external audit, not as a replacement for the AlphaQ objective. If split-aware objectives improve under `paper_primary_nc_depth_ratio` or `paper_zx_best_nonclifford_depth`, then the internal AlphaQ-only objective is producing candidates that look better under a ZX-calculus splitting lens.
