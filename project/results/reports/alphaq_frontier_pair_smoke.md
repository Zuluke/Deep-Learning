# AlphaQ Frontier-Pair Smoke Comparison

CSV: `results/csv/alphaq_frontier_pair_smoke.csv`.

`frontier_pair` is an AlphaQ-only objective. ZX is used here only as an external paper-style audit.

| target | objective | T-count | QASM ratio | legacy primary | paper primary | paper vs factor_count | paper NC-depth vs factor_count |
|---|---|---:|---:|---:|---:|---:|---:|
| mod_5_4 | factor_count | 7 | 0.678 | 0.2034 | 0.3721 | 1 | 1 |
| mod_5_4 | mixed_pair | 7 | 0.678 | 0.2034 | 0.3721 | 1 | 1 |
| mod_5_4 | frontier_pair | 7 | 0.678 | 0.2034 | 0.3721 | 1 | 1 |
| gf_2pow2_mult | factor_count | 17 | 3.119 | 1.091 | 0.925 | 1 | 1 |
| gf_2pow2_mult | mixed_pair | 17 | 2.619 | 0.8409 | 0.75 | 0.8108 | 0.8108 |
| gf_2pow2_mult | frontier_pair | 17 | 2.619 | 0.8409 | 0.75 | 0.8108 | 0.8108 |

## Reading

- `gf_2pow2_mult`: `frontier_pair` paper-primary=0.75, factor_count paper-primary=0.925, mixed_pair paper-primary=0.75.
- `mod_5_4`: `frontier_pair` paper-primary=0.3721, factor_count paper-primary=0.3721, mixed_pair paper-primary=0.3721.
