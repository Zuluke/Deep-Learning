# Summary: arXiv 2504.16004 Clifford and Non-Clifford Splitting

The paper proposes a ZX-calculus procedure to detect a border separating a Clifford section from a non-Clifford section of a circuit. The procedure pushes non-Clifford spiders as far as possible, defines a preliminary border, then recursively expands the non-Clifford side whenever two-qubit gates cross the proposed border. The result is a circuit split of the form U_C U_NC or U_NC U_C, depending on whether the Clifford section is on the left or right.

The paper's metric is circuit-level and border-level: how much Clifford depth can be separated from the non-Clifford section, and what practical use cases follow from that separation. It is not an AlphaQuantum objective and does not optimize tensor decompositions directly.

For this project, the closest comparable metric is primary_nc_depth_ratio, which measures the ZX-detected non-Clifford core depth of a candidate relative to the original circuit depth. The AlphaQ mixed_pair objective is only a proxy: it penalizes mixed/pair-heavy tensor factors before QASM materialization, hoping that this yields a better separable circuit after synthesis.

Current project comparison:
- mixed_pair improves internal factor overlap/Jaccard proxies in most paired targets.
- On primary_nc_depth_ratio where ZX metrics are available, mixed_pair improves 4 targets, ties 1, and worsens 3 against factor_count.
- Therefore mixed_pair is directionally useful but not a reliable replacement for the paper's actual border detection metric.

Conclusion: the paper's method remains the more faithful evaluator of Clifford/non-Clifford splitting. The project contribution is different: use AlphaQ-only objectives to generate candidates that may later score better under such an external splitting audit.
