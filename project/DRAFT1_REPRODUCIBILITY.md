# Draft 1 Reproducibility Protocol

This repository is organized so that Draft 1 can be audited from three levels of evidence: tensor-level reproduction of the AlphaTensor-Quantum paper, circuit-level baseline artifacts for the course report, and formal equivalence checks for the subset of reconstructed circuits where the verifier terminates.

## Evidence Layers

1. **Paper reproduction, primary evidence.** The command `uv run python scripts/reproduce_paper_results.py` loads the official DeepMind `.npz` decompositions, reconstructs the corresponding signature tensors over GF(2), validates equality against vendored `.tensor.npy` targets, detects Toffoli/CS gadget patterns, and recomputes the effective T-count reported in the paper. The current local run gives `96/96` exact matches against the encoded paper values and `1038/1038` tensor-equal decomposition candidates.

2. **Circuit-level Entrega 1 artifacts, practical evidence.** The command `uv run python scripts/make_entrega1_outputs.py` builds the Draft 1 tables and figures from `results/csv/final_metrics.csv`, comparing `Original`, `PyZX`, and `AlphaTensor-public` on selected Clifford+T benchmarks. This layer is used for the report figures and for the structural Clifford/non-Clifford discussion.

3. **Formal verification, complementary audit.** The command `uv run python scripts/run_formal_verification.py --scope entrega1 --timeout-sec 30` checks normalized QASM pairs through `circuit-to-tensor verify`, which calls `feynver`. The complete audit may include `timeout` or `inconclusive` rows, so the main formal table `results/csv/entrega1_metrics_formally_verified.csv` keeps only candidates with `formal_verification_status == equal`, plus their original baselines.

## One-command Pipeline

The full Draft 1 pipeline can be rerun with:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/run_draft1_pipeline.py --run-formal
```

For a faster check that reuses the latest formal verification CSV:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/run_draft1_pipeline.py
```

The pipeline writes `results/reports/draft1_pipeline_manifest.json`, recording command return codes, elapsed time, and output tails for auditing.

## Expected Core Artifacts

- `results/reproducibility/paper/paper_results_long.csv`
- `results/reproducibility/paper/paper_tensor_validation.csv`
- `results/reproducibility/paper/paper_benchmark_comparison.csv`
- `results/reproducibility/paper/paper_family_summary.csv`
- `results/reproducibility/paper/paper_summary.json`
- `results/csv/entrega1_metrics.csv`
- `results/csv/entrega1_metrics_formally_verified.csv`
- `results/figures/entrega1/*.png`
- `results/reports/paper_reproduction_summary.md`
- `results/reports/entrega1_experimental_summary.md`
- `results/reports/entrega1_formally_verified_summary.md`
- `results/reports/formal_verification_entrega1_summary.md`
- `results/reports/draft1_pipeline_manifest.json`
- `notebooks/entrega1_reproducao_alphatensor_quantum.ipynb`
- `paper/main.tex`
- `paper/main.pdf`

## Integrity Criteria for Draft 1

The Draft 1 results should be considered internally consistent only if all of the following are true:

- `paper_summary.json` reports `num_exact_matches == num_comparison_rows`.
- `paper_summary.json` reports `num_tensor_equal_candidates == num_long_rows`.
- `pytest tests -q` passes.
- `jupyter nbconvert --execute notebooks/entrega1_reproducao_alphatensor_quantum.ipynb` succeeds.
- The formal table contains no optimized candidate with `timeout` or `inconclusive`.
- `paper/main.tex` compiles to `paper/main.pdf` without unresolved citations or references.
- The paper cites the tensor-level reproduction as the primary AlphaTensor-Quantum replication and treats QASM/feynver results as a complementary audit.

## Scope Statement

Draft 1 does not retrain AlphaTensor-Quantum. This is intentional and scientifically defensible for the current delivery: the project reproduces the public optimized decompositions, validates them tensorially, and adds an independently implemented structural analysis layer inspired by Clifford/non-Clifford splitting. Full training or reward modification is reserved for later project stages.
