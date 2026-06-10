# AlphaQ Guarded Portfolio Selection: Journal Checkpoint

Snapshot date: 2026-06-09 (supersedes the 2026-06-07 checkpoint).

This document consolidates the current scientific state of the AlphaQuantum/
AlphaQ line after the portfolio-budget formalization and the formal
verification campaign.

## 1. The Method (As It Should Appear In The Manuscript)

**Guarded objective-portfolio selection with a materialization budget.**

1. **Portfolio generation.** For a target tensor, generate candidate
   decompositions under the K=4 deployed AlphaQ objectives: `factor_count`
   (baseline), `factor_count_pair_cap`, `mixed_pair`, `frontier_pair`.
2. **Cheap-feature ranking.** Rank candidates with a sparse linear scorer over
   factor-structure features only (factor count, qubit concentration, support
   weight, pairwise overlap, pairwise Jaccard) — no ZX metrics, no
   materialization, no QASM in the loop. The scorer is trained
   leave-one-target-out.
3. **Budgeted, guarded materialization.** Materialize only `m` candidates.
   The guarded variant always spends one budget slot on the baseline
   candidate, so for any `m >= 2` the final circuit is **never worse than the
   AlphaQuantum baseline in T-count by construction**.
4. Keep the best materialized circuit by (T-count, QASM depth, primary
   non-Clifford depth ratio).

Two findings motivate this framing and are results in their own right:

- **No single objective dominates.** Post-hoc oracle objectives split
  13/9/5/2 across the four portfolio members (29 groups). A "better single
  cost function" is not supported by the data.
- **A-priori circuit-feature classifiers are at chance.** Circuit-conditioned
  selectors (softmax/kNN/centroid over tensor-level features) do not beat a
  shuffled-label control at the current dataset size; the signal lives in the
  post-decomposition factor structure.

## 2. Headline Evidence (`scripts/analyze_alphaq_portfolio_budget.py`)

Leave-one-target-out, on the consolidated dataset (29 train-ready groups, 16
unique targets):

| policy | scope | oracle-T recovery | T wins/losses vs baseline | sign-test p | random-ranking recovery | permutation p |
|---|---|---:|---:|---:|---:|---:|
| top-1 | groups | 29/29 | 10/0 | 9.8e-4 | 15.5/29 | < 1/2000 |
| guarded top-2 | groups | 29/29 | 10/0 | 9.8e-4 | 24.0/29 | 1e-3 |
| top-1 | targets (dedup) | 16/16 | 7/0 | 7.8e-3 | 8.7/16 | < 1/2000 |
| guarded top-2 | targets (dedup) | 16/16 | 7/0 | 7.8e-3 | 12.5/16 | 5e-3 |

- The budget-1 learned selection already recovers oracle T-count on every
  group; random ranking recovers ~54%.
- **Leave-one-functional-family-out** (selector never sees the held-out
  construction family): pure top-1 shows its first regression (6 wins / 1
  loss at target level), while **guarded top-2 removes the regression**
  (6 wins / 0 losses, sign-test p = 0.016, oracle-T 15/16). This is the
  empirical case for the guard: it converts a heuristic that can fail
  off-distribution into a method that is never worse by construction.
  (`results/csv/alphaq_portfolio_budget_lofo_summary.csv`)
- Geometric-mean T-count ratio vs baseline: 0.950 (groups), 0.925 (targets) —
  including ties, with zero regressions.
- Median QASM-depth ratio is 1.0 (depth-conservative).
- Cost: the oracle needs K materializations + audits per target; guarded
  top-2 needs 2. Decomposition-stage features are byproducts of optimization.

## 3. Formal Verification Campaign (New)

52 candidate proofs across the external batteries (feynver path-sum +
exact numeric isometry checking with postselected gadget ancillas,
`scripts/verify_candidates_numeric.py`):

- **Proven equal**: 30 feynver `equal` + 1 `equal-numeric` (mod_mult_55).
  This includes the flagship barenco_tof_4 `mixed_pair` candidate
  (T-ratio 0.656), proven equal by feynver.
- **Characterized assembly defects**: nc_tof_4 and vbe_adder_3 candidates
  equal the original composed with a **target-constant** signed basis
  permutation whose phase polynomial has GF(2) degree 3 (non-Clifford). The
  defect signature is *identical across all objectives of a target*
  (objective-independent), so it originates in the shared target
  assembly/correction step, not in any decomposition. Relative
  (selector-level) claims are internally consistent; absolute T-counts on
  these targets need the assembly repair.
- cuccaro_adder_n4 candidates equal the benchmark **block reference** up to a
  Clifford (degree-1) frame; the original-vs-blocks gap is the benchmark's
  own hopt gadget convention. gf_2pow4_mult is a clean scaled isometry with a
  correction still being characterized (Clifford conjugation test pending).
- The benchmark block reconstructions themselves verify exactly against the
  original circuits (checked for nc_tof_4), so the defect is ours, not the
  benchmark's.

**Action item (pre-submission blocker):** repair the shared-parity assembly
correction layer for the affected targets, re-materialize, re-verify.

## 4. Evidence Gates

Current: 4/6 pass.

| gate | status |
|---|---|
| `selector_loto` | pass |
| `external_generalization_coverage` | pass (12 complete external targets after fixing completeness accounting to the deployed 4-objective portfolio) |
| `external_nonbaseline_effect` | pass |
| `depth_control` | pass |
| `dataset_scale_and_label_diversity` | partial (29/30 train-ready groups) |
| `formal_verification_coverage` | fail until the assembly defect is repaired (29/44 merged rows proven, 7 characterized defects, 8 pending characterization) |

Bookkeeping fixes made in this checkpoint: external completeness was being
measured against six objectives, two of which (`depth_guarded_mixed_pair`,
`t_preserving_frontier_pair`) had never been run anywhere; the consolidation
also omitted the `article_core`/`article_extended` runs that contain
`frontier_pair`. Both are corrected; the deployed portfolio is K=4.

## 5. In-Flight Cluster Work (Apuana)

| job | purpose |
|---|---|
| 2994 `article_repair2_barenco` | repair the 3 failed objectives of the article_core barenco_tof_4 group (closes the dataset-scale gate: 29 -> 30 groups) |
| 2995 `article_repair2_vbe` | complete vbe_adder_3 mixed_pair |
| 2996 `journal_full_nc_tof_5_long` | failed target retry with +67% MILP budget (scaling evidence) |
| 2997 `journal_full_gf_2pow5_mult_long` | failed target retry with +67% MILP budget (scaling evidence) |

New runs record optimization/materialization wall-times, enabling an
empirical cost-asymmetry claim.

## 6. What We Can Claim Now

> Inside AlphaTensor-Quantum, the choice of decomposition objective is
> target-dependent (no portfolio member dominates). A guarded portfolio
> selection layer using only cheap factor-structure features recovers
> oracle-level T-count on all evaluated targets (16/16, permutation p < 1e-3)
> with zero T-count regressions by construction, at 2/K of the
> materialization cost. The largest verified improvement is a 34% T-count
> reduction (barenco_tof_4, formally proven equivalent).

Not yet claimable: absolute T-counts on nc_tof_4 / vbe_adder_3 /
cuccaro_adder_n4 / gf_2pow4_mult (pending assembly repair); scalability
beyond tensor size ~14 (pending the long-budget retries and a future
restricted-action method).

## 7. Reproduction

```bash
cd /Users/caio/Deep-Learning/project
PYTHONPATH=.:external uv run python scripts/refresh_alphaq_journal_evidence.py
PYTHONPATH=.:external uv run python scripts/analyze_alphaq_portfolio_budget.py
PYTHONPATH=.:external uv run python scripts/analyze_alphaq_verification_impact.py
```

Verification (requires feynver and the synced candidate artifacts):

```bash
PYTHONPATH=.:external uv run python scripts/verify_beam_materializer_candidates.py \
  --beam-csv results/csv/alphaq_objective_beam_policy_external_validation_article_core_grid.csv \
  --output-root results/verification/alphaq_external_article_core \
  --materializer-prefix selected-beam \
  --path-map "/home/CIN/cacl2/Deep-Learning/project=$PWD"
PYTHONPATH=.:external uv run python scripts/verify_candidates_numeric.py \
  --verification-roots results/verification/alphaq_external_article_core,results/verification/alphaq_external_article_extended,results/verification/alphaq_external_night_long \
  --output-csv results/verification/alphaq_external_numeric/verification_numeric.csv \
  --report-path results/verification/alphaq_external_numeric/verification_numeric.md
```
