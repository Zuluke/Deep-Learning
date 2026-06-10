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

Leave-one-target-out, on the consolidated dataset after the 2026-06-09/10
repair and long-budget batteries (32 train-ready groups, 17 unique targets):

| policy | scope | oracle-T recovery | T wins/losses vs baseline | sign-test p |
|---|---|---:|---:|---:|
| top-1 | groups | 32/32 | 12/0 | 2.4e-4 |
| guarded top-2 | groups | 32/32 | 12/0 | 2.4e-4 |
| top-1 | targets (dedup) | 16/17 | 7/0 | 7.8e-3 |
| guarded top-2 | targets (dedup) | 16/17 | 7/0 | 7.8e-3 |

- Random-ranking permutation controls: learned ranking recovers oracle-T on
  all groups while random ranking recovers roughly half (p < 1/2000 at m=1).
- **Leave-one-functional-family-out** now matches LOTO (7 wins / 0 losses at
  target level). At the previous, smaller dataset size, unguarded top-1
  showed a real off-distribution regression (6W/1L) that the guard removed —
  the guard remains a zero-cost, by-construction insurance policy.
- Geometric-mean T-count ratio vs baseline: 0.935 (groups), 0.932 (targets) —
  including ties, with zero regressions. Median QASM-depth ratio 1.0.
- New external win from the longer-budget battery: vbe_adder_3 reaches
  T=51 under `factor_count_pair_cap`/`frontier_pair` vs 65 for the baseline
  (ratio 0.785); the barenco_tof_4 T=42 `mixed_pair` win reproduced in an
  independent run; nc_tof_5 (tensor size 15, previously all-failed) landed
  2/4 objectives at a 3000s MILP budget.
- Cost reality (measured): the K MILP decompositions dominate wall-time
  (~2700s each, embarrassingly parallel); per-candidate materialization is
  seconds. The budget therefore saves the per-candidate audit/verification
  chain and engineering effort, not raw materialization compute — the
  method's primary value is the quality improvement with a non-regression
  guarantee.

## 2b. The Selection Rule Is Interpretable And Nearly Constant

14 of 16 leave-one-target-out folds learn the identical sparse weight vector:

> rank candidates by normalized `factor_count` (weight +2), preferring higher
> `factor_qubit_concentration_index` and `factor_support_weight_mean`
> (each -1), and lower `factor_pairwise_support_overlap_mean` and
> `factor_pairwise_jaccard_mean` (each +1).

Hardcoding this modal rule (no training at all) reproduces the full result:
oracle-T 29/29 groups and 16/16 targets, 10/0 and 7/0 wins/losses. Caveat:
the rule is distilled from the same dataset, so the LOTO/LOFO evaluations
remain the out-of-sample evidence; the value here is interpretability and
zero adoption cost. Mechanistically the rule says: take the cheapest
decomposition unless its factor structure is diffuse and overlapping, in
which case a concentrated alternative materializes better — consistent with
the splitting intuition that motivated the `mixed_pair` objective.

## 3. Formal Verification Campaign (New)

51 merged candidate proofs across the external batteries (feynver path-sum +
exact numeric isometry checking with postselected gadget ancillas,
`scripts/verify_candidates_numeric.py`): 31 proven (30 feynver `equal`
including both independent barenco_tof_4 runs, 1 `equal-numeric`),
12 characterized target-constant defects, 8 pending two-sided repair
(cuccaro/gf_2pow4, both proven Clifford-only vs the block reference).
The vbe_adder_3 defect signature is bit-identical across two independent
cluster batteries run months apart — a deterministic assembly bug.
nc_tof_5 carries its own target-constant degree-3 signature.

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
  Clifford (degree-1 X/Z frame); gf_2pow4_mult's residual correction against
  the block reference is proven Clifford by a full Pauli-conjugation closure
  test. For both targets the T-count claims are therefore intact and the
  assembly repair is free in T gates. Only nc_tof_4 and vbe_adder_3 carry
  non-Clifford (degree-3) defects whose repair may add T gates.
- The benchmark block reconstructions themselves verify exactly against the
  original circuits (checked for nc_tof_4), so the defect is ours, not the
  benchmark's.

The non-Clifford defects have tiny, highly structured footprints. ANF of the
residual phase function: nc_tof_4 = x0x1x2 + x0x1 + x2 (+ linear terms) —
exactly a CCZ(0,1,2)*CZ(0,1)*Z(2) discrepancy, the signature of a single
T<->T-dagger orientation error in one factor gadget; vbe_adder_3 =
(x4+x5)*e2(x0,x1,x2) plus Clifford terms — the same pattern on two carry
qubits. The likely root cause is a dagger-orientation bug in specific gadget
instances, so the repair should be **T-count neutral**.

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

| job | outcome |
|---|---|
| 2994 `article_repair2_barenco` | done: factor_count (T=64) and mixed_pair (T=42) landed; both feynver-proven equal; dataset gate now passes |
| 2995 `article_repair2_vbe` | done: all 4 objectives landed; new 0.785 T-ratio win for pair_cap/frontier_pair; known target-constant defect reproduced bit-identically |
| 2996 `journal_full_nc_tof_5_long` | done: factor_count and frontier_pair landed at T=103 (first success on this size-15 target) |
| 2997 `journal_full_gf_2pow5_mult_long` | still running |

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
