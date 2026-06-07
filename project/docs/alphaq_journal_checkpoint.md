# AlphaQ Split-Select Journal Checkpoint

Snapshot date: 2026-06-07.

This document consolidates the current AlphaQuantum/AlphaQ Split-Select development state. It is intended as a readable checkpoint for the project owner and for a third agent reviewing the next scientific steps.

## 1. What We Are Studying

The working hypothesis is that changing the AlphaQuantum synthesis objective away from pure `T-count` can sometimes improve the final circuit because it changes the structure of the tensor decomposition. In particular, objectives that encourage factor concentration or reduce mixed/overlapping structure may make later synthesis/materialization easier.

The current method is not a ZX-in-the-loop method and not a reward-learning result. It is an AlphaQ-only objective-selection layer:

1. For each target circuit, generate candidate tensor decompositions using several AlphaQ objectives.
2. Materialize each decomposition to QASM using the same shared-parity/beam materializer.
3. Compare the resulting circuits by T-count and QASM depth.
4. Learn/evaluate a lightweight selector that chooses which AlphaQ objective to use for a target.

The key point is that the method is not "always use the splitting/mixed objective." The evidence so far supports adaptive objective selection.

## 2. Objective Variants

The external battery currently compares three AlphaQ objective variants:

| variant | role | intuition |
|---|---|---|
| `factor_count` | baseline | Minimize the number of decomposition factors, closest to the original T-count-oriented behavior. |
| `factor_count_pair_cap` | concentration-aware baseline | Keep factor count objective but impose pair-overlap caps to avoid overly entangled/reused factor structure. |
| `mixed_pair` | splitting-oriented objective | Penalize mixed/pair structure more directly, encouraging decompositions that may concentrate non-Clifford structure. |

After decompositions are generated, the materializer and downstream audits are kept fixed. This makes the comparison primarily about the AlphaQ objective choice.

## 3. Current Result Summary

The main plot is:

`results/figures/alphaq_journal_current_state.png`

It compares, for each complete external benchmark, the best objective against the `factor_count` baseline. Bars are T-count ratios and diamonds are QASM-depth ratios. Values below `1.0` improve over the baseline.

Current best external objective relative to `factor_count`:

| target | best objective | T-count ratio | QASM-depth ratio | interpretation |
|---|---|---:|---:|---|
| `barenco_tof_4` | `mixed_pair` | 0.656 | 0.893 | Strong win in both T-count and depth. |
| `gf_2pow4_mult` | `factor_count_pair_cap` | 0.965 | 1.002 | Small T-count win with essentially tied depth. |
| `mod_mult_55` | `mixed_pair` | 1.000 | 0.940 | Same T-count, lower QASM depth. |
| `hamming_weight_n6` | `factor_count_pair_cap` | 1.000 | 0.971 | Same T-count, lower QASM depth. |
| `cuccaro_adder_n4` | `factor_count` | 1.000 | 1.000 | Baseline remains best. |
| `hamming_weight_n7` | `factor_count` | 1.000 | 1.000 | Baseline remains best. |
| `nc_tof_4` | `factor_count` | 1.000 | 1.000 | Baseline remains best. |
| `vbe_adder_3` | `factor_count` | 1.000 | 1.000 | Baseline remains best. |

Interpretation: the signal is real but heterogeneous. Some circuits benefit from `mixed_pair`, some from `pair_cap`, and some should stay with `factor_count`.

## 4. Selector Evidence

The current Split-Select policy is `split_select_linear_alphaq`.

From `results/csv/alphaq_split_select_summary.csv`:

| policy | oracle matches | T-count wins vs baseline | QASM wins vs baseline |
|---|---:|---:|---:|
| `baseline_factor_count` | 8/18 | 0/18 | 0/18 |
| `split_select_linear_alphaq` | 13/18 | 5/18 | 5/18 |
| `oracle_posthoc` | 18/18 | 5/18 | 8/18 |

This means the learned/linear AlphaQ-only selector recovers a substantial fraction of the post-hoc objective choice while staying conservative on depth.

The selector is not yet a journal-level learned model. It is better described as prototype evidence that target-dependent objective selection is worthwhile.

## 5. Evidence Gates

Current gate audit: `results/csv/alphaq_journal_evidence_gates.csv`.

| gate | status | current evidence |
|---|---|---|
| `selector_loto` | pass | Split-Select gets 13/18 oracle matches vs 8/18 for baseline; 5/18 T-count wins. |
| `external_nonbaseline_effect` | pass | 4 non-baseline external improvements and 0 non-baseline regressions. |
| `depth_control` | pass | Median QASM ratio is 1.0; QASM non-worse in 15/18 evaluated groups. |
| `dataset_scale_and_label_diversity` | partial | 18/22 train-ready groups; need more scale and diversity. |
| `external_generalization_coverage` | partial | 8 complete external targets, 1 partial target, 2 failed targets; only 2 external families. |
| `formal_verification_coverage` | partial | 24 verification rows, 21 equal, 3 inconclusive, 0 failures. |
| `overall_journal_readiness` | not-yet-journal-ready | Passed gates are promising but not enough for a robust journal claim. |

Current decision: not yet journal-ready.

## 6. Current External Battery Status

The current external battery includes:

Completed/usable targets:

- `barenco_tof_4`
- `cuccaro_adder_n4`
- `gf_2pow4_mult`
- `hamming_weight_n6`
- `hamming_weight_n7`
- `mod_mult_55`
- `nc_tof_4`
- `vbe_adder_3`

Partial:

- `cuccaro_adder_n5`

Failed under the current full-action objective-grid setup:

- `gf_2pow5_mult`
- `nc_tof_5`

As of the latest check, Slurm job `2393` for `cuccaro_adder_n5` is still running on Apuana. It has no stderr. Its current checkpoint contains:

- `factor_count`: materialized but very poor, with `T-count=590` and `qasm_depth_ratio=17.347`.
- `factor_count_pair_cap`: failed.
- `mixed_pair`: still running/pending in the job output.

This strongly suggests that larger targets are not well served by simply repeating the same full-action MILP setup.

## 7. What We Can Claim Now

Reasonable current claim:

> Objective choice inside AlphaQ matters. A target-dependent Split-Select layer can improve T-count and/or QASM depth over the factor-count baseline on several external circuits while preserving conservative behavior on cases where factor-count remains best.

Stronger mechanistic interpretation:

> The gains are consistent with the idea that concentrating or reshaping non-Clifford tensor factors can make the downstream materialization more effective. However, the best objective is circuit-dependent, so the contribution should be framed as adaptive AlphaQ objective selection rather than as a universal mixed/splitting objective.

Claims that are not yet justified:

- This is not yet robust journal-level evidence.
- This is not yet a scalable method for larger tensors.
- This is not yet a formally verified result for all promoted candidates.
- This is not yet evidence that `mixed_pair` alone dominates `factor_count`.

## 8. Recommended Next Actions

The immediate next step should not be another blind full-action job on the same hard targets.

The current next-battery file is:

`results/csv/alphaq_journal_next_battery.csv`

It recommends:

| target | recommended stage | reason |
|---|---|---|
| `gf_2pow5_mult` | `tensor-v3-screen` | Full-action attempt failed; screen the tensor/frontier first before repeating. |
| `nc_tof_5` | `tensor-v3-screen` | Full-action attempt failed; screen the tensor/frontier first before repeating. |
| `cuccaro_adder_n5` | `tensor-v3-screen` | Current full-action behavior is poor/partial; screen before further full-action. |
| `hwb_6` | `restricted-action-pilot` | Tensor size is too large for full enumeration; needs a restricted/beam action policy. |

Practical recommendation:

1. Let job `2393` finish, but do not make it the main bottleneck.
2. Freeze this checkpoint as the current evidence package.
3. Ask a third agent to evaluate whether the scientific story should focus on:
   - adaptive objective selection;
   - mechanism via factor concentration;
   - restricted action spaces for scaling;
   - or a narrower benchmark-family claim.
4. Only after that, run a larger and more decisive battery.

## 9. Files To Send To A Third Agent

Suggested compact packet:

1. `docs/alphaq_journal_checkpoint.md`
2. `results/figures/alphaq_journal_current_state.png`
3. `results/csv/alphaq_journal_evidence_gates.csv`
4. `results/csv/alphaq_external_runs_consolidated.csv`
5. `results/csv/alphaq_split_select_summary.csv`
6. `results/csv/alphaq_split_select_details.csv`
7. `results/csv/alphaq_objective_selection_dataset.csv`
8. `results/csv/alphaq_journal_next_battery.csv`
9. `results/reports/alphaq_journal_evidence.md`
10. `results/reports/alphaq_journal_current_state.md`

Optional source files if the agent wants implementation detail:

- `scripts/run_decomposition_objective_ablation.py`
- `scripts/analyze_alphaq_split_select.py`
- `scripts/analyze_alphaq_journal_evidence.py`
- `scripts/consolidate_alphaq_external_runs.py`
- `scripts/plot_alphaq_journal_state.py`

## 10. Suggested Prompt For A Third Agent

Use the following prompt:

```text
We are studying an AlphaQuantum/AlphaTensor-Quantum variant where the key intervention is not ZX-in-the-loop and not a new neural reward yet. The current method generates tensor decompositions under multiple AlphaQ-only objectives (`factor_count`, `factor_count_pair_cap`, `mixed_pair`), materializes them with a fixed shared-parity/beam pipeline, and then uses a lightweight Split-Select policy to choose the objective per circuit.

Please review the attached checkpoint and artifacts. I want you to assess:

1. Whether the current evidence supports the scientific claim that target-dependent AlphaQ objective selection is useful.
2. Whether the mechanism should be framed as factor concentration / splitting-inspired structure, or something more conservative.
3. Whether the current gate audit is a reasonable standard for journal readiness.
4. What the next decisive experiment should be, given that several larger full-action runs failed or became pathological.
5. Whether we should prioritize tensor-v3 screening, restricted action spaces, formal verification, or broader benchmark coverage.

Please be critical. Distinguish what is already supported from what remains speculative.
```
