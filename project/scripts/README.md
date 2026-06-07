# Active Script Map

This directory contains the active project scripts. Historical one-off analyses
belong in `../archive/scripts/`.

## 1. Paper / Entrega Reproduction

Entry points:

- `reproduce_paper_results.py`: tensor-level reproduction of public
  AlphaTensor-Quantum decompositions.
- `compute_metrics.py`: builds method-level metrics from available QASM
  artifacts.
- `make_entrega1_outputs.py`: creates Entrega/paper CSVs, figures, and reports.
- `run_formal_verification.py`: optional formal verification over promoted QASM
  candidates.
- `run_draft1_pipeline.py`: end-to-end Draft 1 pipeline wrapper.

## 2. Structural / Tensor-V3 Audit

Entry points:

- `replay_public_decompositions.py`: replays public decompositions and can
  rerank with `--selection-objective tensor-v3`.
- `structural_target.py`: pure structural target metrics.
- `tensor_split_core.py`: AlphaQ-only tensor-splitting metrics.
- `analyze_tensor_v3_selection_ablation.py`
- `analyze_tensor_v3_phase_slack_sensitivity.py`
- `compare_tensor_v3_profiles.py`
- `analyze_tensor_v3_guard_surface.py`
- `materialize_tensor_v3_guarded_profile.py`

Use this group when studying splitting proxies/frontiers without changing the
AlphaQ training loop.

## 3. AlphaQ Split-Reward / Training

Entry points:

- `run_demo_train.py`: upstream-style training/smoke runs with optional
  split-reward flags.
- `run_split_reward_ablation.py`
- `run_split_reward_target_sweep.py`
- `run_split_prior_grid.py`
- `analyze_split_reward_sweep_results.py`
- `analyze_split_reward_intermediates.py`
- `consolidate_split_reward_sweep.py`

Use this group for loop-level reward experiments. These are separate from the
current journal checkpoint, which is mostly objective selection after
decomposition.

## 4. Shared-Parity / Objective Selection

Entry points:

- `optimize_linear_span_candidate.py`: builds AlphaQ tensor decompositions under
  different objectives.
- `materialize_shared_parity_candidate.py`: turns candidate factors into QASM.
- `run_decomposition_objective_ablation.py`: compares objective variants under
  the same materializer.
- `run_beam_materializer_ablation.py`
- `run_best_objective_beam_ablation.py`
- `run_objective_beam_policy_grid.py`
- `factor_concentration_metrics.py`
- `verify_beam_materializer_candidates.py`

This is the main machinery behind the current `factor_count`,
`factor_count_pair_cap`, and `mixed_pair` comparisons.

## 5. Journal Evidence Layer

Entry points:

- `run_alphaq_external_validation_pipeline.py`: external validation pipeline for
  target batches.
- `consolidate_alphaq_external_runs.py`: folds standard, night-long, and
  journal-battery runs into one table.
- `build_alphaq_objective_selection_dataset.py`: builds the supervised
  objective-selection dataset.
- `analyze_alphaq_split_select.py`: evaluates Split-Select policies.
- `analyze_alphaq_journal_evidence.py`: emits evidence gates and next-battery
  recommendations.
- `build_alphaq_journal_battery_commands.py`: turns next-battery rows into
  runnable or blocked cluster commands.
- `refresh_alphaq_journal_evidence.py`: runs the journal refresh chain.
- `plot_alphaq_journal_state.py`: regenerates the current-state figure.

Recommended refresh:

```bash
PYTHONPATH=.:external uv run python scripts/refresh_alphaq_journal_evidence.py
PYTHONPATH=.:external uv run python scripts/plot_alphaq_journal_state.py
```

## 6. Cluster Helpers

- `setup_env.sh`: local environment setup helper.
- `run_split_reward_cluster_sweep.sh`
- `slurm_split_reward_target_sweep.sbatch`

The external-validation submit wrapper lives at the repository root:

```text
/Users/caio/Deep-Learning/submit_external_validation_apuana.sh
```

## Maintenance Rule

If a script is not part of one of the groups above, either document why it is
active or move it to `../archive/scripts/`.
