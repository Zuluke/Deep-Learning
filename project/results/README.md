# Results Map

The top-level `results/` folder contains generated outputs. Most large
experiment directories are intentionally not versioned. Promote only compact
CSV/figure/report artifacts that are part of the current story.

## Active Directories

- `csv/`: compact result tables.
- `figures/`: compact figures used by reports or the paper draft.
- `reports/`: human-readable summaries.
- `verification/`: formal-verification summaries and proof artifacts.
- `reproducibility/`: public AlphaTensor-Quantum reproduction outputs.
- `logs/`: cluster and local run logs.
- `archive/`: old experiments and scratch outputs kept out of the active view.

## Current AlphaQ Journal Checkpoint

Primary handoff document:

- `../docs/alphaq_journal_checkpoint.md`

Primary current-state figure:

- `figures/alphaq_journal_current_state.png`

Primary CSVs:

- `csv/alphaq_journal_evidence_gates.csv`
- `csv/alphaq_external_runs_consolidated.csv`
- `csv/alphaq_split_select_summary.csv`
- `csv/alphaq_split_select_details.csv`
- `csv/alphaq_objective_selection_dataset.csv`
- `csv/alphaq_journal_next_battery.csv`

Compact source CSVs needed to refresh that checkpoint are also versioned:

- `csv/alphaq_decomposition_objective_ablation.csv`
- `csv/alphaq_decomposition_objective_holdout_ablation.csv`
- `csv/alphaq_objective_beam_policy_grid.csv`
- `csv/alphaq_decomposition_objective_external_validation*.csv`
- `csv/alphaq_objective_beam_policy_external_validation*_grid.csv`
- `csv/alphaq_external_validation_readiness.csv`

Primary reports:

- `reports/alphaq_journal_evidence.md`
- `reports/alphaq_journal_current_state.md`
- `reports/alphaq_journal_battery_commands.md`

Interpretation: this checkpoint is promising but not journal-ready. It supports
target-dependent AlphaQ objective selection, while scale, coverage, and formal
verification remain open.

## Entrega / Paper Artifacts

Core tables:

- `csv/entrega1_metrics.csv`
- `csv/entrega1_metrics_formally_verified.csv`

Core reports:

- `reports/paper_reproduction_summary.md`
- `reports/entrega1_experimental_summary.md`
- `reports/entrega1_formally_verified_summary.md`
- `reports/formal_verification_entrega1_summary.md`

Core figures:

- `figures/entrega1/`
- `figures/entrega1_formal/`

## AlphaQ Development Artifacts

The current active AlphaQ line has two promoted artifact layers:

1. Shared-parity/materialization studies:
   - `csv/alphaq_shared_parity_study.csv`
   - `csv/alphaq_shared_parity_study_expansion_pilot.csv`
   - `csv/alphaq_shared_parity_expanded_candidates.csv`

2. Selector and journal evidence:
   - listed in the checkpoint section above.

Objective/materializer controls are active as scripts and tests, but their
large or highly detailed generated outputs should stay ignored unless they
become part of a checkpoint. Historical experiment outputs live under
`archive/old_experiments/` so they remain available without defining the active
story.

## Rule For New Results

When a new experiment finishes:

1. Keep large run directories ignored.
2. Promote only the compact CSV/report/figure needed to understand the result.
3. Add the promoted artifact to this README only if it is part of the active
   narrative.
4. Move obsolete exploratory outputs to `archive/old_experiments/` if they are
   in the way.
