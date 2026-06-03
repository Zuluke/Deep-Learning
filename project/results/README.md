# Results layout

The top level of `results/` is reserved for active artifacts.

- `csv/`, `figures/`, `reports/`: current Entrega/paper outputs.
- `reproducibility/`: tensor-level reproduction of public AlphaTensor-Quantum
  artifacts.
- `verification/`: formal verification summaries and proof artifacts.
- `alphaq_split_reward/`, `alphaq_split_reward_smoke/`,
  `alphaq_split_prior_grid/`: current AlphaQuantum split-reward runs.
- `logs/`, `models/`: active training logs and lightweight model artifacts.

Promoted AlphaQ shared-parity checkpoint artifacts:

- `csv/alphaq_shared_parity_study.csv`: four-target core study.
- `csv/alphaq_shared_parity_study_expansion_pilot.csv`: four-target expansion
  pilot.
- `csv/alphaq_shared_parity_expanded_candidates.csv`: combined eight-target
  comparison with formal verification status.
- `reports/alphaq_shared_parity_*.md`: matching human-readable summaries.
- `figures/alphaq_shared_parity_*.png`: compact visual summaries.
- `verification/alphaq_shared_parity_*/verification_summary.json`: formal
  status summaries only; individual proof logs and candidate directories remain
  experiment outputs.

Historical experiment outputs are preserved under `archive/old_experiments/`.
Temporary scratch outputs and loose `tmp_*` files live under `archive/tmp/`.
New large experiment outputs should stay ignored by default unless they are
explicitly promoted to a report, CSV, figure, or archived artifact.
