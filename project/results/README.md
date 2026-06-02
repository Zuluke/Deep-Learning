# Results layout

The top level of `results/` is reserved for active artifacts.

- `csv/`, `figures/`, `reports/`: current Entrega/paper outputs.
- `reproducibility/`: tensor-level reproduction of public AlphaTensor-Quantum
  artifacts.
- `verification/`: formal verification summaries and proof artifacts.
- `alphaq_split_reward/`, `alphaq_split_reward_smoke/`,
  `alphaq_split_prior_grid/`: current AlphaQuantum split-reward runs.
- `logs/`, `models/`: active training logs and lightweight model artifacts.

Historical experiment outputs are preserved under `archive/old_experiments/`.
Temporary scratch outputs and loose `tmp_*` files live under `archive/tmp/`.
New large experiment outputs should stay ignored by default unless they are
explicitly promoted to a report, CSV, figure, or archived artifact.
