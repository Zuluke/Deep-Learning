# Active scripts

This directory is intentionally kept small. Scripts here are part of the active
project surface:

- paper and Entrega reproduction;
- QASM metrics, formal verification, and structural auditing;
- AlphaQuantum split-reward training, sweeps, materialization, and candidate
  verification;
- shared helper modules used by those flows.

Current shared-parity checkpoint:

- `run_shared_parity_study.py --preset core` reproduces the four-target core
  shared-parity study.
- `run_shared_parity_study.py --preset expanded` runs the larger configured
  study surface. Use explicit `--targets` for pilots when local runtime or disk
  space matters.
- `analyze_materialized_candidates.py` combines materialized summaries and can
  annotate them with one or more `verification_summary.json` files.

Historical probes and one-off analyses live in `../archive/scripts/`. Move a
script back here only when it becomes part of the active workflow again.
