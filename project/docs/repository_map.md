# Repository Map

This is the current navigation map for `/Users/caio/Deep-Learning/project`.

## What Is Active Now

There are two active project fronts:

1. **Entrega/paper reproduction**
   - Goal: reproduce/audit public AlphaTensor-Quantum decompositions and produce
     the course/paper artifacts.
   - Start with `scripts/reproduce_paper_results.py` and
     `scripts/make_entrega1_outputs.py`.

2. **AlphaQ Split-Select journal evidence**
   - Goal: test whether AlphaQ-only objective selection improves circuit
     materialization compared with the factor-count baseline.
   - Start with `docs/alphaq_journal_checkpoint.md`.
   - Refresh with `scripts/refresh_alphaq_journal_evidence.py`.

## Top-Level Folders

| path | purpose |
|---|---|
| `src/` | reusable reproduction library code. |
| `scripts/` | active CLI scripts and analysis modules. |
| `tests/` | active pytest suite. |
| `docs/` | human-readable project maps and scientific checkpoints. |
| `results/` | generated outputs; only compact promoted artifacts should be versioned. |
| `archive/` | old scripts/tests from inactive project stages. |
| `external/` | vendored upstream snapshots and benchmark tooling. |
| `notebooks/` | notebook artifacts for Entrega/paper workflows. |

## Current AlphaQ Evidence Chain

The active journal-evidence chain is:

```text
external validation runs
  -> consolidate_alphaq_external_runs.py
  -> build_alphaq_objective_selection_dataset.py
  -> analyze_alphaq_split_select.py
  -> analyze_alphaq_journal_evidence.py
  -> plot_alphaq_journal_state.py
```

One-command refresh:

```bash
cd /Users/caio/Deep-Learning/project
PYTHONPATH=.:external uv run python scripts/refresh_alphaq_journal_evidence.py
PYTHONPATH=.:external uv run python scripts/plot_alphaq_journal_state.py
```

Key outputs:

- `docs/alphaq_journal_checkpoint.md`
- `results/figures/alphaq_journal_current_state.png`
- `results/csv/alphaq_journal_evidence_gates.csv`
- `results/csv/alphaq_external_runs_consolidated.csv`
- `results/csv/alphaq_objective_selection_dataset.csv`
- `results/csv/alphaq_split_select_summary.csv`
- `results/csv/alphaq_journal_next_battery.csv`

## Current Scientific Interpretation

The current evidence supports this narrow claim:

> Target-dependent AlphaQ objective selection can improve T-count and/or QASM
> depth over the factor-count baseline on several external circuits.

It does not yet support a journal-ready scalability claim.

Open blockers:

- more external target coverage;
- more family diversity;
- formal verification of promoted candidates;
- a better scaling path for larger targets where full-action MILP becomes
  pathological.

## How To Keep The Repo Understandable

Use these rules when adding future work:

1. Add a new script to `scripts/` only if it is part of an active workflow.
2. If a script is exploratory and no longer active, move it to
   `archive/scripts/`.
3. Keep large generated directories under `results/` ignored.
4. Promote compact artifacts only when they are part of a documented result.
5. Update `docs/alphaq_journal_checkpoint.md` or create a new checkpoint before
   asking a third agent to evaluate the work.
6. Do not keep one-off GPT handoff folders in the repository root.

## Focused Validation

For the current journal-evidence layer:

```bash
cd /Users/caio/Deep-Learning/project
PYTHONPATH=.:external uv run pytest \
  tests/test_alphaq_journal_evidence.py \
  tests/test_alphaq_journal_battery_commands.py \
  tests/test_alphaq_split_select.py \
  tests/test_refresh_alphaq_journal_evidence.py \
  -q
```

For the full active suite:

```bash
uv run pytest tests -q
```
