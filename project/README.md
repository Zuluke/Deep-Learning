# AlphaTensor-Quantum / AlphaQ Split-Select

This project started as a reproduction and audit toolkit for **Quantum Circuit
Optimization with AlphaTensor**. The current active development has two
surfaces:

1. **Entrega/paper reproduction**: public AlphaTensor-Quantum decompositions,
   QASM reconstruction, structural metrics, and formal checks.
2. **AlphaQ Split-Select research**: AlphaQ-only objective selection over tensor
   decompositions, with external validation toward a possible journal-level
   result.

Start here:

- `docs/repository_map.md`: how the repository is organized now.
- `docs/alphaq_journal_checkpoint.md`: current scientific checkpoint and
  handoff text for a third reviewer/agent.
- `results/figures/alphaq_journal_current_state.png`: current one-plot summary
  of the AlphaQ Split-Select evidence.
- `results/README.md`: active result artifacts and what is generated.
- `scripts/README.md`: active script groups.

## Setup

```bash
cd /Users/caio/Deep-Learning/project
uv sync --group dev
```

For the upstream AlphaTensor-Quantum demo environment:

```bash
uv sync --group demo-cpu --group dev
```

## Core Commands

Paper/Entrega reproduction:

```bash
uv run python scripts/reproduce_paper_results.py
uv run python scripts/make_entrega1_outputs.py
```

Current AlphaQ journal-evidence refresh:

```bash
PYTHONPATH=.:external uv run python scripts/refresh_alphaq_journal_evidence.py
PYTHONPATH=.:external uv run python scripts/plot_alphaq_journal_state.py
```

Focused tests for the current AlphaQ evidence layer:

```bash
PYTHONPATH=.:external uv run pytest \
  tests/test_alphaq_journal_evidence.py \
  tests/test_alphaq_journal_battery_commands.py \
  tests/test_alphaq_split_select.py \
  tests/test_refresh_alphaq_journal_evidence.py \
  -q
```

Full active test suite:

```bash
uv run pytest tests -q
```

## Current Scientific State

The current AlphaQ result is promising but not journal-ready. The evidence says:

- Split-Select improves over the factor-count baseline on several external
  circuits.
- The best objective is target-dependent: sometimes `mixed_pair`, sometimes
  `factor_count_pair_cap`, sometimes plain `factor_count`.
- Larger full-action targets are beginning to fail or produce pathological
  candidates, so the next step should be a deliberate scaling experiment rather
  than another blind full-action rerun.

For the full explanation, use `docs/alphaq_journal_checkpoint.md`.

## Repository Hygiene

- Keep reusable code in `scripts/` or `src/`.
- Keep generated experiment directories under `results/`; promote only compact
  CSV/figure/report artifacts when they become part of the active story.
- Put historical one-off scripts in `archive/scripts/`.
- Do not keep GPT handoff packets in the repository root after they have been
  sent.
