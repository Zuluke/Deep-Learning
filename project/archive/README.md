# Archive

This folder preserves code and tests from earlier project stages without keeping
them in the active top-level workflow.

- `scripts/`: exploratory analyses, old reproduction wrappers, and one-off
  diagnostic commands.
- `tests/`: tests that belong to archived scripts.

Archived files are kept for reference. They are not expected to be exercised by
the active `uv run pytest tests -q` suite unless they are restored to
`project/scripts/`.
