# Paper Build Notes

The tensor-v3 guard table in `main.tex` is generated from the audited CSV rather
than edited by hand.

Regenerate the table and its manifest:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/export_paper_tensor_v3_guard_table.py
```

Check that the committed table artifacts are current:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/export_paper_tensor_v3_guard_table.py --check
```

Materialize the guarded tensor-v3 profile and its pure selection manifest:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/materialize_tensor_v3_guarded_profile.py
```

Regenerate the submission-readiness audit:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/audit_submission_readiness.py
```

Regenerate the submission-gap priority report:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/analyze_submission_gaps.py
```

Regenerate sampled postselection diagnostics for frontier candidates that formal
verification could not prove equal:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/diagnose_postselection_equivalence.py --max-columns 64
uv run python scripts/diagnose_block_replacement_equivalence.py --max-columns 64
uv run python scripts/analyze_submission_gaps.py
```

Build the manuscript:

```bash
cd /Users/caio/Deep-Learning/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript now reads its committed figures from `paper/imgs/` and its
generated guard table from `paper/tables/`. It should not depend on ignored
experiment figure directories.

Current build caveat: the draft still has unresolved bibliography warnings for
some background citations. The PDF is generated successfully, but the references
need a final bibliography pass before a polished submission.

The generated manifest is `paper/tables/tensor_v3_guard_table.json`; it records
the source CSV hash, generated table hash, guard thresholds, B/T/W counts, and
the selected table rows.

Older guarded selector manifests were archived under
`project/results/archive/old_experiments/`. Regenerating the guarded profile may
create a fresh top-level result directory; promote only the final report/table
artifacts back into the active result surface.

The readiness audit is written to
`project/results/reports/submission_readiness_audit.md` and summarizes which
claims are currently supported, which remain scoped, and which artifacts back
each decision.

The gap-priority report is written to
`project/results/reports/submission_gap_priorities.md` and separates verifier
coverage gaps from candidate-diversity gaps.

The postselection diagnostic is written to
`project/results/reports/postselection_equivalence_diagnostics.md`. It is a
sampled diagnostic only; formal claims still require `feynver` equality.
The block-replacement diagnostic is written to
`project/results/reports/block_replacement_equivalence_diagnostics.md` and
compares ressynthesized blocks directly against their compiled `cnotphase`
references.
