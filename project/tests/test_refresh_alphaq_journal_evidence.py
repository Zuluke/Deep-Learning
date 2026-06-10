from __future__ import annotations

from scripts.refresh_alphaq_journal_evidence import REFRESH_STEPS
from scripts.refresh_alphaq_journal_evidence import command_for_step
from scripts.refresh_alphaq_journal_evidence import run_refresh


def test_refresh_runs_steps_in_dependency_order() -> None:
    calls = []

    def runner(cmd, **kwargs):
        calls.append((cmd, kwargs))

    run_refresh(python_bin="python-test", runner=runner)

    scripts = [call[0][1] for call in calls]
    assert scripts == [step.script for step in REFRESH_STEPS]
    assert scripts == [
        "scripts/consolidate_alphaq_external_runs.py",
        "scripts/build_alphaq_objective_selection_dataset.py",
        "scripts/analyze_alphaq_split_select.py",
        "scripts/train_alphaq_circuit_objective_selector.py",
        "scripts/analyze_alphaq_portfolio_budget.py",
        "scripts/analyze_alphaq_journal_evidence.py",
    ]
    assert all(call[1]["check"] is True for call in calls)


def test_command_for_step_uses_requested_python_binary() -> None:
    assert command_for_step(REFRESH_STEPS[0], "python-test") == [
        "python-test",
        "scripts/consolidate_alphaq_external_runs.py",
    ]
