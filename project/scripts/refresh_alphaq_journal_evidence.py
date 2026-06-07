from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RefreshStep:
    name: str
    script: str


REFRESH_STEPS = (
    RefreshStep("external run consolidation", "scripts/consolidate_alphaq_external_runs.py"),
    RefreshStep("objective-selection dataset", "scripts/build_alphaq_objective_selection_dataset.py"),
    RefreshStep("Split-Select evaluation", "scripts/analyze_alphaq_split_select.py"),
    RefreshStep("journal evidence audit", "scripts/analyze_alphaq_journal_evidence.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh all derived AlphaQ journal-evidence artifacts in dependency order."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for child scripts.",
    )
    return parser.parse_args()


def command_for_step(step: RefreshStep, python_bin: str) -> list[str]:
    return [python_bin, step.script]


def run_refresh(
    *,
    python_bin: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    for step in REFRESH_STEPS:
        cmd = command_for_step(step, python_bin)
        print(f"+ refresh {step.name}", flush=True)
        runner(cmd, cwd=PROJECT_ROOT, check=True, text=True)


def main() -> int:
    args = parse_args()
    run_refresh(python_bin=args.python_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
