from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.consolidate_split_reward_sweep import is_solved_row
from scripts.consolidate_split_reward_sweep import needs_backfill


def test_failed_solved_row_needs_materialization_backfill() -> None:
    row = {
        "status": "ok",
        "best_effective_t_cost": "10.0",
        "materialization_status": "failed",
    }

    assert is_solved_row(row)
    assert needs_backfill(row)


def test_not_solved_row_is_not_backfilled() -> None:
    row = {
        "status": "ok",
        "best_effective_t_cost": "",
        "materialization_status": "not-solved",
    }

    assert not is_solved_row(row)
    assert not needs_backfill(row)


def test_ok_materialization_is_not_backfilled() -> None:
    row = {
        "status": "ok",
        "best_effective_t_cost": "17.0",
        "materialization_status": "ok",
    }

    assert is_solved_row(row)
    assert not needs_backfill(row)
