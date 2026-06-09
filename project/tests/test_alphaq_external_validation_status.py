from __future__ import annotations

from scripts.analyze_alphaq_external_validation_status import status_rows
from scripts.analyze_alphaq_external_selector_transfer import REQUIRED_OBJECTIVES


def test_status_rows_requires_all_objectives_and_transfer() -> None:
    readiness = [
        {"target": "complete", "readiness_status": "ready-full-action"},
        {"target": "partial", "readiness_status": "ready-full-action"},
        {"target": "failed", "readiness_status": "ready-full-action"},
        {"target": "ready", "readiness_status": "ready-full-action"},
    ]
    decomposition = [
        *[
            {"target": "complete", "objective_variant": objective}
            for objective in sorted(REQUIRED_OBJECTIVES)
        ],
        {"target": "partial", "objective_variant": "factor_count"},
        {"target": "failed", "objective_variant": "factor_count"},
        {
            "target": "failed",
            "objective_variant": "factor_count_pair_cap",
            "execution_status": "failed",
        },
        *[
            {"target": "ready", "objective_variant": objective}
            for objective in sorted(REQUIRED_OBJECTIVES)
        ],
    ]
    transfer = [{"target": "complete"}]

    rows = {row["target"]: row for row in status_rows(
        readiness_rows=readiness,
        decomposition_rows=decomposition,
        transfer_rows=transfer,
    )}

    assert rows["complete"]["validation_status"] == "complete"
    assert rows["partial"]["validation_status"] == "pending-partial"
    assert rows["failed"]["validation_status"] == "failed-partial"
    assert rows["ready"]["validation_status"] == "ready-for-transfer"


def test_status_rows_ignores_non_full_action_readiness_targets() -> None:
    rows = status_rows(
        readiness_rows=[
            {"target": "external", "readiness_status": "ready-full-action"},
            {"target": "restricted", "readiness_status": "ready-restricted-action"},
        ],
        decomposition_rows=[],
        transfer_rows=[],
    )

    assert [row["target"] for row in rows] == ["external"]
