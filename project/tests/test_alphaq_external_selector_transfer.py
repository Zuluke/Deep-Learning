from __future__ import annotations

from scripts.analyze_alphaq_external_selector_transfer import complete_target_rows
from scripts.analyze_alphaq_external_selector_transfer import ok_rows
from scripts.analyze_alphaq_external_selector_transfer import REQUIRED_OBJECTIVES
from scripts.analyze_alphaq_external_selector_transfer import summary_rows


def test_ok_rows_skips_failed_external_decompositions() -> None:
    rows = [
        {"target": "a", "execution_status": "ok"},
        {"target": "b", "execution_status": "failed"},
        {"target": "c"},
    ]

    assert [row["target"] for row in ok_rows(rows)] == ["a", "c"]


def test_complete_target_rows_requires_all_objective_variants() -> None:
    rows = [
        *[
            {"target": "complete", "objective_variant": objective}
            for objective in sorted(REQUIRED_OBJECTIVES)
        ],
        {"target": "partial", "objective_variant": "factor_count"},
    ]

    complete = complete_target_rows(rows)

    assert {row["target"] for row in complete} == {"complete"}


def test_summary_rows_counts_external_transfer_matches() -> None:
    rows = [
        {
            "target": "external",
            "selected_objective": "factor_count_pair_cap",
            "exact_oracle_match": True,
            "oracle_equivalent": True,
            "override_accepted": False,
            "qasm_worse_than_baseline": False,
        }
    ]

    summary = summary_rows(rows)[0]

    assert summary["targets"] == 1
    assert summary["exact_oracle_matches"] == 1
    assert summary["oracle_equivalent_matches"] == 1
    assert summary["qasm_worse_than_baseline"] == 0
