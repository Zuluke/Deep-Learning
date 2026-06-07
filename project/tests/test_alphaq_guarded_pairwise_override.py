from __future__ import annotations

from scripts.analyze_alphaq_guarded_pairwise_override import baseline_objective_by_target
from scripts.analyze_alphaq_guarded_pairwise_override import reject_reason
from scripts.analyze_alphaq_guarded_pairwise_override import summary_rows


def test_baseline_objective_uses_min_factor_count_with_paircap_tie() -> None:
    rows = [
        {"target": "toy", "objective_variant": "factor_count", "factor_count": "3"},
        {"target": "toy", "objective_variant": "factor_count_pair_cap", "factor_count": "3"},
        {"target": "toy", "objective_variant": "mixed_pair", "factor_count": "2"},
    ]

    assert baseline_objective_by_target(rows)["toy"] == "mixed_pair"

    tied = [row for row in rows if row["objective_variant"] != "mixed_pair"]
    assert baseline_objective_by_target(tied)["toy"] == "factor_count_pair_cap"


def test_reject_reason_distinguishes_same_objective_and_qasm_worse() -> None:
    assert reject_reason("a", "a", 2, 1) == "same-objective"
    assert reject_reason("a", "b", 2, 1) == "qasm-worse"
    assert reject_reason("a", "b", 1, 2) == "not-accepted"


def test_summary_rows_counts_guarded_overrides_and_equivalence() -> None:
    details = [
        {
            "exact_oracle_match": True,
            "oracle_equivalent": True,
            "override_accepted": True,
            "tcount_ratio": "1",
            "primary_ratio": "0.9",
            "qasm_ratio": "1",
            "target": "a",
            "selected_objective": "factor_count",
        },
        {
            "exact_oracle_match": False,
            "oracle_equivalent": True,
            "override_accepted": False,
            "tcount_ratio": "1",
            "primary_ratio": "1",
            "qasm_ratio": "1",
            "target": "b",
            "selected_objective": "factor_count_pair_cap",
        },
    ]

    summary = summary_rows(details)[0]

    assert summary["exact_oracle_matches"] == 1
    assert summary["oracle_equivalent_matches"] == 2
    assert summary["overrides_accepted"] == 1
    assert summary["joint_nonworse"] == 2
