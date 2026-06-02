from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_submission_gaps import build_gap_rows
from scripts.analyze_submission_gaps import block_phase_summary
from scripts.analyze_submission_gaps import priority_for_gap
from scripts.analyze_submission_gaps import verification_gap_kind


def test_submission_gap_kind_marks_circuit_with_no_equal_candidates_blocked() -> None:
    assert (
        verification_gap_kind(total=3, equal=0, best_all_status="inconclusive")
        == "verification-blocked-circuit"
    )
    assert (
        priority_for_gap(
            gap_kind="verification-blocked-circuit",
            total=3,
            equal=0,
            loco_test_candidates=None,
        )
        == "high"
    )


def test_submission_gap_rows_separate_verification_from_candidate_diversity() -> None:
    frontier_rows = [
        {
            "circuit_id": "blocked",
            "candidate_id": "blocked:combo0",
            "verification_status": "inconclusive",
            "structural_cost": "1.0",
            "tcount_after": "7",
        },
        {
            "circuit_id": "singleton",
            "candidate_id": "singleton:combo0",
            "verification_status": "equal",
            "structural_cost": "2.0",
            "tcount_after": "7",
        },
        {
            "circuit_id": "partial",
            "candidate_id": "partial:combo0",
            "verification_status": "equal",
            "structural_cost": "3.0",
            "tcount_after": "7",
        },
        {
            "circuit_id": "partial",
            "candidate_id": "partial:combo1",
            "verification_status": "inconclusive",
            "structural_cost": "4.0",
            "tcount_after": "7",
        },
    ]
    loco_rows = [
        {"circuit_id": "singleton", "num_test_candidates": "1"},
        {"circuit_id": "partial", "num_test_candidates": "2"},
    ]

    rows = {
        row["circuit_id"]: row
        for row in build_gap_rows(
            frontier_verification_rows=frontier_rows,
            loco_rows=loco_rows,
        )
    }

    assert rows["blocked"]["priority"] == "high"
    assert rows["blocked"]["gap_kind"] == "verification-blocked-circuit"
    assert rows["singleton"]["priority"] == "medium"
    assert rows["singleton"]["gap_kind"] == "verified-frontier"
    assert rows["partial"]["priority"] == "medium"
    assert rows["partial"]["gap_kind"] == "nonblocking-inconclusives"


def test_submission_gap_rows_surface_linear_phase_mismatch() -> None:
    frontier_rows = [
        {
            "circuit_id": "phasey",
            "candidate_id": "phasey:combo0",
            "verification_status": "inconclusive",
            "structural_cost": "1.0",
            "tcount_after": "7",
        }
    ]
    block_rows = [
        {
            "circuit_id": "phasey",
            "candidate_id": "phasey:combo0",
            "block_diagnostic_status": "sampled-columnwise-phase-mismatch",
            "monomial_phase_status": "ok",
            "monomial_output_mismatch_count": "0",
            "monomial_phase_delta_sign_only": "1",
            "monomial_phase_delta_global_only": "0",
            "monomial_phase_delta_degree": "1",
            "monomial_phase_delta_num_terms": "3",
        }
    ]

    rows = build_gap_rows(
        frontier_verification_rows=frontier_rows,
        loco_rows=[],
        block_rows=block_rows,
    )

    assert rows[0]["block_phase_summary"] == (
        "monomial-ok=1;output-match=1;sign-only=1;"
        "global-only=0;degrees=1;term-counts=3"
    )
    assert "linear input-dependent sign" in rows[0]["recommended_action"]


def test_block_phase_summary_aggregates_monomial_rows() -> None:
    summary = block_phase_summary(
        [
            {
                "monomial_phase_status": "ok",
                "monomial_output_mismatch_count": "0",
                "monomial_phase_delta_sign_only": "1",
                "monomial_phase_delta_global_only": "0",
                "monomial_phase_delta_degree": "1",
                "monomial_phase_delta_num_terms": "5",
            },
            {
                "monomial_phase_status": "ok",
                "monomial_output_mismatch_count": "0",
                "monomial_phase_delta_sign_only": "1",
                "monomial_phase_delta_global_only": "0",
                "monomial_phase_delta_degree": "1",
                "monomial_phase_delta_num_terms": "7",
            },
        ]
    )

    assert summary == (
        "monomial-ok=2;output-match=2;sign-only=2;"
        "global-only=0;degrees=1;term-counts=5,7"
    )
