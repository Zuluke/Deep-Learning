from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json


DEFAULT_FRONTIER_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT
    / "verification"
    / "frontier"
    / "alphaq_candidate_frontier_verification.csv"
)
DEFAULT_LOCO_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_loco_eval.csv"
DEFAULT_POSTSELECTION_DIAGNOSTICS_CSV = (
    DEFAULT_CSV_ROOT / "postselection_equivalence_diagnostics.csv"
)
DEFAULT_BLOCK_REPLACEMENT_DIAGNOSTICS_CSV = (
    DEFAULT_CSV_ROOT / "block_replacement_equivalence_diagnostics.csv"
)
DEFAULT_OUTPUT_CSV = DEFAULT_CSV_ROOT / "submission_gap_priorities.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_CSV_ROOT / "submission_gap_priorities.json"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "submission_gap_priorities.md"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    numeric = coerce_float(value)
    return None if numeric is None else int(numeric)


def rank_cost(row: dict[str, Any]) -> tuple[float, float]:
    structural = coerce_float(row.get("structural_cost"))
    tcount = coerce_float(row.get("tcount_after"))
    return (
        float("inf") if structural is None else structural,
        float("inf") if tcount is None else tcount,
    )


def verification_gap_kind(total: int, equal: int, best_all_status: str | None) -> str:
    if total == 0:
        return "missing-frontier"
    if equal == 0:
        return "verification-blocked-circuit"
    if equal == total:
        return "verified-frontier"
    if best_all_status != "equal":
        return "oracle-possibly-blocked"
    return "nonblocking-inconclusives"


def priority_for_gap(
    *,
    gap_kind: str,
    total: int,
    equal: int,
    loco_test_candidates: int | None,
) -> str:
    if gap_kind == "verification-blocked-circuit":
        return "high"
    if gap_kind == "oracle-possibly-blocked":
        return "high"
    if loco_test_candidates is not None and loco_test_candidates <= 1:
        return "medium"
    if total > equal:
        return "medium"
    return "low"


def diagnostic_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("diagnostic_status") or "not-run"
        counts[status] = counts.get(status, 0) + 1
    return counts


def diagnostic_counts_text(counts: dict[str, int]) -> str:
    return ";".join(f"{key}={counts[key]}" for key in sorted(counts))


def block_phase_summary(rows: list[dict[str, str]]) -> str:
    monomial_rows = [row for row in rows if row.get("monomial_phase_status") == "ok"]
    if not monomial_rows:
        return ""
    output_match = sum(
        1 for row in monomial_rows if coerce_int(row.get("monomial_output_mismatch_count")) == 0
    )
    sign_only = sum(
        1 for row in monomial_rows if coerce_int(row.get("monomial_phase_delta_sign_only")) == 1
    )
    global_only = sum(
        1 for row in monomial_rows if coerce_int(row.get("monomial_phase_delta_global_only")) == 1
    )
    degrees = sorted(
        {
            degree
            for row in monomial_rows
            if (degree := coerce_int(row.get("monomial_phase_delta_degree"))) is not None
        }
    )
    term_counts = sorted(
        {
            count
            for row in monomial_rows
            if (count := coerce_int(row.get("monomial_phase_delta_num_terms"))) is not None
        }
    )
    parts = [
        f"monomial-ok={len(monomial_rows)}",
        f"output-match={output_match}",
        f"sign-only={sign_only}",
        f"global-only={global_only}",
    ]
    if degrees:
        parts.append("degrees=" + ",".join(str(item) for item in degrees))
    if term_counts:
        parts.append("term-counts=" + ",".join(str(item) for item in term_counts))
    return ";".join(parts)


def action_for_gap(
    gap_kind: str,
    loco_test_candidates: int | None,
    postselection_counts: dict[str, int] | None = None,
    block_counts: dict[str, int] | None = None,
    block_phase: str = "",
) -> str:
    postselection_counts = postselection_counts or {}
    block_counts = block_counts or {}
    if block_counts.get("sampled-global-match", 0) > 0:
        return "block replacement matches locally; focus on full-circuit postselection/formal proof"
    if block_counts.get("sampled-columnwise-phase-mismatch", 0) > 0:
        if "sign-only" in block_phase and "degrees=1" in block_phase:
            return "candidate block differs by a linear input-dependent sign; inspect phase convention or explicit diagonal correction"
        return "candidate block has input-dependent phase; inspect decomposition phase convention"
    if postselection_counts.get("sampled-global-match", 0) > 0:
        return "full candidate has sampled postselection match; prioritize formal proof"
    if gap_kind == "verification-blocked-circuit":
        if postselection_counts:
            return "sampled postselection found no global match; use block diagnostic to distinguish proof issue from candidate issue"
        return "resolve formal verification or exclude from learning claim until equal candidates exist"
    if gap_kind == "oracle-possibly-blocked":
        if postselection_counts:
            return "resolve inconclusive best candidates; sampled diagnostics can triage likely invalid candidates"
        return "resolve inconclusive best candidates because the oracle may be censored"
    if loco_test_candidates is not None and loco_test_candidates <= 1:
        return "generate or verify additional candidates before treating LOCO as strong generalization"
    if gap_kind == "nonblocking-inconclusives":
        return "optional cleanup; current best verified candidate is already equal"
    if gap_kind == "verified-frontier":
        return "no immediate action"
    return "inspect missing frontier artifacts"


def build_gap_rows(
    *,
    frontier_verification_rows: list[dict[str, str]],
    loco_rows: list[dict[str, str]],
    postselection_rows: list[dict[str, str]] | None = None,
    block_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    loco_by_circuit = {row["circuit_id"]: row for row in loco_rows if row.get("circuit_id")}
    postselection_by_circuit: dict[str, list[dict[str, str]]] = {}
    for row in postselection_rows or []:
        if row.get("circuit_id"):
            postselection_by_circuit.setdefault(row["circuit_id"], []).append(row)
    postselection_by_candidate = {
        row["candidate_id"]: row
        for row in postselection_rows or []
        if row.get("candidate_id")
    }
    block_by_circuit: dict[str, list[dict[str, str]]] = {}
    for row in block_rows or []:
        if row.get("circuit_id"):
            block_by_circuit.setdefault(row["circuit_id"], []).append(row)
    block_by_candidate: dict[str, list[dict[str, str]]] = {}
    for row in block_rows or []:
        if row.get("candidate_id"):
            block_by_candidate.setdefault(row["candidate_id"], []).append(row)
    circuit_ids = sorted(
        {
            *(row.get("circuit_id", "") for row in frontier_verification_rows),
            *(row.get("circuit_id", "") for row in loco_rows),
        }
        - {""},
        key=natural_sort_key,
    )
    output_rows: list[dict[str, Any]] = []
    for circuit_id in circuit_ids:
        subset = [
            row
            for row in frontier_verification_rows
            if row.get("circuit_id") == circuit_id
        ]
        equal_rows = [row for row in subset if row.get("verification_status") == "equal"]
        non_equal_rows = [
            row for row in subset if row.get("verification_status") != "equal"
        ]
        best_all = min(subset, key=rank_cost) if subset else None
        best_equal = min(equal_rows, key=rank_cost) if equal_rows else None
        best_non_equal = min(non_equal_rows, key=rank_cost) if non_equal_rows else None
        postselection_subset = postselection_by_circuit.get(circuit_id, [])
        postselection_status_counts = diagnostic_counts(postselection_subset)
        block_subset = block_by_circuit.get(circuit_id, [])
        block_status_counts = diagnostic_counts(
            [
                {
                    **row,
                    "diagnostic_status": row.get("block_diagnostic_status"),
                }
                for row in block_subset
            ]
        )
        phase_summary = block_phase_summary(block_subset)
        loco_row = loco_by_circuit.get(circuit_id)
        loco_test_candidates = (
            coerce_int(loco_row.get("num_test_candidates")) if loco_row else None
        )
        gap_kind = verification_gap_kind(
            total=len(subset),
            equal=len(equal_rows),
            best_all_status=best_all.get("verification_status") if best_all else None,
        )
        priority = priority_for_gap(
            gap_kind=gap_kind,
            total=len(subset),
            equal=len(equal_rows),
            loco_test_candidates=loco_test_candidates,
        )
        best_equal_cost = coerce_float(best_equal.get("structural_cost")) if best_equal else None
        best_all_cost = coerce_float(best_all.get("structural_cost")) if best_all else None
        output_rows.append(
            {
                "circuit_id": circuit_id,
                "priority": priority,
                "gap_kind": gap_kind,
                "frontier_candidates": len(subset),
                "equal_candidates": len(equal_rows),
                "non_equal_candidates": len(non_equal_rows),
                "loco_test_candidates": loco_test_candidates,
                "best_all_candidate_id": best_all.get("candidate_id") if best_all else None,
                "best_all_status": best_all.get("verification_status") if best_all else None,
                "best_all_structural_cost": best_all_cost,
                "best_equal_candidate_id": best_equal.get("candidate_id") if best_equal else None,
                "best_equal_structural_cost": best_equal_cost,
                "best_non_equal_candidate_id": (
                    best_non_equal.get("candidate_id") if best_non_equal else None
                ),
                "best_non_equal_postselection_status": (
                    postselection_by_candidate.get(best_non_equal.get("candidate_id"), {}).get(
                        "diagnostic_status"
                    )
                    if best_non_equal
                    else None
                ),
                "best_non_equal_block_status": (
                    diagnostic_counts(
                        [
                            {
                                **row,
                                "diagnostic_status": row.get("block_diagnostic_status"),
                            }
                            for row in block_by_candidate.get(
                                best_non_equal.get("candidate_id"), []
                            )
                        ]
                    )
                    if best_non_equal
                    else None
                ),
                "postselection_diagnostic_counts": diagnostic_counts_text(
                    postselection_status_counts
                ),
                "block_diagnostic_counts": diagnostic_counts_text(block_status_counts),
                "block_phase_summary": phase_summary,
                "best_non_equal_structural_cost": (
                    coerce_float(best_non_equal.get("structural_cost"))
                    if best_non_equal
                    else None
                ),
                "equal_cost_gap_vs_best_all": (
                    None
                    if best_equal_cost is None or best_all_cost is None
                    else best_equal_cost - best_all_cost
                ),
                "recommended_action": action_for_gap(
                    gap_kind,
                    loco_test_candidates,
                    postselection_status_counts,
                    block_status_counts,
                    phase_summary,
                ),
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        output_rows,
        key=lambda row: (
            priority_order.get(str(row["priority"]), 99),
            natural_sort_key(str(row["circuit_id"])),
        ),
    )


def write_report(rows: list[dict[str, Any]], report_path: Path, csv_path: Path) -> Path:
    priority_counts: dict[str, int] = {}
    for row in rows:
        priority = str(row["priority"])
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    table_lines = [
        "| circuit | priority | gap kind | equal/frontier | LOCO candidates | diagnostics | phase summary | best all | best equal | action |",
        "|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        best_all = (
            ""
            if not row.get("best_all_candidate_id")
            else f"`{row['best_all_candidate_id']}` ({row['best_all_status']})"
        )
        best_equal = (
            ""
            if not row.get("best_equal_candidate_id")
            else f"`{row['best_equal_candidate_id']}`"
        )
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['circuit_id']}`",
                    f"`{row['priority']}`",
                    f"`{row['gap_kind']}`",
                    f"{row['equal_candidates']}/{row['frontier_candidates']}",
                    "" if row["loco_test_candidates"] is None else str(row["loco_test_candidates"]),
                    "; ".join(
                        item
                        for item in (
                            (
                                "post:"
                                + str(row.get("postselection_diagnostic_counts"))
                                if row.get("postselection_diagnostic_counts")
                                else ""
                            ),
                            (
                                "block:"
                                + str(row.get("block_diagnostic_counts"))
                                if row.get("block_diagnostic_counts")
                                else ""
                            ),
                        )
                        if item
                    ),
                    str(row.get("block_phase_summary") or ""),
                    best_all,
                    best_equal,
                    str(row["recommended_action"]),
                ]
            )
            + " |"
        )
    text = [
        "# Submission Gap Priorities",
        "",
        f"- Gap CSV: `{csv_path}`.",
        (
            "- Priority counts: "
            + ", ".join(
                f"{key}={priority_counts.get(key, 0)}"
                for key in ("high", "medium", "low")
            )
            + "."
        ),
        "",
        "## Reading",
        "",
        (
            "The current bottlenecks split into two different mechanisms. "
            "`cuccaro_adder_n3` and `vbe_adder_3` are verification-blocked: their "
            "frontier candidates exist, but none are formally equal in the frontier "
            "audit. `hamming_weight_n4` and `hamming_weight_n5` are not verification "
            "failures; they are candidate-diversity limitations because the retained "
            "frontier has a single verified candidate."
        ),
        (
            "The block-replacement diagnostic sharpens the high-priority cases. "
            "`cuccaro_adder_n3` has a locally matching replacement block, so its "
            "remaining problem is full-circuit postselection/formal proof. "
            "`vbe_adder_3` has locally column-matching replacements with "
            "a sign-only, degree-1 input-dependent phase, so its issue is now "
            "localized to a linear phase convention or missing diagonal correction."
        ),
        "",
        "## Prioritized Circuits",
        "",
        *table_lines,
        "",
        "## Concrete Next Actions",
        "",
        "1. Attack `vbe_adder_3` first by tracing the linear input-dependent sign terms in the block replacements and checking whether the compiler convention expects an omitted diagonal correction.",
        "2. Then inspect `cuccaro_adder_n3` at the full-circuit verification layer, because its replacement block samples as a global match.",
        "3. Generate or verify more hamming-weight candidates before using those circuits as strong LOCO evidence.",
        "4. Treat `qft_4` inconclusives as lower priority because the best structural candidate is already formally equal.",
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prioritize remaining submission gaps by frontier verification and LOCO coverage."
    )
    parser.add_argument(
        "--frontier-verification-csv",
        type=Path,
        default=DEFAULT_FRONTIER_VERIFICATION_CSV,
    )
    parser.add_argument("--loco-csv", type=Path, default=DEFAULT_LOCO_CSV)
    parser.add_argument(
        "--postselection-diagnostics-csv",
        type=Path,
        default=DEFAULT_POSTSELECTION_DIAGNOSTICS_CSV,
    )
    parser.add_argument(
        "--block-replacement-diagnostics-csv",
        type=Path,
        default=DEFAULT_BLOCK_REPLACEMENT_DIAGNOSTICS_CSV,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_gap_rows(
        frontier_verification_rows=load_csv_rows(args.frontier_verification_csv),
        loco_rows=load_csv_rows(args.loco_csv),
        postselection_rows=load_csv_rows(args.postselection_diagnostics_csv),
        block_rows=load_csv_rows(args.block_replacement_diagnostics_csv),
    )
    write_csv_rows(rows, args.output_csv)
    priority_counts: dict[str, int] = {}
    for row in rows:
        priority = str(row["priority"])
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    write_json(
        {
            "frontier_verification_csv": str(args.frontier_verification_csv),
            "loco_csv": str(args.loco_csv),
            "postselection_diagnostics_csv": str(args.postselection_diagnostics_csv),
            "block_replacement_diagnostics_csv": str(
                args.block_replacement_diagnostics_csv
            ),
            "output_csv": str(args.output_csv),
            "report_path": str(args.report_path),
            "priority_counts": priority_counts,
            "num_rows": len(rows),
        },
        args.output_json,
    )
    write_report(rows, args.report_path, args.output_csv)
    print(
        json.dumps(
            {
                "num_rows": len(rows),
                "output_csv": str(args.output_csv),
                "priority_counts": priority_counts,
                "report_path": str(args.report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
