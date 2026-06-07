from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_external_selector_transfer import REQUIRED_OBJECTIVES
from scripts.run_best_objective_beam_ablation import read_csv


DEFAULT_READINESS_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_readiness.csv"
DEFAULT_DECOMPOSITION_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_external_validation.csv"
DEFAULT_TRANSFER_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_selector_transfer_details.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_status.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_external_validation_status.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit external-validation coverage target by target."
    )
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS_CSV)
    parser.add_argument("--decomposition-csv", type=Path, default=DEFAULT_DECOMPOSITION_CSV)
    parser.add_argument("--transfer-detail-csv", type=Path, default=DEFAULT_TRANSFER_DETAIL_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def status_rows(
    *,
    readiness_rows: list[dict[str, str]],
    decomposition_rows: list[dict[str, str]],
    transfer_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    targets = [
        row
        for row in readiness_rows
        if row.get("readiness_status") == "ready-full-action"
    ]
    transfer_targets = {row["target"] for row in transfer_rows if row.get("target")}
    rows = []
    for target_row in targets:
        target = target_row["target"]
        by_objective = {
            row.get("objective_variant", ""): row
            for row in decomposition_rows
            if row.get("target") == target
        }
        ok = sorted(
            objective
            for objective in REQUIRED_OBJECTIVES
            if objective_status(by_objective.get(objective)) == "ok"
        )
        failed = sorted(
            objective
            for objective in REQUIRED_OBJECTIVES
            if objective_status(by_objective.get(objective)) == "failed"
        )
        missing = sorted(REQUIRED_OBJECTIVES - set(ok) - set(failed))
        transfer_status = "ok" if target in transfer_targets else "missing"
        validation_status = classify_target_status(ok, failed, missing, transfer_status)
        rows.append(
            {
                "target": target,
                "family": target_row.get("family", ""),
                "tensor_size": target_row.get("tensor_size", ""),
                "completed_objectives": ",".join(ok),
                "failed_objectives": ",".join(failed),
                "missing_objectives": ",".join(missing),
                "transfer_status": transfer_status,
                "validation_status": validation_status,
                "next_action": next_action(validation_status, failed, missing),
            }
        )
    return rows


def objective_status(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing"
    if row.get("execution_status", "ok") == "ok":
        return "ok"
    return "failed"


def classify_target_status(
    ok: list[str],
    failed: list[str],
    missing: list[str],
    transfer_status: str,
) -> str:
    if len(ok) == len(REQUIRED_OBJECTIVES) and transfer_status == "ok":
        return "complete"
    if failed:
        return "failed-partial"
    if ok and missing:
        return "pending-partial"
    if len(ok) == len(REQUIRED_OBJECTIVES):
        return "ready-for-transfer"
    return "pending"


def next_action(validation_status: str, failed: list[str], missing: list[str]) -> str:
    if validation_status == "complete":
        return "Use in external selector-transfer evidence."
    if validation_status == "ready-for-transfer":
        return "Run beam grid and guarded selector-transfer analysis."
    if validation_status == "failed-partial":
        return f"Rerun failed objectives with longer MILP budget: {', '.join(failed)}."
    if validation_status == "pending-partial":
        return f"Complete missing objectives without fallback: {', '.join(missing)}."
    return "Run the external-validation pipeline."


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "family",
        "tensor_size",
        "completed_objectives",
        "failed_objectives",
        "missing_objectives",
        "transfer_status",
        "validation_status",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["validation_status"] == "complete"]
    pending = [row for row in rows if row["validation_status"] != "complete"]
    lines = [
        "# AlphaQ external validation status",
        "",
        f"CSV: `{csv_path}`.",
        "",
        "This report audits the immediate full-action external validation queue. A target is counted as complete only after all three objective variants are materialized and the guarded selector-transfer analysis includes that target.",
        "",
        "## Bottom line",
        "",
        f"Completed external targets: {len(completed)}/{len(rows)}.",
        f"Pending external targets: {len(pending)}/{len(rows)}.",
        "",
        "| target | status | completed objectives | failed objectives | missing objectives | transfer | next action |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {status} | {completed} | {failed} | {missing} | {transfer} | {action} |".format(
                target=row["target"],
                status=row["validation_status"],
                completed=row["completed_objectives"] or "-",
                failed=row["failed_objectives"] or "-",
                missing=row["missing_objectives"] or "-",
                transfer=row["transfer_status"],
                action=row["next_action"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = status_rows(
        readiness_rows=read_csv(args.readiness_csv),
        decomposition_rows=read_csv(args.decomposition_csv),
        transfer_rows=read_csv(args.transfer_detail_csv),
    )
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
